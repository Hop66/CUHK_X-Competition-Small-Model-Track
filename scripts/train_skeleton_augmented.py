#!/usr/bin/env python3
"""CUHK-X —— 骨架混合增强训练：NTU 清洗分类(A001-A040 中 13 类语义一致)直接混入我们 40 类训练集。

对比 multi-task：单一 40 类头（无任务干扰），NTU 样本标签=映射后的我们类，
数据利用率 100%（每 epoch 全量参与）。

对照：骨架基线（MB_lite, 2891 样本）fold0 ~0.45-0.51；multi-task 0.5264
判读：fold0 ours_val 显著 > 0.5264 → 混合增强有效 → 全量+骨架融合

用法:
    python scripts/train_skeleton_augmented.py --fold 0 \
        --ntu_root data/external/ntu/ntu60 --pretrained weights/mb_lite_latest_epoch.bin
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset import build_balanced_sampler
from src.ntu_dataset import NTUSkeletonDataset
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.split import split_by_subject

from train_skeleton_motionbert import ActionNet, BASE, load_pretrained_backbone
from src.motionbert.action_net import ActionNetDualExpert


def dual_expert_loss(model, x, y, crit, lam_cpl=0.5, lam_nor=0.1):
    """BHaRNet III-B/IV-D：L = CE(u)+CE(l)+λcpl·CE(u+l 之和) + λnor·CE(NoisyOR)。"""
    import torch.nn.functional as F
    lo_u, lo_l, lo_s = model(x, need_components=True)
    loss = crit(lo_u, y) + crit(lo_l, y) + lam_cpl * crit(lo_s, y)
    if lam_nor > 0:
        pu = F.softmax(lo_u, dim=-1)
        pl = F.softmax(lo_l, dim=-1)
        p_nor = 1.0 - (1.0 - pu) * (1.0 - pl)   # 至少一专家置信
        loss = loss + lam_nor * crit(p_nor, y)
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device).unsqueeze(1)
        y = y.to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]                      # 双专家 → 确定性 logit 和
        correct += (out.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ntu_root", default="data/external/ntu/ntu60")
    ap.add_argument("--pretrained", default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_aug")
    ap.add_argument("--clean", action="store_true",
                    help="对原始 3D 骨架做时间平滑/坏帧修复/零帧插值（对照 0.5447 基线）")
    ap.add_argument("--part_aware", action="store_true",
                    help="可学习部位权重（每关节一个，初始按任务先验，动作由部位主导）")
    ap.add_argument("--task_prior_part", action="store_true",
                    help="part-aware 用 40 类任务先验初始化部位权重（26/40 类是手/上肢主导）而非全 1")
    ap.add_argument("--norm", type=str, default="shoulder", choices=["shoulder", "torso"],
                    help="骨架归一化：shoulder=肩宽；torso=骨盆-颈（0.711 notebook 建议，躯干比肩宽稳）")
    ap.add_argument("--sched", type=str, default="cosine", choices=["cosine", "step"],
                    help="cosine=原版（0.5447 基线来源）；step=StepLR(step10,0.5)（run_all 已跑 0.5113，验证 cosine 更好）")
    ap.add_argument("--dual_expert", action="store_true",
                    help="BHaRNet-B 双专家：上肢头(胸脊+头颈+双臂 9 关节, 手/上肢 28 类) + 躯干下肢头"
                         "（8 关节）。损失=CE(u)+CE(l)+λcpl·CE(和)+λnor·CE(NoisyOR)。对照 0.5447")
    ap.add_argument("--lam_cpl", type=float, default=0.5, help="双专家互补损失权重")
    ap.add_argument("--lam_nor", type=float, default=0.1, help="双专家 Noisy-OR 正则权重（0 关闭）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    # 我们骨架 + NTU 映射样本（单一 40 类）
    tr_ours = MotionBertSkeletonDataset(tr_clips, args.num_frames, True, input3d=True,
                                        clean=args.clean, norm=args.norm)
    tr_ntu = NTUSkeletonDataset(args.ntu_root, args.num_frames, True, label_ours=True)
    va_ours = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=True,
                                        clean=args.clean, norm=args.norm)

    combined = ConcatDataset([tr_ours, tr_ntu])
    # 全部训练标签（ours 40 类 + NTU 映射后的我们类，从文件列表直接推导）
    import re
    from src.ntu_dataset import FNAME_RE, NTU_TO_OURS
    ntu_labels = [NTU_TO_OURS[int(FNAME_RE.search(f.name).group(5))]
                  for f in tr_ntu.files]
    all_labels = [c.action_id for c in tr_clips] + ntu_labels
    print(f"ours={len(tr_ours)} ntu_mapped={len(tr_ntu)} 混合训练集={len(combined)} "
          f"（NTU 覆盖类: {sorted(set(ntu_labels))}）", flush=True)

    sampler = build_balanced_sampler(all_labels)
    tr_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ours, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
    if args.dual_expert:
        model = ActionNetDualExpert(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                                    dropout_ratio=0.5, hidden_dim=512, num_joints=17).to(device)
        print(f"ActionNetDualExpert params ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M"
              f" dual upper(9j)+lower(8j) clean={args.clean} norm={args.norm}", flush=True)
    else:
        model = ActionNet(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                          dropout_ratio=0.5, version="class", hidden_dim=512,
                          num_joints=17, part_aware=args.part_aware).to(device)
        print(f"ActionNet params ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M"
              f" part_aware={args.part_aware} clean={args.clean} norm={args.norm}", flush=True)
    if args.part_aware and args.task_prior_part:
        # H3.6M-17 关节: 0骨盆, 1-6腿, 7-10躯干头, 11-13左臂, 14-16右臂
        # 40 类中 26 类手/上肢主导（手嘴/手物/精细），9 类下肢，3 类静坐 → 任务先验初始化
        prior = torch.tensor([1.0, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05,
                              0.95, 0.95, 0.95, 0.95,
                              1.12, 1.12, 1.12, 1.12, 1.12, 1.12],
                             device=device)
        with torch.no_grad():
            model.part_logits.data = torch.atanh((prior - 1.0).clamp(-0.99, 0.99))
        print("part-aware v2: 任务先验初始化（臂 1.12× / 腿 1.05× / 头躯干 0.95×）", flush=True)

    if isinstance(model, ActionNetDualExpert):
        head_params = list(model.head_upper.parameters()) + list(model.head_lower.parameters())
    else:
        head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ] + (([{"params": [model.part_logits], "lr": args.lr_head}]
          if (not isinstance(model, ActionNetDualExpert) and args.part_aware) else [])),
        lr=args.lr_backbone, weight_decay=0.01)
    # cosine=原版（0.5447 基线来源，对骨架 transformer 更好）；step=StepLR（run_all 基线 0.5113，验证更差）
    if args.sched == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"augmented_fold{args.fold}.pth"
    best, no_improve = 0.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device).unsqueeze(1), y.to(device)
            opt.zero_grad()
            if isinstance(model, ActionNetDualExpert):
                loss = dual_expert_loss(model, x, y, crit, args.lam_cpl, args.lam_nor)
            else:
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate(model, va_loader, device)
        if acc > best:
            best, no_improve = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"ours_val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= args.patience:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== 混合增强 fold{args.fold} best ours_val = {best:.4f} "
          f"（对照 multi-task 0.5264 / 基线 0.51；显著 >0.53 则 NTU 混合有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
