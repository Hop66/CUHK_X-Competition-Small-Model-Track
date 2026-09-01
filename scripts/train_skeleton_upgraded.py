#!/usr/bin/env python3
"""CUHK-X —— 骨架升级训练：AimCLR 式极端增强 + 24 帧 + NTU 混合（特化跨被试）。

方法本质：骨架动作识别 = 关节相对运动模式（跨被试不变）。
特化到我们的任务（cross-subject、2891 小数据、MMPose 提取骨架）：
  1. 骨架跨被试差异 = 体型/速度/关节抖动/视角 → 增强模拟这些变化（AimCLR 式）：
     时序放缩（模拟动作快慢）、关节遮蔽（模拟丢失/遮挡）、高斯噪声（MMPose 抖动）、剪切
  2. 24 帧序列（clip ~29 帧，覆盖更完整动作，比 16 帧更充分）
  3. 可选 NTU 清洗分类混合（13 类映射，用户已验证 0.5447 有效）

对照：骨架基线 0.51 / multi-task 0.5264 / 混合增强 0.5447
判读：fold0 > 0.5447 显著 → 骨架升级有效

用法:
    python scripts/train_skeleton_upgraded.py --fold 0 \
        [--ntu_root data/external/ntu/ntu60] --pretrained weights/mb_lite_latest_epoch.bin
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset import build_balanced_sampler
from src.ntu_dataset import NTUSkeletonDataset
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.split import split_by_subject

from train_skeleton_motionbert import ActionNet, BASE, load_pretrained_backbone


def aimclr_augment(x, p=0.8):
    """AimCLR 式极端增强（特化：模拟跨被试的体型/速度/噪声/遮挡差异）。

    x: [T,17,3] 归一化后的骨架（训练时，在 device 上应用）。
    1) 时序放缩 0.8-1.2（动作快慢差异）
    2) 关节遮蔽 20%（模拟关节丢失/遮挡）
    3) 高斯噪声 σ=0.02（MMPose 估计抖动）
    4) xy 剪切 ±0.2（视角/体型）
    """
    if torch.rand(1).item() > p:
        return x
    B, T, J, C = x.shape if x.dim() == 4 else (1, *x.shape)
    x = x.view(B, T, J, C)
    # 1) 时序放缩：时间维随机 0.8-1.2 重采样，再 resize 回固定 T（保证输出形状不变）
    if torch.rand(1).item() < 0.5:
        s = float(torch.empty(1).uniform_(0.8, 1.2))
        x = x.permute(0, 3, 2, 1)                       # [B,C=3,J,T]
        x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(J, T), mode="bilinear", align_corners=False)  # 强制回 T
        x = x.permute(0, 3, 2, 1)                       # [B,T,J,3]
    # 2) 关节遮蔽：每样本随机遮 ~20% 关节
    if torch.rand(1).item() < 0.5:
        mask = torch.rand(B, J, device=x.device) > 0.2
        x = x * mask[:, None, :, None]
    # 3) 高斯噪声
    if torch.rand(1).item() < 0.6:
        x = x + torch.randn_like(x) * 0.02
    # 4) xy 剪切（模拟视角/体型横向差异）
    if torch.rand(1).item() < 0.5:
        sh = float(torch.empty(1).uniform_(-0.2, 0.2))
        x = x + torch.tensor([0.0, sh, 0.0], device=x.device)
    return x.view(B, T, J, C)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device).unsqueeze(1)
        y = y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ntu_root", default="", help="非空则混入 NTU 清洗分类样本")
    ap.add_argument("--pretrained", default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=5e-4,
                    help="backbone lr（MotionBERT 是 transformer，可比 IG65M 大）")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--aug_strength", type=float, default=0.8, help="AimCLR 增强概率")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_upgraded")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    tr_ours = MotionBertSkeletonDataset(tr_clips, args.num_frames, True, input3d=True)
    va_ours = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=True)
    all_labels = [c.action_id for c in tr_clips]

    datasets = [tr_ours]
    if args.ntu_root:
        tr_ntu = NTUSkeletonDataset(args.ntu_root, args.num_frames, True, label_ours=True)
        datasets.append(tr_ntu)
        from src.ntu_dataset import FNAME_RE, NTU_TO_OURS
        import re
        ntu_labels = [NTU_TO_OURS[int(FNAME_RE.search(f.name).group(5))]
                      for f in tr_ntu.files]
        all_labels += ntu_labels
        print(f"NTU 混合: {len(tr_ntu)} 样本", flush=True)
    combined = ConcatDataset(datasets)
    print(f"训练集: ours={len(tr_ours)} 共={len(combined)}（24帧 + AimCLR 增强 p={args.aug_strength}）", flush=True)

    sampler = build_balanced_sampler(all_labels)
    tr_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ours, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
    model = ActionNet(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                      dropout_ratio=0.5, version="class", hidden_dim=512,
                      num_joints=17).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], lr=args.lr_backbone, weight_decay=0.01)
    # StepLR 不衰减到 0（CosineAnnealing 会衰减到 ~0 导致后 30ep 不学习）
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"upgraded_fold{args.fold}.pth"
    best, no_improve = 0.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x = aimclr_augment(x.to(device), args.aug_strength).unsqueeze(1)
            y = y.to(device)
            opt.zero_grad()
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

    print(f"== 骨架升级 fold{args.fold} best ours_val = {best:.4f} "
          f"（对照混合增强 0.5447；显著 >0.55 则升级有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
