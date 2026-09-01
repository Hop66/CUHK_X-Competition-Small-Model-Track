#!/usr/bin/env python3
"""CUHK-X —— Step 3：双视角骨架双分支训练（共享 backbone + 双输入头）

热像姿态桥（2026-08-23 顶会方法论定稿）：
  分支1 = 热像 2D 骨架 [x,y,conf]（KeypointRCNN 零样本迁移提取，outputs/thermal_skeleton/*.npz）
  分支2 = NYX 3D 骨架 [x,y,z]（Skeleton predictions，原生 3D 匹配 model_pos 3D 预训练）
  共享 DSTformer encoder + 双输入头（预训练 joints_embed 初始化）+ 双分类头 + logits 加权融合
  3D 分支绕竖直轴随机旋转增强（补两相机视点差，NTU cross-view 标准）

用法:
    python scripts/train_dual_branch.py --pretrained weights/mb_lite_latest_epoch.bin \
        --folds 3 --balanced --save_dir outputs/dual_branch
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.dual_branch import DualBranchActionNet
from src.dual_dataset import DualSkeletonDataset, build_dual_pairs
from src.motionbert.dstformer import DSTformer
from src.split import split_by_subject

# MotionBERT 架构超参（dim_feat/mlp_ratio 因预训练权重而异，其余固定）
BASE = dict(dim_in=3, dim_out=3, dim_rep=512, depth=5,
            num_heads=8, num_joints=17, maxlen=243)
ARCH_LITE = dict(dim_feat=256, mlp_ratio=4)     # MB_lite (H36M/AMASS 预训练, 61MB)
ARCH_RELEASE = dict(dim_feat=512, mlp_ratio=2)  # MB_release / NTU 动作权重 (162MB)


def _strip_prefix(state: dict) -> dict:
    """剥离 module./backbone./head. 前缀，得到 DSTformer backbone 的 state_dict。"""
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            out[k[len("backbone."):]] = v
        elif k.startswith("head."):
            continue  # 丢弃分类头（我们自建 40 类 head）
        else:
            out[k] = v
    return out


def _infer_arch(state: dict) -> dict:
    """按 joints_embed.weight 的 out_features 推断架构（512=MB_release, 256=MB_lite）。"""
    for k, v in state.items():
        if k.endswith("joints_embed.weight") and v.ndim >= 2:
            return dict(ARCH_RELEASE) if v.shape[0] == 512 else dict(ARCH_LITE)
    return dict(ARCH_LITE)


def load_pretrained_backbone(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_pos" in ckpt:
        state = _strip_prefix(ckpt["model_pos"])
        kind = "MB_lite 预训练(model_pos)"
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = _strip_prefix(ckpt["model"])
        kind = "ActionNet 动作权重(model)"
    else:
        state = _strip_prefix(ckpt) if isinstance(ckpt, dict) else ckpt
        kind = "裸 state_dict"
    if not state:
        raise RuntimeError(f"权重解析失败：{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    arch = _infer_arch(state)
    arch_name = "MB_release(dim_feat=512)" if arch["dim_feat"] == 512 else "MB_lite(dim_feat=256)"
    print(f"检测到 {kind}，架构 {arch_name}（{len(state)} 个参数）", flush=True)
    cfg = dict(BASE)
    cfg.update(arch)
    backbone = DSTformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **cfg)
    model_dict = backbone.state_dict()
    matched = discarded = 0
    for k, v in state.items():
        if k in model_dict and model_dict[k].size() == v.size():
            model_dict[k] = v
            matched += 1
        else:
            discarded += 1
    backbone.load_state_dict(model_dict)
    print(f"pretrained loaded: matched={matched} discarded={discarded} "
          f"(~{sum(p.numel() for p in backbone.parameters())/1e6:.1f}M)", flush=True)
    return backbone


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (fused_acc, acc_2d, acc_3d)。融合权重来自模型（learn 或 fixed）。"""
    model.eval()
    cf = c2 = c3 = total = 0
    for x2, x3, y, _ in loader:
        x2, x3, y = x2.to(device), x3.to(device), y.to(device)
        fused, l2, l3 = model(x2, x3)
        cf += (fused.argmax(-1) == y).sum().item()
        c2 += (l2.argmax(-1) == y).sum().item()
        c3 += (l3.argmax(-1) == y).sum().item()
        total += y.numel()
    return cf / max(total, 1), c2 / max(total, 1), c3 / max(total, 1)


def train_fold(model, train_loader, val_loader, device, epochs, fold, save_path,
               lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.01,
               label_smoothing=0.1, patience=8, loss_mode="fused"):
    param_groups = [
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.input_head_2d.parameters(), "lr": lr_backbone},
        {"params": model.input_head_3d.parameters(), "lr": lr_backbone},
        {"params": model.head_2d.parameters(), "lr": lr_head},
        {"params": model.head_3d.parameters(), "lr": lr_head},
    ]
    if model.fusion == "learn":
        param_groups.append({"params": model.logit_w, "lr": lr_head})
    opt = torch.optim.AdamW(param_groups, lr=lr_backbone, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best = 0.0
    no_improve = 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x2, x3, y, _ in train_loader:
            x2, x3, y = x2.to(device), x3.to(device), y.to(device)
            opt.zero_grad()
            fused, l2, l3 = model(x2, x3)
            if loss_mode == "fused":
                loss = crit(fused, y)                     # 融合 logits 单 CE（联合优化）
            else:                                         # dual：两分支独立 CE 相加
                loss = crit(l2, y) + crit(l3, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        fa, a2, a3 = evaluate(model, val_loader, device)
        w = float(model.fusion_weight().cpu())
        if fa > best:
            best = fa
            no_improve = 0
            if save_path is not None:
                torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val_fused={fa:.4f}(2d={a2:.4f}/3d={a3:.4f}) best={best:.4f} "
              f"w={w:.3f} lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= patience:
            print(f"[fold{fold}] early stop @ ep{ep+1}（{patience} ep 无提升）", flush=True)
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--thermal_out", type=str, default="outputs/thermal_skeleton",
                    help="Step 2 热像 2D 骨架 npz 根目录")
    ap.add_argument("--pretrained", type=str, default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--conf_norm", type=str, default="sigmoid",
                    choices=["sigmoid", "minmax", "none"],
                    help="热像 conf 归一化：sigmoid(默认,无参→(0,1)) / minmax(全局) / none(恒1.0 对照)")
    ap.add_argument("--rot_angle", type=float, default=30.0,
                    help="3D 分支绕竖直轴随机旋转角度（补相机视点差，NTU cross-view 标准）")
    ap.add_argument("--loss_mode", type=str, default="fused", choices=["fused", "dual"],
                    help="fused=融合 logits 单 CE（联合优化，默认）；dual=两分支独立 CE 相加")
    ap.add_argument("--fusion", type=str, default="learn", choices=["learn", "fixed"],
                    help="logits 融合权重：learn=可学习（默认）；fixed=固定 0.5")
    ap.add_argument("--save_dir", type=str, default="outputs/dual_branch")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    thermal_out = Path(args.thermal_out).expanduser()
    minmax = args.conf_norm == "minmax"
    pairs, conf_global = build_dual_pairs(root, thermal_out, minmax=minmax)
    print(f"device={device} 双源齐全 clips={len(pairs)} "
          f"subjects={len(set(c.subject for c in pairs))} "
          f"classes={len(set(c.action_id for c in pairs))} conf_norm={args.conf_norm} "
          f"rot_angle={args.rot_angle} loss={args.loss_mode} fusion={args.fusion}", flush=True)
    if conf_global is not None:
        print(f"  minmax conf 全局范围: [{conf_global[0]:.3f}, {conf_global[1]:.3f}]", flush=True)

    folds = split_by_subject(pairs, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    accs = []
    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_clips = [pairs[i] for i in tr_idx]
        va_clips = [pairs[i] for i in va_idx]
        tr_ds = DualSkeletonDataset(tr_clips, thermal_out, args.num_frames, True,
                                    conf_norm=args.conf_norm, rot_angle=args.rot_angle,
                                    conf_global=conf_global)
        va_ds = DualSkeletonDataset(va_clips, thermal_out, args.num_frames, False,
                                    conf_norm=args.conf_norm, rot_angle=args.rot_angle,
                                    conf_global=conf_global)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               shuffle=sampler is None,  # 无 sampler 也必须重排（每 epoch 随机增强）
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        # 每折重载预训练 + 重建模型（避免跨 fold 状态泄漏）
        backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
        model = DualBranchActionNet(backbone=backbone, dim_rep=BASE["dim_rep"],
                                    num_classes=40, dropout_ratio=0.5,
                                    hidden_dim=args.hidden_dim, num_joints=17,
                                    fusion=args.fusion).to(device)
        if fi == target[0]:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"DualBranchActionNet total params ~{n_params/1e6:.1f}M "
                  f"(fp32 ~{n_params*4/1e6:.0f}MB)", flush=True)

        save_path = save_dir / f"motionbert_fold{fi}.pth"
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, fi, save_path,
                          label_smoothing=args.label_smoothing, patience=args.patience,
                          loss_mode=args.loss_mode)
        accs.append(best)
        print(f"== fold {fi} best val = {best:.4f} -> {save_path}")

    if len(accs) > 1:
        print(f"\n==== mean val across {len(accs)} folds: {np.mean(accs):.4f} "
              f"(std {np.std(accs):.4f}) ====", flush=True)


if __name__ == "__main__":
    main()
