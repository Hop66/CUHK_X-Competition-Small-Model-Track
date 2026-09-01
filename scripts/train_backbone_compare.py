#!/usr/bin/env python3
"""CUHK-X —— B 线：骨干对比 fold0 验证（换骨干能否质变？）

目的：验证"更强的视频骨干（torchvision Kinetics 预训练）在 cross-subject 下
是否显著超 R2Plus1D-34（IG-65M）"。
  - thermal 3ch 先验证（直接匹配预训练域，最公平）
  - 基线：R2Plus1D-34 thermal fold0 = 0.6339；main fold0 = 0.6609
候选骨干：r2plus1d34 / x3d_m / x3d_l / s3d / mvit_v2_s / swin3d_t

用法:
    python scripts/train_backbone_compare.py --backbone mvit_v2_s --modality thermal --fold 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.backbone_compare import build_backbone
from src.dataset import (DepthIRVideoDataset, ThermalVideoDataset,
                         build_balanced_sampler, build_thermal_index,
                         build_train_index)
from src.split import split_by_subject


def _to_model_input(x, backbone):
    """按骨干输入约定转换：[B,T,C,H,W] → 模型期望形状。

    r2plus1d34：期望 [B,T,C,H,W]（其 forward 内部再 permute 成 [B,C,T,H,W]）。
    swin3d_t/x3d/s3d/mvit：期望 [B,C,T,H,W]。
    """
    if backbone == "r2plus1d34":
        return x
    return x.permute(0, 2, 1, 3, 4)


@torch.no_grad()
def evaluate(model, loader, device, backbone, track_crop=False, vote_k=1):
    model.eval()
    correct = total = 0
    for b in loader:
        x = b[0].to(device)                    # [B,T,C,H,W]
        B, T = x.shape[0], x.shape[1]
        if vote_k > 1 and T % vote_k == 0:     # 3D 段级投票：拆 K 个短段，独立 3D 前向
            x = x.reshape(B, vote_k, T // vote_k, *x.shape[2:]).reshape(
                B * vote_k, T // vote_k, *x.shape[2:])
        x = _to_model_input(x, backbone)
        y = b[2].to(device) if track_crop else b[1].to(device)
        tr = b[1].to(device) if track_crop else None
        out = model(x, tr)
        if vote_k > 1:
            out = out.view(B, vote_k, -1).mean(1)   # 段级 logits 平均（14th 投票精髓的 3D 版）
        correct += (out.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_fold(model, train_loader, val_loader, device, epochs, lr, fold, backbone,
               save_path=None, label_smoothing=0.1, patience=15, track_crop=False,
               vote_k=1):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best, no_improve = 0.0, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for b in train_loader:
            x = b[0].to(device)                # [B,T,C,H,W]
            B, T = x.shape[0], x.shape[1]
            if vote_k > 1 and T % vote_k == 0:
                x = x.reshape(B, vote_k, T // vote_k, *x.shape[2:]).reshape(
                    B * vote_k, T // vote_k, *x.shape[2:])
            x = _to_model_input(x, backbone)
            y = b[2].to(device) if track_crop else b[1].to(device)
            tr = b[1].to(device) if track_crop else None
            opt.zero_grad()
            out = model(x, tr)
            if vote_k > 1:
                out = out.view(B, vote_k, -1).mean(1)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate(model, val_loader, device, backbone, track_crop=track_crop, vote_k=vote_k)
        if acc > best:
            best, no_improve = acc, 0
            if save_path is not None:
                torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= patience:
            print(f"[fold{fold}] early stop @ ep{ep+1}", flush=True)
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--backbone", type=str, default="r2plus1d34",
                    choices=["r2plus1d34", "x3d_m", "x3d_l", "s3d", "mvit_v2_s", "swin3d_t"])
    ap.add_argument("--modality", type=str, default="thermal", choices=["thermal", "depthir"])
    ap.add_argument("--crop_cache", type=str, default="bbox_thermal_train.json")
    ap.add_argument("--weights", type=str, default="ig65m_r2plus1d34.pth",
                    help="仅 r2plus1d34 用（IG-65M）；其余骨干用 torchvision Kinetics 预训练")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--sample_mode", type=str, default="uniform", choices=["uniform", "segment"],
                    help="时间采样：uniform=endpoint 全程均匀；segment=全球覆盖分段（14th-place 移植）")
    ap.add_argument("--gray_norm", action="store_true",
                    help="仅 thermal：灰度拉伸归一化 (x-0.5)/0.25 替代 Kinetics norm")
    ap.add_argument("--track_crop", action="store_true",
                    help="逐帧跟人裁剪 + 轨迹通道（仅 r2plus1d34/main 3D 支持；人物满框+位移/尺度显式回补）")
    ap.add_argument("--vote_k", type=int, default=1,
                    help="3D 段级投票：clip 拆成 K 个等长短段独立 3D 前向 → 段级 logits 平均（14th 投票精髓 3D 版，T 须能被 K 整除）")
    ap.add_argument("--box_path", type=str, default="bbox_ir_perframe.json",
                    help="detect --per_frame 产出的逐帧 bbox（IR 帧，main 主线）")
    ap.add_argument("--save_dir", type=str, default="outputs/backbone_cmp")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    in_channels = 3 if args.modality == "thermal" else 4

    if args.modality == "thermal":
        clips = build_thermal_index(root)
        ds_cls = ThermalVideoDataset
    else:
        clips = build_train_index(root)
        ds_cls = DepthIRVideoDataset
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    if args.track_crop and args.backbone != "r2plus1d34":
        raise ValueError("track_crop 仅 r2plus1d34（main 3D）支持；其它骨干无轨迹分支")
    track_kw = dict(track_crop=args.track_crop, box_path=args.box_path) if args.track_crop else {}
    # gray_norm: thermal 灰度拉伸归一化 (0.5,0.25)（0.8 队 notebook 移植）；depthir 用 Kinetics 默认
    ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
    if args.modality == "thermal":
        tr_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop,
                                    aug_strength=args.aug_strength,
                                    sample_mode=args.sample_mode, mean_std=ms, **track_kw)
        va_ds = ThermalVideoDataset(va_clips, args.num_frames, args.size, False, crop,
                                    sample_mode=args.sample_mode, mean_std=ms, **track_kw)
    else:
        tr_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop,
                                    aug_strength=args.aug_strength,
                                    sample_mode=args.sample_mode, **track_kw)
        va_ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop,
                                    sample_mode=args.sample_mode, **track_kw)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    wpath = Path(args.weights).expanduser() if args.backbone == "r2plus1d34" else None
    model = build_backbone(args.backbone, num_classes=40, in_channels=in_channels,
                           weights_path=str(wpath) if wpath else None,
                           traj_dim=(4 if args.track_crop else 0)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} backbone={args.backbone} modality={args.modality} "
          f"clips={len(clips)} params={n_params/1e6:.1f}M (fp32 {n_params*4/1e6:.0f}MB) "
          f"sample={args.sample_mode} track={args.track_crop}", flush=True)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.backbone}_{args.modality}_fold{args.fold}.pth"
    best = train_fold(model, tr_loader, va_loader, device, args.epochs, args.lr, args.fold,
                      args.backbone, save_path, label_smoothing=args.label_smoothing,
                      track_crop=args.track_crop, vote_k=args.vote_k)
    print(f"== {args.backbone} {args.modality} fold{args.fold} best val = {best:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
