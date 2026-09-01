#!/usr/bin/env python3
"""CUHK-X —— FrameNet 轻量逐帧基线（14th-place 移植）: 验证 per-frame logits mean 是否关键

对照：thermal 基线 fold0 = 0.6339（R2+1D-34 + IG65M + Kinetics + 时序 GAP，16 帧 128px）
本实验：从零 2D CNN + per-frame logits mean + segment 采样 + gray_norm + 我们 subject fold 协议
判读：
  fold0 ≥ 0.60           → per-frame 投票在不靠预训练/时序建模下已接近强基线 → thermal 有效信息
                          主要在"单帧外观+时间投票"层 → FrameNet(~5M) 可当轻量独立热像视角入 0.73 集成
  fold0 ≥ 0.6339 或接近  → 帧级 logits 平均确为关键载体 → 考虑移植逐帧池化到强模型
  fold0 明显 < 0.55      → 纯单帧外观信息不够，3D 时序建模不可替代，per-frame 非关键
用法: python scripts/train_framenet_thermal.py --fold 0 [--sample_mode segment] [--gray_norm]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (ThermalVideoDataset, build_thermal_index,
                         build_balanced_sampler)
from src.framenet import (FrameNet, build_frame_net_pretrained)
from src.split import split_by_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop_cache", type=str, default="")
    ap.add_argument("--arch", type=str, default="framenet",
                    choices=["framenet", "resnet18", "resnet34"],
                    help="framenet=从零小 CNN（已验证 0.21 判负）；resnet18/34=ImageNet 预训练 2D"
                         "+ per-frame logits mean（0.8 组方法强化版）")
    ap.add_argument("--sample_mode", type=str, default="segment", choices=["uniform", "segment"])
    ap.add_argument("--gray_norm", action="store_true")
    ap.add_argument("--imagenet_norm", action="store_true",
                    help="用 ImageNet 归一化（匹配 resnet18/34 预训练域；与 gray_norm 互斥，域匹配 A/B）")
    ap.add_argument("--track_crop", action="store_true",
                    help="逐帧跟人裁剪 + 轨迹通道 [cx,cy,bw,bh]（人物满框+位移/尺度显式回补）")
    ap.add_argument("--box_path", type=str, default="bbox_thermal_perframe.json",
                    help="detect --per_frame 产出的逐帧 bbox（track_crop 用）")
    ap.add_argument("--no_smooth", action="store_true",
                    help="不用 label smoothing（严格复刻 14th notebook：8ep + 无 LS）")
    ap.add_argument("--save_dir", type=str, default="outputs/thermal_framenet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_thermal_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    crop_cache = {}
    if args.crop_cache:
        import json
        crop_cache = json.loads(Path(args.crop_cache).read_text(encoding="utf-8"))
        print(f"crop cache: {len(crop_cache)} entries", flush=True)

    if args.imagenet_norm:
        ms = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # 匹配 ImageNet 预训练域（resnet18/34）
    elif args.gray_norm:
        ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))             # 热像灰度拉伸（0.8 组专用）
    else:
        ms = None
    track_kw = dict(track_crop=args.track_crop, box_path=args.box_path) if args.track_crop else {}
    train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                   aug_strength=2, sample_mode=args.sample_mode, mean_std=ms,
                                   **track_kw)
    val_ds = ThermalVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                 sample_mode=args.sample_mode, mean_std=ms, **track_kw)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips])
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    model = (build_frame_net_pretrained(args.arch, 40, traj_dim=(4 if args.track_crop else 0))
             if args.arch != "framenet"
             else FrameNet(num_classes=40)).to(device)
    print(f"arch={args.arch} params ~{sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"fold{args.fold} train={len(tr_clips)} val={len(va_clips)} "
          f"frames={args.num_frames} size={args.size} sample={args.sample_mode} "
          f"gray={args.gray_norm} track={args.track_crop}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=(0.0 if args.no_smooth else 0.1))

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for it in va_loader:
                x = it[0].to(device)
                tr = it[1].to(device) if args.track_crop else None
                y = it[2 if args.track_crop else 1].to(device)
                out = model(x, tr)
                correct += (out.argmax(-1) == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"framenet_fold{args.fold}.pth"
    best, no_improve = 0.0, 0
    for ep in range(args.epochs):
        model.train()
        run_loss, n = 0.0, 0
        t0 = time.time()
        for it in tr_loader:
            x = it[0].to(device)
            tr = it[1].to(device) if args.track_crop else None
            y = it[2 if args.track_crop else 1].to(device)
            opt.zero_grad()
            loss = crit(model(x, tr), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        acc = evaluate()
        if acc > best:
            best, no_improve = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if no_improve >= 15:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== FrameNet fold{args.fold} best = {best:.4f} "
          f"（对照 thermal 基线 0.6339）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
