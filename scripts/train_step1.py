#!/usr/bin/env python3
"""
CUHK-X Step 1 —— 运动优先主线训练（Depth_Color+IR 4ch → 视频模型）

流程：
  1. 建训练索引（2891 clip / 18 被试 / 40 类）
  2. subject-fold 交叉验证（3 折，每折 6 被试验证）
  3. 类平衡采样 + Adam + cosine
  4. 每折训练/验证，输出 val 准确率

用法:
    python scripts/train_step1.py --backbone r2plus1d --num_frames 16 --epochs 30
    python scripts/train_step1.py --backbone tsm_resnet18 --num_frames 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import (ClipIndex, DepthIRVideoDataset, build_train_index,
                         ThermalVideoDataset, build_thermal_index,
                         build_balanced_sampler)
from src.split import split_by_subject
from src.model import build_model


def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    correct = total = 0
    run_loss = 0.0
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            run_loss += crit(out, y).item() * y.numel()
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1), run_loss / max(total, 1)


def train_fold(model, train_loader, val_loader, device, epochs, lr, fold, save_path=None,
               label_smoothing=0.0, full=False, save_every=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        if full:
            # 全量模式：无 val，每 save_every ep 保存快照 + 最后 ep 保存最终（CosineAnnealing 收敛）
            if save_path is not None and ((ep + 1) % save_every == 0 or ep == epochs - 1):
                torch.save(model.state_dict(), save_path)
            print(f"[{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
                  f"(FULL) saved={save_path} ({time.time()-t0:.1f}s)", flush=True)
            continue
        acc, val_loss = evaluate(model, val_loader, device, label_smoothing)
        if acc > best:
            best = acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val_loss={val_loss:.4f} val={acc:.4f} best={best:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--backbone", type=str, default="r2plus1d",
                    choices=["r2plus1d", "r2plus1d34", "tsm_resnet18"])
    ap.add_argument("--weights", type=str, default="", help="IG-65M 预训练权重路径（r2plus1d34 用）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop_cache", type=str, default="", help="bbox_cache.json 路径，缺省不裁剪")
    ap.add_argument("--ir_mask", action="store_true", help="IR Otsu 人像掩码抑制 Depth 背景")
    ap.add_argument("--frame_diff", action="store_true", help="追加帧差运动通道（显式抓运动，4→8ch）")
    ap.add_argument("--no_balanced", action="store_true", help="关闭类平衡采样")
    ap.add_argument("--modality", type=str, default="depthir", choices=["depthir", "thermal"],
                    help="depthir=Depth+IR 4ch；thermal=Thermal 3ch")
    ap.add_argument("--save_dir", type=str, default="", help="保存 best checkpoint 的目录（缺省不保存）")
    ap.add_argument("--fold", type=int, default=-1, help="只跑指定折（-1=全折）")
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="CrossEntropy 标签平滑（缓解过拟合/校准，0.1 推荐）")
    ap.add_argument("--full", action="store_true",
                    help="全量训练：全部数据不分折、无 val，保存最后 epoch（全量冲刺用）")
    ap.add_argument("--full_save_every", type=int, default=10, help="全量模式快照保存间隔")
    ap.add_argument("--seed", type=int, default=42, help="全量模式随机种子（多 seed 集成用）")
    ap.add_argument("--aug_strength", type=int, default=2,
                    help="训练增强强度档位 0/1/2/3（0=无,1=温和,2=当前强,3=更强）")
    ap.add_argument("--sample_mode", type=str, default="uniform", choices=["uniform", "segment"],
                    help="时间采样：uniform=endpoint 全程均匀（现状）；segment=全球覆盖分段"
                         "（均分 num_frames 段每段 1 帧，train 段内随机/eval 段中，14th-place 移植）")
    ap.add_argument("--gray_norm", action="store_true",
                    help="仅 thermal：用灰度拉伸归一化 (x-0.5)/0.25 替代 Kinetics norm"
                         "（14th-place baseline 用；thermal 单通道灰度更匹配）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} backbone={args.backbone} frames={args.num_frames}")

    root = Path(args.train_root).expanduser()
    if args.modality == "thermal":
        clips = build_thermal_index(root)
        in_channels = 6 if args.frame_diff else 3
    else:
        clips = build_train_index(root)
        in_channels = 8 if args.frame_diff else 4
    print(f"modality={args.modality} clips={len(clips)} "
          f"subjects={len(set(c.subject for c in clips))} "
          f"classes={len(set(c.action_id for c in clips))}")

    crop_cache = {}
    if args.crop_cache:
        crop_cache = json.loads(Path(args.crop_cache).read_text(encoding="utf-8"))
        print(f"crop cache loaded: {len(crop_cache)} entries")

    folds = split_by_subject(clips, n_folds=args.folds)
    target_folds = range(len(folds)) if args.fold < 0 else [args.fold]

    # ================= FULL 全量训练（不分折，多 seed） =================
    if args.full:
        tr_clips = clips
        if args.modality == "thermal":
            ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
            train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_frame_diff=args.frame_diff, seed=args.seed,
                                           aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode, mean_std=ms)
        else:
            train_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_ir_mask=args.ir_mask, use_frame_diff=args.frame_diff,
                                           seed=args.seed, aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode)
        sampler = None
        if not args.no_balanced:
            sampler = build_balanced_sampler([c.action_id for c in tr_clips])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        model = build_model(args.backbone, num_classes=40, in_channels=in_channels,
                            n_segment=args.num_frames,
                            weights_path=args.weights or None).to(device)
        save_path = None
        if args.save_dir:
            d = Path(args.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            save_path = d / f"{args.backbone}_{args.modality}_full_seed{args.seed}.pth"
        print(f"==== FULL 全量训练（{len(tr_clips)} clips, seed={args.seed}, "
              f"epochs={args.epochs}, lr={args.lr}, label_smoothing={args.label_smoothing}）====", flush=True)
        train_fold(model, train_loader, None, device, args.epochs, args.lr, "full",
                   save_path, label_smoothing=args.label_smoothing,
                   full=True, save_every=args.full_save_every)
        print(f"==== FULL DONE: {save_path} ====", flush=True)
        return

    fold_accs = []
    for fi in target_folds:
        train_idx, val_idx = folds[fi]
        tr_clips = [clips[i] for i in train_idx]
        va_clips = [clips[i] for i in val_idx]

        if args.modality == "thermal":
            ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
            train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_frame_diff=args.frame_diff,
                                           aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode, mean_std=ms)
            val_ds = ThermalVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                         use_frame_diff=args.frame_diff,
                                         sample_mode=args.sample_mode, mean_std=ms)
        else:
            train_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_ir_mask=args.ir_mask, use_frame_diff=args.frame_diff,
                                           aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode)
            val_ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                         use_ir_mask=args.ir_mask, use_frame_diff=args.frame_diff,
                                         sample_mode=args.sample_mode)

        sampler = None
        if not args.no_balanced:
            sampler = build_balanced_sampler([c.action_id for c in tr_clips])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

        model = build_model(args.backbone, num_classes=40, in_channels=in_channels,
                            n_segment=args.num_frames,
                            weights_path=args.weights or None).to(device)
        save_path = None
        if args.save_dir:
            d = Path(args.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            save_path = d / f"{args.backbone}_{args.modality}_fold{fi}.pth"
        best = train_fold(model, train_loader, val_loader, device, args.epochs, args.lr, fi,
                          save_path, label_smoothing=args.label_smoothing)
        fold_accs.append(best)
        print(f"== fold {fi} best val = {best:.4f} -> {save_path}")

    if len(fold_accs) > 1:
        print(f"\n==== mean val across {len(fold_accs)} folds: {np.mean(fold_accs):.4f} "
              f"(std {np.std(fold_accs):.4f}) ====")


if __name__ == "__main__":
    main()
