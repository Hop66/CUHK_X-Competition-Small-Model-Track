#!/usr/bin/env python3
"""CUHK-X —— main CE 基线（普通 Linear 头）多折评估（ArcFace 的可靠对照）

与 train_main_arcface.py 同结构/同 scheduler/同协议（StepLR + 多折 mean±std + aug_strength），
保证 ArcFace vs CE 对比公平（此前骨架教训：scheduler 不同会让对比失真）。

用法:
    python scripts/train_main_baseline.py --fold -1 --weights ig65m_r2plus1d34.pth --aug_strength 2
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
from torch.utils.data import DataLoader

from src.dataset import DepthIRVideoDataset, build_train_index
from src.model import R2Plus1D34
from src.split import split_by_subject


class BaselineModel(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.encoder = R2Plus1D34(40, 4, weights_path).encoder  # [B,512]
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 40))

    def forward(self, x):
        return self.head(self.encoder(x.permute(0, 2, 1, 3, 4)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--aug_strength", type=int, default=2, help="增强强度（1 在 fold0 优于 2）")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1, help="-1=跑全部折并输出 mean±std")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--save_dir", type=str, default="outputs/main_baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    crop_cache = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    target_folds = range(len(folds)) if args.fold < 0 else [args.fold]
    fold_accs = []
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    for fi in target_folds:
        tr_idx, va_idx = folds[fi]
        tr_clips = [clips[i] for i in tr_idx]
        va_clips = [clips[i] for i in va_idx]
        tr_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                    use_frame_diff=False, aug_strength=args.aug_strength)
        va_ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                    use_frame_diff=False)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
        print(f"device={device} fold{fi} train={len(tr_ds)} val={len(va_ds)} "
              f"CE baseline aug={args.aug_strength}", flush=True)

        model = BaselineModel(args.weights).to(device)
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr * 10},
        ], lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        save_path = save_dir / f"baseline_aug{args.aug_strength}_fold{fi}.pth"
        best, no_improve = 0.0, 0
        for ep in range(args.epochs):
            t0 = time.time()
            model.train()
            run_loss, n = 0.0, 0
            for x, y, _ in tr_loader:
                x, y = x.to(device), y.to(device)
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
            print(f"[fold{fi}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"ce_val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if no_improve >= args.patience:
                print(f"[fold{fi}] early stop @ ep{ep+1}", flush=True)
                break

        fold_accs.append(best)
        print(f"== CE baseline fold{fi} best val = {best:.4f} → {save_path}", flush=True)

    if len(fold_accs) > 1:
        print(f"\n==== CE baseline aug{args.aug_strength} {len(fold_accs)} 折 mean = "
              f"{np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f} ====")
        print("此即 ArcFace 的可靠对照（同协议）", flush=True)
    else:
        print(f"== CE baseline fold{args.fold} best = {fold_accs[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
