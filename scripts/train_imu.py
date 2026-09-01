#!/usr/bin/env python3
"""CUHK-X —— IMU 单模态摸底（P2，异源惯性）

- 1D CNN（3 conv + GAP + head），输入 [T, 30]（5 设备 × acc3+gyro3）
- 3 折 subject split，balanced sampler
- 判据：单模态 val 是否 >0.5 且与 main 互补（决定是否加入动态集成）

用法:
    python scripts/train_imu.py --epochs 30 --folds 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler, build_train_index
from src.imu_dataset import FEAT_DIM, IMUDataset, build_imu_index
from src.split import split_by_subject


class IMUCNN(nn.Module):
    """1D CNN：输入 [B, T, 30] -> [B, 40]。"""

    def __init__(self, num_classes: int = 40, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)      # [B, T, 30] -> [B, 30, T]
        x = self.encoder(x)          # [B, 256, T/8]
        x = x.mean(dim=-1)           # GAP -> [B, 256]
        return self.head(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_fold(model, tr_loader, va_loader, device, epochs, fold, save_path, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    for ep in range(epochs):
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
            best = acc
            if save_path is not None:
                torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--T", type=int, default=128, help="IMU 时序采样帧数")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--no_jitter", action="store_true")
    ap.add_argument("--save_dir", default="outputs/imu")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    clips = build_imu_index(root, main_clips)
    print(f"device={device} clips={len(clips)} classes={len(set(c.action_id for c in clips))}", flush=True)

    folds = split_by_subject(clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    accs = []
    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_clips = [clips[i] for i in tr_idx]
        va_clips = [clips[i] for i in va_idx]
        tr_ds = IMUDataset(tr_clips, args.T, True, seed=42, jitter=not args.no_jitter)
        va_ds = IMUDataset(va_clips, args.T, False)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        model = IMUCNN().to(device)
        save_path = save_dir / f"imu_fold{fi}.pth"
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, fi, save_path, args.lr)
        accs.append(best)
        print(f"== fold {fi} best val = {best:.4f} -> {save_path}", flush=True)

    if len(accs) > 1:
        print(f"\n==== IMU mean val across {len(accs)} folds: {np.mean(accs):.4f} "
              f"(std {np.std(accs):.4f}) ====", flush=True)
        print("判据: mean val >0.5 且与 main 互补 → 值得加入动态集成", flush=True)


if __name__ == "__main__":
    main()
