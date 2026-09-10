#!/usr/bin/env python3
"""CUHK-X — Thermal GRU+Attention 流（复刻 public "Thermal Specialist v2"）。

per-frame 2D ResNet 编码 + BiGRU + Attention 加权时序池 → 40 类。
与 R2+1D(3D) thermal 完全正交，作为 prob-avg 成员。
对照: thermal R2+1D fold0=0.5966 / nf24=0.6030。
用法:
  python scripts/train_thermal_gru_attn.py --fold 0 --epochs 60 --save_dir outputs/th_gru
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import ThermalVideoDataset, build_thermal_index
from src.split import split_by_subject


class ResBlock(nn.Module):
    def __init__(self, a, b, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b))
        self.skip = nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b)) \
            if (a != b or s != 1) else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


class ThermalGRUNet(nn.Module):
    def __init__(self, num_classes=40, feat_dim=256, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.frame_enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(32, 64), ResBlock(64, 64),
            ResBlock(64, 128, 2), ResBlock(128, 128),
            ResBlock(128, feat_dim, 2), ResBlock(feat_dim, feat_dim),
            nn.AdaptiveAvgPool2d(1))
        self.gru = nn.GRU(feat_dim, hidden_dim, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.frame_enc(x.reshape(B * T, C, H, W)).flatten(1).view(B, T, -1)
        g, _ = self.gru(f)
        w = F.softmax(self.attn(g), dim=1)
        return self.classifier((g * w).sum(dim=1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_cache", default="bbox_thermal_train.json")
    ap.add_argument("--save_dir", default="outputs/th_gru")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_thermal_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_c = [clips[i] for i in tr_idx]
    va_c = [clips[i] for i in va_idx]
    crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    tr_ds = ThermalVideoDataset(tr_c, args.num_frames, args.size, True, crop,
                                aug_strength=args.aug_strength)
    va_ds = ThermalVideoDataset(va_c, args.num_frames, args.size, False, crop)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=True, timeout=300)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True, timeout=300)
    print(f"train={len(tr_ds)} val={len(va_ds)} nf={args.num_frames}", flush=True)

    model = ThermalGRUNet().to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_dir) / f"th_gru_fold{args.fold}.pth"
    best = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl = n = 0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tl += loss.item() * y.numel(); n += y.numel()
        sched.step()
        acc = evaluate(model, va_loader, device)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "best_acc": best, "epoch": ep + 1}, save_path)
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={tl/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.1e} ({time.time()-t0:.0f}s)",
              flush=True)
    print(f"== ThermalGRU fold{args.fold} best={best:.4f} (对照 R2+1D-th 0.5966/0.6030) -> {save_path}",
          flush=True)


if __name__ == "__main__":
    main()
