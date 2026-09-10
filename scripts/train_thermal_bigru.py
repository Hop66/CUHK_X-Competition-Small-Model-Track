#!/usr/bin/env python3
"""Thermal BiGRU 复现(fork cuhk-x-thermal-specialist-v2)

架构: per-frame ResNet-ish 编码(conv7+ResBlock→256) + 2层双向GRU(256) + attention + classifier。
归一: (x-0.5)/0.25 灰度归一(其 14th 基线同款)。
协议: 我们的 subject-3 折(split_by_subject) + bbox crop + 类平衡采样, 与之可比的 fold val。

用法:
  python scripts/train_thermal_bigru.py --folds 0 1 2 --epochs 60 \
      --lr 3e-4 --batch_size 32 --save_dir outputs/th_bigru
输出: 每折 best val + mean(对照 R2+1D thermal fold0 Kinet=0.5966 / gray=53524-T1)
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (ThermalVideoDataset, build_balanced_sampler,
                         build_thermal_index)
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
        o, _ = self.gru(f)
        a = F.softmax(self.attn(o).squeeze(-1), 1)
        v = (o * a.unsqueeze(-1)).sum(1)
        return self.classifier(v)


def evaluate(model, loader, device):
    model.eval()
    nc = nt = 0
    with torch.no_grad():
        for b in loader:
            x = b[0].to(device); y = b[1]
            nc += (model(x).argmax(1).cpu() == y).sum().item(); nt += y.numel()
    return nc / max(nt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--crop", default="bbox_thermal_train.json")
    ap.add_argument("--save_dir", default="outputs/th_bigru")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import json
    crop_cache = json.loads(Path(args.crop).read_text(encoding="utf-8"))
    root = Path(args.train_root).expanduser()
    clips = build_thermal_index(root)
    folds = split_by_subject(clips, n_folds=3)
    GRAY = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))

    bests = []
    for f in args.folds:
        tr_idx, va_idx = folds[f]
        tr_c = [clips[i] for i in tr_idx]; va_c = [clips[i] for i in va_idx]
        tr_ds = ThermalVideoDataset(tr_c, args.num_frames, args.size, True, crop_cache,
                                    use_frame_diff=False, mean_std=GRAY, sample_mode="uniform",
                                    seed=args.seed)
        va_ds = ThermalVideoDataset(va_c, args.num_frames, args.size, False, crop_cache,
                                    use_frame_diff=False, mean_std=GRAY, sample_mode="uniform",
                                    seed=args.seed)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               sampler=build_balanced_sampler([c.action_id for c in tr_c]),
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
        model = ThermalGRUNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        best = 0.0
        sd = "%d_%d" % (args.size, args.num_frames)
        for ep in range(args.epochs):
            model.train(); tl = n = 0
            for b in tr_loader:
                x = b[0].to(device); y = b[1].to(device)
                loss = crit(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * x.size(0); n += x.size(0)
            sched.step()
            acc = evaluate(model, va_loader, device)
            if acc > best:
                best = acc
                fp = Path(args.save_dir); fp.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "best_acc": best, "fold": f}, fp / f"bigru_f{f}.pth")
            if (ep + 1) % 10 == 0:
                print(f"[fold{f}] ep{ep+1}/{args.epochs} loss={tl/max(n,1):.4f} val={acc:.4f} best={best:.4f}", flush=True)
        bests.append(best)
        print(f"== fold{args.fold if False else f} best val = {best:.4f}", flush=True)
    print(f"[BiGRU-thermal({sd})] folds {args.folds} best = {[round(b,4) for b in bests]} "
          f"mean={np.mean(bests):.4f}  (对照 R2+1D-thermal fold0 Kinet=0.5966)", flush=True)


if __name__ == "__main__":
    main()
