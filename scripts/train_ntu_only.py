#!/usr/bin/env python3
"""诊断：NTU 单独训练（NTU 内部按被试划分）→ 验证 NTU 数据/25→17 映射有效性。

判读：
  NTU val acc > 0.7 → NTU 数据有效，问题在 multi-task 方法（数据浪费+lr 死）→ 改两阶段
  NTU val acc ~0.5 → NTU 数据/映射有问题 → 先修数据管线

用法:
    python scripts/train_ntu_only.py --ntu_root data/external/ntu/ntu60 \
        --pretrained weights/mb_lite_latest_epoch.bin
"""

import argparse
import re
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.motionbert.action_net import ActionHeadClassification
from src.motionbert.dstformer import DSTformer
from src.ntu_dataset import NTUSkeletonDataset

from train_skeleton_motionbert import BASE, load_pretrained_backbone

FNAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


class NTUModel(nn.Module):
    """backbone + 单 NTU 40 类头。"""

    def __init__(self, backbone, dim_rep=512, num_classes=40, hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        self.head = ActionHeadClassification(0.5, dim_rep, num_classes, 17, hidden_dim)

    def forward(self, x):
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        feat = self.backbone.get_representation(x)
        feat = feat.reshape(N, M, T, J, -1)
        return self.head(feat)


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
    ap.add_argument("--ntu_root", default="data/external/ntu/ntu60")
    ap.add_argument("--pretrained", default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="backbone lr（NTU 数据多可用大 lr）")
    ap.add_argument("--val_subjects", type=int, default=8, help="验证被试数（NTU60 共 40）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/ntu_diag")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 按被试划分（P001-P040，最后 N 个被试验证）
    all_files = []
    for f in sorted(Path(args.ntu_root).glob("*.skeleton")):
        m = FNAME_RE.search(f.name)
        if m and int(m.group(5)) <= 40:
            all_files.append(f)
    va_set = set(range(41 - args.val_subjects, 41))  # 验证被试号
    tr_files = [f for f in all_files if int(FNAME_RE.search(f.name).group(3)) not in va_set]
    va_files = [f for f in all_files if int(FNAME_RE.search(f.name).group(3)) in va_set]
    print(f"NTU: 全部={len(all_files)} 训练={len(tr_files)} 验证={len(va_files)}", flush=True)

    tr_ds = NTUSkeletonDataset(args.ntu_root, args.num_frames, True, seed=42, files=tr_files)
    va_ds = NTUSkeletonDataset(args.ntu_root, args.num_frames, False, files=va_files)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    print(f"device={device} steps/ep={len(tr_loader)}", flush=True)

    backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
    model = NTUModel(backbone, dim_rep=BASE["dim_rep"], num_classes=40, hidden_dim=512).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * 10},
    ], lr=args.lr, weight_decay=0.01)
    # StepLR 缓慢衰减（避免 Cosine 衰减到 0 死学习）
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    best, no_improve = 0.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device).unsqueeze(1), y.to(device)
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
            torch.save({"model": model.state_dict(), "best_acc": best}, save_dir / "ntu_only.pth")
        else:
            no_improve += 1
        print(f"[ntu] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"ntu_val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= 15:
            print(f"[ntu] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== NTU 单独 best val = {best:.4f} "
          f"（>0.7 → 数据有效，multi-task 方法问题 → 改两阶段；~0.5 → 数据/映射问题）", flush=True)


if __name__ == "__main__":
    main()
