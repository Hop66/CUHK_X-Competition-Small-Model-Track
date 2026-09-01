#!/usr/bin/env python3
"""CUHK-X —— main + Center Loss（CE + λ·类中心紧凑）多折评估

Center Loss（Wen et al. 2016）：在 CE 上叠加"特征到类中心的距离"项，强制类内紧凑。
与 ArcFace 同属"损失层面"跨身份泛化，但更简单（无 margin/normalize）。
结构/scheduler/协议与 train_main_baseline.py / train_main_arcface.py 完全一致，可三方对比。

用法:
    python scripts/train_main_centerloss.py --fold -1 --weights ig65m_r2plus1d34.pth --aug_strength 2
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


class CenterLoss(nn.Module):
    """标准 Center Loss：特征到对应类中心的 L2 距离（Wen et al. 2016）。"""

    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
                   + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).t()
                   .expand(batch_size, self.num_classes))
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        return dist.clamp(min=1e-12, max=1e12).sum() / batch_size


class CenterLossModel(nn.Module):
    def __init__(self, weights_path, num_classes=40, feat_dim=512):
        super().__init__()
        self.encoder = R2Plus1D34(40, 4, weights_path).encoder  # [B,512]
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, num_classes))
        self.center_loss = CenterLoss(num_classes, feat_dim)

    def forward(self, x):
        feat = self.encoder(x.permute(0, 2, 1, 3, 4))
        return self.head(feat), feat


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
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
    ap.add_argument("--lambda_c", type=float, default=1e-3, help="Center Loss 权重")
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1, help="-1=跑全部折并输出 mean±std")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--save_dir", type=str, default="outputs/main_centerloss")
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
              f"CenterLoss λ={args.lambda_c} aug={args.aug_strength}", flush=True)

        model = CenterLossModel(args.weights).to(device)
        # centers 单独参数组：无 weight decay（Center Loss 论文要求）
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr * 10},
            {"params": model.center_loss.centers, "lr": args.lr, "weight_decay": 0.0},
        ], lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        save_path = save_dir / f"centerloss_aug{args.aug_strength}_fold{fi}.pth"
        best, no_improve = 0.0, 0
        for ep in range(args.epochs):
            t0 = time.time()
            model.train()
            run_loss, n = 0.0, 0
            for x, y, _ in tr_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits, feat = model(x)
                loss = crit(logits, y) + args.lambda_c * model.center_loss(feat, y)
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
                  f"cl_val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if no_improve >= args.patience:
                print(f"[fold{fi}] early stop @ ep{ep+1}", flush=True)
                break

        fold_accs.append(best)
        print(f"== CenterLoss fold{fi} best val = {best:.4f} → {save_path}", flush=True)

    if len(fold_accs) > 1:
        print(f"\n==== CenterLoss aug{args.aug_strength} {len(fold_accs)} 折 mean = "
              f"{np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f} ====", flush=True)
    else:
        print(f"== CenterLoss fold{args.fold} best = {fold_accs[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
