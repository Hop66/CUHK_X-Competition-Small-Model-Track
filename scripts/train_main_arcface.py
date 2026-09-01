#!/usr/bin/env python3
"""CUHK-X —— main 训练 + ArcFace 度量学习头（跨身份泛化，从未试过的损失层面）

背景：cross-subject 动作识别本质 = 人脸识别/ReID（同一类跨主体）。ArcFace 是这些领域
      0.7→0.8 的核心方法（强制类内紧凑 + 类间分离），我们所有实验都用 CE，从未试过损失层面。
对照：main 4ch 全量/2折 fold0（增强后 0.6609）、0.73 保底
判读：fold0 > 0.6609（CE 基线）→ ArcFace 有效 → 全量重训 → 参与融合冲 0.8

用法:
    python scripts/train_main_arcface.py --fold 0 --weights ig65m_r2plus1d34.pth
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import ClipIndex, DepthIRVideoDataset, build_train_index
from src.model import R2Plus1D34
from src.split import split_by_subject


class ArcMarginProduct(nn.Module):
    """ArcFace（Additive Angular Margin）标准实现，来自 ArcFace 论文/arcface-pytorch。
    s=scale（30），m=margin（0.5），normalize 特征 + normalize 权重 → cos(θ)，加 margin 后 CE。
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inp, label):
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > math.cos(math.pi - self.m),
                              phi, cosine - math.sin(self.m) * self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class ArcFaceModel(nn.Module):
    """R2Plus1D34 encoder（fc=Identity）→ ArcFace 头 → 40 类（加 margin）。"""

    def __init__(self, weights_path, s=30.0, m=0.5):
        super().__init__()
        self.encoder = R2Plus1D34(40, 4, weights_path).encoder  # [B,512]
        self.arcface = ArcMarginProduct(512, 40, s=s, m=m)

    def forward(self, x, label=None):
        feat = self.encoder(x.permute(0, 2, 1, 3, 4))
        if label is not None:
            return self.arcface(feat, label)
        # 推理：余弦相似度 argmax
        cos = F.linear(F.normalize(feat), F.normalize(self.arcface.weight))
        return cos


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
    ap.add_argument("--s", type=float, default=30.0, help="ArcFace scale")
    ap.add_argument("--m", type=float, default=0.5, help="ArcFace margin")
    ap.add_argument("--aug_strength", type=int, default=2, help="增强强度（aug_search fold0: 1=0.6652 > 2=0.6609）")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1,
                    help="-1=跑全部折并输出 mean±std（可靠评估协议）；>=0=只跑单折")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--save_dir", type=str, default="outputs/main_arcface")
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
              f"ArcFace(s={args.s},m={args.m}) 对照 CE 基线", flush=True)

        model = ArcFaceModel(args.weights, s=args.s, m=args.m).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"ArcFaceModel params ~{n_params/1e6:.0f}M "
              f"(fp32 ~{n_params*4/1e6:.0f}MB → int5 ~{n_params*4/1e6/8:.0f}MB)", flush=True)

        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.arcface.parameters(), "lr": args.lr * 10},   # ArcFace 头新学要快
        ], lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        save_path = save_dir / f"arcface_fold{fi}.pth"
        best, no_improve = 0.0, 0
        for ep in range(args.epochs):
            t0 = time.time()
            model.train()
            run_loss, n = 0.0, 0
            for x, y, _ in tr_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x, y)          # ArcFace logits（带 margin）
                loss = crit(out, y)
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
                  f"arcface_val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if no_improve >= args.patience:
                print(f"[fold{fi}] early stop @ ep{ep+1}", flush=True)
                break

        fold_accs.append(best)
        print(f"== ArcFace main fold{fi} best val = {best:.4f} → {save_path}", flush=True)

    if len(fold_accs) > 1:
        print(f"\n==== ArcFace {len(fold_accs)} 折 mean = {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f} ====")
        print("⚠️ 可靠评估：需与 CE 基线同协议（多折 mean±std）对比，均值差 > std 才有把握判定有效/无效", flush=True)
    else:
        print(f"== ArcFace main fold{args.fold} best val = {fold_accs[0]:.4f} "
              f"（单折仅供快筛，最终判据用 --fold -1 多折 mean±std）", flush=True)


if __name__ == "__main__":
    main()
