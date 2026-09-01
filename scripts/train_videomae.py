#!/usr/bin/env python3
"""VideoMAE-S 接入验证 —— fold0 thermal（对比 R2Plus1D-34 基线 0.6339）。

自监督预训练（VideoMAE-S K400）域鲁棒，可能是 thermal（弱模态）的更强骨干。
- 输入 [B,3,T,H,W]，128×128×16（patch 后 8×8×8=512 tokens）
- 归一化：ImageNet（0.485/0.229，匹配 VideoMAE 预训练域）
- 权重：官方 pretrain（encoder 部分加载，pos_embed/cls_token 随机）

用法:
    python scripts/train_videomae.py --ckpt weights/videomae_s_k400_pretrain.pth --fold 0
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ThermalVideoDataset, build_balanced_sampler, build_thermal_index
from src.split import split_by_subject
from src.videomae import VideoMAES, load_videomae_pretrained

# VideoMAE 预训练域用 ImageNet 归一化
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device).permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]->[B,C,T,H,W]
        y = y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ckpt", type=str, default="weights/videomae_s_k400_pretrain.pth")
    ap.add_argument("--crop_cache", type=str, default="bbox_thermal_train.json")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--normalize", type=str, default="imagenet", choices=["imagenet", "kinetics"],
                    help="归一化：imagenet=0.485/0.229（VideoMAE 预训练域）；kinetics=0.432/0.228（与 R2+1D 基线一致）")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="线性 warmup 轮数（ViT 微调建议 5）")
    ap.add_argument("--patience", type=int, default=15, help="early stop 无改善轮数")
    ap.add_argument("--save_dir", type=str, default="outputs/videomae")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    clips = build_thermal_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    ms = (IMAGENET_MEAN, IMAGENET_STD) if args.normalize == "imagenet" else None
    tr_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop,
                                aug_strength=args.aug_strength, mean_std=ms)
    va_ds = ThermalVideoDataset(va_clips, args.num_frames, args.size, False, crop, mean_std=ms)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    model = VideoMAES(num_classes=40, in_channels=3,
                      num_frames=args.num_frames, img_size=args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} VideoMAE-S params={n_params/1e6:.1f}M "
          f"(fp32 {n_params*4/1e6:.0f}MB) clips={len(clips)}", flush=True)
    matched, total = load_videomae_pretrained(model, Path(args.ckpt).expanduser(), device)
    assert matched == total, f"加载不完整 {matched}/{total}，检查 key 映射"
    print("✅ 预训练权重加载完整", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    wu = args.warmup_epochs
    if wu > 0:
        def _lr_lambda(ep):
            if ep < wu:
                return (ep + 1) / wu
            t = (ep - wu) / max(args.epochs - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best, no_improve = 0.0, 0
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"videomae_s_fold{args.fold}.pth"
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x = x.to(device).permute(0, 2, 1, 3, 4)
            y = y.to(device)
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
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= args.patience:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break
    print(f"== VideoMAE-S thermal fold{args.fold} best val = {best:.4f} "
          f"（对比 R2Plus1D-34 基线 0.6339；>0.65 则骨干有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
