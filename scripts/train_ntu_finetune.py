#!/usr/bin/env python3
"""CUHK-X —— 阶段2：NTU 预训练 backbone → 我们数据微调（fold0 验证）。

加载阶段1的 NTU 预训练 backbone（DSTformer），换我们 40 类头，
在我们数据上微调（lr 低保预训练特征 + StepLR 不衰减到 0）。

对照：骨架基线 0.51 / 混合增强 0.5447
判读：fold0 > 0.55 显著 → NTU 预训练-微调有效 → 全量微调 + 参与融合

用法:
    python scripts/train_ntu_finetune.py --fold 0 \
        --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.motionbert.action_net import ActionHeadClassification
from src.motionbert.dstformer import DSTformer
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.split import split_by_subject

from train_skeleton_motionbert import BASE


class FinetuneNet(nn.Module):
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


def build_backbone(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    bb_state = ckpt["backbone"]
    bb = DSTformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **BASE).to(device)
    bb.load_state_dict(bb_state)
    return bb


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
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ntu_ckpt", required=True, help="阶段1 NTU 预训练 backbone")
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_finetune")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    tr_ds = MotionBertSkeletonDataset(tr_clips, args.num_frames, True, input3d=True)
    va_ds = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=True)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    print(f"微调: train={len(tr_ds)} val={len(va_ds)}（NTU 预训练 backbone）", flush=True)

    backbone = build_backbone(Path(args.ntu_ckpt).expanduser(), device)
    model = FinetuneNet(backbone, dim_rep=BASE["dim_rep"], num_classes=40, hidden_dim=512).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], lr=args.lr_backbone, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"finetune_fold{args.fold}.pth"
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
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"ours_val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= args.patience:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== NTU预训练微调 fold{args.fold} best ours_val = {best:.4f} "
          f"（对照混合增强 0.5447；>0.55 显著则预训练-微调有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
