#!/usr/bin/env python3
"""骨架 full 全量训练（见过全部训练数据的 Skeleton clips，无 val，供 test 推理 + 3成员链）。

协议与 train_ntu_finetune.py fold 版完全一致(balanced + nf24 + 同 backbone + 同超参)，
只是不分区、无 val、跑满 epochs 保存(final)。
用法: python scripts/train_ntu_finetune_full.py --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth
产物: outputs/skeleton_finetune_full/finetune_full.pth
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from train_skeleton_motionbert import BASE
from train_ntu_finetune import FinetuneNet, build_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ntu_ckpt", required=True)
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_finetune_full")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    ds = MotionBertSkeletonDataset(clips, args.num_frames, True, input3d=True)
    sampler = build_balanced_sampler([c.action_id for c in clips])
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"full 骨架: train={len(ds)} device={device}", flush=True)

    backbone = build_backbone(Path(args.ntu_ckpt).expanduser(), device)
    model = FinetuneNet(backbone, dim_rep=BASE["dim_rep"], num_classes=40, hidden_dim=512).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], lr=args.lr_backbone, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    sd = Path(args.save_dir).expanduser()
    sd.mkdir(parents=True, exist_ok=True)
    save_path = sd / "finetune_full.pth"
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device).unsqueeze(1), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        if ep == args.epochs - 1:
            torch.save({"model": model.state_dict(), "best_acc": -1.0}, save_path)
        print(f"[FULL] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    print(f"==== SKEL FULL DONE: {save_path} ====")


if __name__ == "__main__":
    main()
