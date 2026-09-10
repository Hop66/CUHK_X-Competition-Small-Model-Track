#!/usr/bin/env python3
"""骨架 KD 蒸馏 AB(fold0): 用强模态(main/th)在 train 上的 soft teacher 指导骨架学习。

对照: native finetune_fold0.py standalone 0.5318, 低权α融合 0.6740(三路)
判据: KD骨架 standalone & 低权融合是否 > native → 蒸馏对融合有无增益(正交性 vs 单模质量)
用法:
    python scripts/train_ntu_kd.py --fold 0 \
        --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth \
        --teacher outputs/teacher/teacher_main_train.pkl \
        --out outputs/skeleton_kd_fold0.pth
"""
import argparse
import pickle
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.motionbert.action_net import ActionHeadClassification
from src.motionbert.dstformer import DSTformer
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.split import split_by_subject

from train_skeleton_motionbert import BASE
from train_ntu_finetune import FinetuneNet, build_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ntu_ckpt", required=True)
    ap.add_argument("--teacher", required=True, help="teacher soft probs pkl {key: probs[40]}")
    ap.add_argument("--teacher2", default="", help="可选第二个 teacher(平均)")
    ap.add_argument("--out", default="outputs/skeleton_kd_fold0.pth")
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--kd_lambda", type=float, default=0.5, help="KD 权重")
    ap.add_argument("--kd_temp", type=float, default=1.0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_kd")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()

    teach = pickle.load(open(args.teacher, "rb"))
    if args.teacher2:
        t2 = pickle.load(open(args.teacher2, "rb"))
        for k in teach:
            if k in t2:
                teach[k] = 0.5 * torch.tensor(teach[k]) + 0.5 * torch.tensor(t2[k])
    print(f"teacher keys={len(teach)} lam={args.kd_lambda} T={args.kd_temp}", flush=True)

    clips = build_skeleton_index(root)
    folds = split_by_subject(clips, n_folds=3)
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
    print(f"KD微调: train={len(tr_ds)} val={len(va_ds)}", flush=True)

    backbone = build_backbone(Path(args.ntu_ckpt).expanduser(), device)
    model = FinetuneNet(backbone, dim_rep=BASE["dim_rep"], num_classes=40, hidden_dim=512).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], lr=args.lr_backbone, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    kd_t, lam = args.kd_temp, args.kd_lambda

    Path(args.save_dir).expanduser().mkdir(parents=True, exist_ok=True)
    save_path = Path(args.out).expanduser()
    best, no_improve, t_eps = 0.0, 0, 0.0
    teacher_use = 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for bi, (x, y, _) in enumerate(tr_loader):
            x, y = x.to(device).unsqueeze(1), y.to(device)
            opt.zero_grad()
            out = model(x)                      # [N,40]
            log_p = F.log_softmax(out / kd_t, -1)
            key0 = f"{tr_clips[bi*args.batch_size].action_id}/{tr_clips[bi*args.batch_size].subject}/{tr_clips[bi*args.batch_size].sample}"
            # teacher 按样本 batch 对齐 (dataset index 顺序 == tr_clips 顺序)
            kd_losses = []
            for j in range(len(x)):
                c = tr_clips[bi * args.batch_size + j]
                k = f"{c.action_id}/{c.subject}/{c.sample}"
                if k in teach:
                    t = torch.tensor(teach[k], dtype=torch.float32, device=device).clamp(1e-7, 1)
                    kd_losses.append(-(t * log_p[j]).sum())
            ce = crit(out, y)
            kd = torch.stack(kd_losses).mean() if kd_losses else torch.zeros((), device=device)
            teacher_use += len(kd_losses)
            loss = (1 - lam) * ce + lam * (kd_t ** 2) * kd
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        t_eps = time.time() - t0
        acc = 0.0
        model.eval()
        with torch.no_grad():
            cc = tt = 0
            for x, y, _ in va_loader:
                x = x.to(device).unsqueeze(1)
                cc += (model(x).argmax(-1) == y.to(device)).sum().item()
                tt += y.numel()
            acc = cc / max(tt, 1)
        if acc > best:
            best, no_improve = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[KD fold{args.fold}] ep{ep+1} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} kd_use/pass={teacher_use} ({t_eps:.0f}s)", flush=True)
        if no_improve >= args.patience:
            break
    print(f"==== KD DONE: {save_path} best={best:.4f} ====")


if __name__ == "__main__":
    main()
