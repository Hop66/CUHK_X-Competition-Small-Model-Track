#!/usr/bin/env python3
"""Transductive 软蒸馏 fold 门控 —— 判定"训练时把未标注折用教师软标签加入"是否真增益。

思路: fold f 上, student 同时优化:
  1) 训练折:a*CE(硬标签) + b*T² KL(softmax/T || 教师/ T)   [常规蒸馏]
  2) val 折(当"未标注测试集"): 仅  b²*T² KL(softmax/T || 教师/ T)  [transductive 注入]
评价: val 折 acc, 对照"不加 val 折"(纯 fold 蒸馏) → 差值决定是否走向真测试 transductive-soft。
注意: 教师软标签含 val 折(key 在 teacher_train.pkl 里), 正是真实 transductive(教师全量训练过)。

用法(fold 门控; 每折最外层):
  python scripts/train_distill_transductive.py --modality main \
    --teacher outputs/teacher/teacher_main_train.pkl --fold 0 --epochs 60 \
    --save_dir outputs/d_trans --tag f0
"""
import argparse
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalVideoDataset,
                         build_balanced_sampler, build_train_index,
                         build_thermal_index)
from src.split import split_by_subject
from src.model import build_model


def load_t(p):
    return pickle.load(open(p, "rb"))


def evaluate(model, loader, device, flip=True):
    model.eval()
    nc = nt = 0
    with torch.no_grad():
        for b in loader:
            x = b[0].to(device); y = b[1]
            o = model(x)
            if flip:
                o = o + model(torch.flip(x, dims=(-1,)))
            nc += (o.argmax(1).cpu() == y).sum().item(); nt += y.numel()
    return nc / max(nt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["main", "thermal"], default="main")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--distill_w", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--crop", default="bbox_train.json")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--tag", default="f")
    ap.add_argument("--no_trans", action="store_true", help="对照: 不加 val 折软标签")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_t(args.teacher)
    import json
    crop_cache = json.loads(Path(args.crop).read_text(encoding="utf-8"))
    root = Path(args.train_root).expanduser()
    clips = build_thermal_index(root) if args.modality == "thermal" else build_train_index(root)
    tr_idx, va_idx = split_by_subject(clips, n_folds=3)[args.fold]
    tr_c = [clips[i] for i in tr_idx]; va_c = [clips[i] for i in va_idx]

    if args.modality == "main":
        mk = lambda cc: DepthIRVideoDataset(cc, args.num_frames, args.size, True,
                                            crop_cache, use_frame_diff=False,
                                            sample_mode="uniform", seed=args.seed, return_key=True)
        model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                            n_segment=args.num_frames, weights_path=args.weights).to(device)
    else:
        mk = lambda cc: ThermalVideoDataset(cc, args.num_frames, args.size, True,
                                            crop_cache, use_frame_diff=False,
                                            sample_mode="uniform", seed=args.seed, return_key=True)
        model = build_model("r2plus1d34", num_classes=40, in_channels=3,
                            n_segment=args.num_frames, weights_path=args.weights).to(device)

    tr_ds = mk(tr_c)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                           sampler=build_balanced_sampler([c.action_id for c in tr_c]),
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    # 教师软标签也允许 val 折加入训练队列(transductive)
    te_un_ds = mk(va_c) if not args.no_trans else None
    te_un_loader = (DataLoader(te_un_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=False)
                    if te_un_ds else None)
    va_ev = DataLoader(
        (ThermalVideoDataset if args.modality == "thermal" else DepthIRVideoDataset)(
            va_c, args.num_frames, args.size, False, crop_cache, use_frame_diff=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def feed_loss(b, w_hard):
        x, y, keys = b[0].to(device), b[1].to(device), b[-1]
        out = model(x)
        l = ce(out, y) * w_hard
        tp = torch.stack([torch.from_numpy(teacher[k]) for k in keys]).to(device)
        l += w_hard * args.distill_w * (args.T ** 2) * F.kl_div(
            F.log_softmax(out / args.T, -1), F.softmax(tp / args.T, -1), reduction="batchmean")
        return l

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train(); tot = 0.0; cnt = 0
        for b in tr_loader:
            loss = feed_loss(b, 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += len(b[0])
        if te_un_loader is not None:          # transductive: 未标注折仅软 KL
            for b in te_un_loader:
                x = b[0].to(device); keys = b[-1]
                out = model(x)
                tp = torch.stack([torch.from_numpy(teacher[k]) for k in keys]).to(device)
                loss = args.distill_w * (args.T ** 2) * F.kl_div(
                    F.log_softmax(out / args.T, -1), F.softmax(tp / args.T, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward(); opt.step()
                tot += 0.5 * loss.item(); cnt += len(x)
        sched.step()
        acc = evaluate(model, va_ev, device)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "best_acc": best, "fold": args.fold},
                       f"{args.save_dir}/{args.modality}_dtrans_{args.tag}_fold{args.fold}.pth")
        if (ep + 1) % 10 == 0:
            print(f"[fold{args.fold}|{'trans' if not args.no_trans else 'plain'}] "
                  f"ep{ep+1}/{args.epochs} val={acc:.4f} best={best:.4f}", flush=True)
    print(f"== fold{args.fold} trans={'no' if args.no_trans else 'yes'} best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
