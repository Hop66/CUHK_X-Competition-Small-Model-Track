#!/usr/bin/env python3
"""蒸馏训练：单流 student ← multi-seed soft 教师（Hinton KD, 温度 T）。

用法（maintance 训练/AB）：
  # 先出教师软标签：
  #   python scripts/prep_distill_teacher.py --modality main \\
  #       --ckpts outputs/pack/main_s42_fold0_int5.pth \\ （+ quantize/p777/avg...）
  #       --root train --out outputs/teacher_main_train.pkl --quantize --flip
  # 再训 student：
  #   python scripts/train_distill_stream.py --modality main \\
  #       --teacher outputs/teacher_main_train.pkl --full --epochs 80 \\
  #       --distill_w 0.5 --T 3.0 --save_dir outputs/main_distill --seed 42
"""
import argparse
import pickle
import sys
from pathlib import Path

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


def load_teacher(pkl):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    print(f"[distill] teacher keys={len(d)}", flush=True)
    return d


def build_dataloaders(args, device):
    root = Path(args.train_root).expanduser()
    import json
    crop_cache = json.loads(Path(args.crop).read_text(encoding="utf-8"))
    if args.modality == "main":
        clips = build_train_index(root)
    else:
        clips = build_thermal_index(root)
    if args.full:
        tr_c, va_c = clips, []
    else:
        folds = split_by_subject(clips, n_folds=args.folds)
        tr_idx, va_idx = folds[args.fold]
        tr_c = [clips[i] for i in tr_idx]
        va_c = [clips[i] for i in va_idx]
    if args.modality == "main":
        mk = lambda cc: DepthIRVideoDataset(cc, args.num_frames, args.size, False,
                                            crop_cache, use_frame_diff=False,
                                            sample_mode="uniform", seed=args.seed,
                                            return_key=True)
    else:
        mk = lambda cc: ThermalVideoDataset(cc, args.num_frames, args.size, False,
                                            crop_cache, use_frame_diff=False,
                                            sample_mode="uniform", seed=args.seed,
                                            return_key=True)
    tr_ds = mk(tr_c)
    va_ds = mk(va_c) if va_c else None
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                           sampler=build_balanced_sampler([c.action_id for c in tr_c]),
                           num_workers=args.workers, pin_memory=True, drop_last=True,
                           timeout=300)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True, timeout=300) if va_ds else None
    return tr_loader, va_loader, tr_ds, va_ds


def evaluate(model, loader, device):
    model.eval()
    nc = nt = 0
    with torch.no_grad():
        for b in loader:
            x = b[0].to(device); y = b[1]
            out = model(x)
            nc += (out.argmax(1).cpu() == y).sum().item(); nt += y.numel()
    return nc / max(nt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["main", "thermal"], required=True)
    ap.add_argument("--teacher", required=True, help="软标签 pkl")
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--distill_w", type=float, default=0.5, help="KL 项权重")
    ap.add_argument("--T", type=float, default=3.0, help="蒸馏温度")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--crop", default="bbox_train.json")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth",
                    help="主干预训练权重(与 train_step1 一致, 蒸馏也须 IG65M 初始才能收敛好)")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full_save_every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args.teacher)
    tr_loader, va_loader, tr_ds, va_ds = build_dataloaders(args, device)

    if args.modality == "main":
        model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                            n_segment=args.num_frames,
                            weights_path=args.weights).to(device)
    else:
        model = build_model("r2plus1d34", num_classes=40, in_channels=3,
                            n_segment=args.num_frames,
                            weights_path=args.weights).to(device)
    print(f"[distill] 使用主干初始化: {args.weights}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    tag = "full" if args.full else f"fold{args.fold}"
    save_path = Path(args.save_dir) / f"{args.modality}_distill_{tag}_seed{args.seed}.pth"

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        tl = 0.0; n = 0
        for b in tr_loader:
            x = b[0].to(device); y = b[1].to(device)
            keys = b[-1]
            x = x.clone()
            # 软目标对齐（数据集增广下同一 key 用同一教师）；跨模态/缺教师 key 回退 one-hot
            yl = y.tolist()
            tv = []
            for k, yi in zip(keys, yl):
                if k in teacher:
                    tv.append(torch.from_numpy(teacher[k]))
                else:
                    oh = torch.zeros(40); oh[yi] = 1.0
                    tv.append(oh)
            t_prob = torch.stack(tv).to(device)
            out = model(x)
            l_hard = ce(out, y)
            l_kl = F.kl_div(F.log_softmax(out / args.T, -1),
                            F.softmax(t_prob / args.T, -1), reduction="batchmean")
            loss = l_hard + args.distill_w * (args.T ** 2) * l_kl
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        msg = f"[{args.modality}|{tag}] ep{ep + 1}/{args.epochs} loss={tl / max(n, 1):.4f}"
        if va_loader is not None:
            acc = evaluate(model, va_loader, device)
            msg += f" val={acc:.4f}"
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "best_acc": best,
                            "epoch": ep + 1}, save_path)
        else:
            if (ep + 1) % args.full_save_every == 0:
                torch.save({"model": model.state_dict(), "best_acc": 0.0,
                            "epoch": ep + 1}, save_path)
            if ep + 1 == args.epochs:
                torch.save({"model": model.state_dict(), "best_acc": 0.0,
                            "epoch": ep + 1}, save_path)
        print(msg, flush=True)
    print(f"[distill] DONE best={best:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
