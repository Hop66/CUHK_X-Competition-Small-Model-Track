#!/usr/bin/env python3
"""CUHK-X —— main(教师)→thermal(学生) 软标签蒸馏（XTinyHAR 式跨模态蒸馏）

方向：强教师(main Depth+IR 0.66-0.71, 冻结) 教 弱学生(thermal 独立视角)。
  → thermal 学到 main 的动作语义，突破自身 0.62 recipe 上限 → 独立视角更强
  → 与 main 集成时互补更有底气。
损失 = CE(thermal, y) + λ·KL(softmax(teacher/T), softmax(student/T))
用法:
    python scripts/train_distill_thermal.py --weights ig65m_r2plus1d34.pth \
        --teacher_ckpt outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth \
        --lr 1e-4 --epochs 60 --fold 0 \
        --crop_cache bbox_train.json --thermal_crop bbox_thermal_train.json \
        --lambda_kl 1.0 --T 4.0 --save_dir outputs/distill_thermal
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex, ThermalVideoDataset,
                         build_balanced_sampler, build_train_index)
from src.model import build_model
from src.split import split_by_subject


class PairedDataset(Dataset):
    """main/thermal 同 index 对齐（同一 clip 两模态）。"""

    def __init__(self, main_ds, th_ds):
        self.main_ds = main_ds
        self.th_ds = th_ds

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, i):
        xm, y, subj = self.main_ds[i]
        xt, _, _ = self.th_ds[i]
        return xm, xt, y, subj


def kl_distill(teacher_logits, student_logits, T=4.0):
    p = F.log_softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") * T * T


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
    ap.add_argument("--backbone", default="r2plus1d34")
    ap.add_argument("--weights", default="", help="IG-65M 预训练权重")
    ap.add_argument("--teacher_ckpt", default="outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth",
                    help="main 教师 checkpoint（冻结）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--lambda_kl", type=float, default=1.0, help="KL 蒸馏权重 λ")
    ap.add_argument("--T", type=float, default=4.0, help="蒸馏温度")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--save_dir", default="outputs/distill_thermal")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    th_clips = []
    for c in main_clips:
        th_dir = root / "Thermal" / c.depth_dir.parent.parent.name / c.subject / c.sample
        th_clips.append(ThermalClipIndex(c.action_id, c.subject, c.sample, th_dir))
    main_crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
    print(f"device={device} λ_kl={args.lambda_kl} T={args.T} clips={len(main_clips)}", flush=True)

    # 教师：main（冻结）
    teacher = build_model(args.backbone, num_classes=40, in_channels=4,
                          n_segment=args.num_frames).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher loaded: {args.teacher_ckpt}", flush=True)

    folds = split_by_subject(main_clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_main = [main_clips[i] for i in tr_idx]
        tr_th = [th_clips[i] for i in tr_idx]
        va_th = [th_clips[i] for i in va_idx]

        tr_main_ds = DepthIRVideoDataset(tr_main, args.num_frames, args.size, True, main_crop,
                                         use_frame_diff=False, aug_strength=args.aug_strength, seed=42)
        tr_th_ds = ThermalVideoDataset(tr_th, args.num_frames, args.size, True, th_crop,
                                       use_frame_diff=False, aug_strength=args.aug_strength, seed=42)
        va_th_ds = ThermalVideoDataset(va_th, args.num_frames, args.size, False, th_crop,
                                       use_frame_diff=False)

        paired = PairedDataset(tr_main_ds, tr_th_ds)
        sampler = build_balanced_sampler([c.action_id for c in tr_main])
        tr_loader = DataLoader(paired, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_th_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        student = build_model(args.backbone, num_classes=40, in_channels=3,
                              n_segment=args.num_frames,
                              weights_path=args.weights or None).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best = 0.0
        save_path = save_dir / f"{args.backbone}_distill_th_fold{fi}.pth"
        for ep in range(args.epochs):
            student.train()
            run_loss = run_ce = run_kl = 0.0
            n = 0
            for xm, xt, ym, _ in tr_loader:
                xm, xt, ym = xm.to(device), xt.to(device), ym.to(device)
                opt.zero_grad()
                with torch.no_grad():
                    t_logits = teacher(xm)
                s_logits = student(xt)
                ce = crit(s_logits, ym)
                kl = kl_distill(t_logits, s_logits, args.T)
                loss = ce + args.lambda_kl * kl
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(ym)
                run_ce += ce.item() * len(ym)
                run_kl += kl.item() * len(ym)
                n += len(ym)
            sched.step()
            acc = evaluate(student, va_loader, device)
            if acc > best:
                best = acc
                torch.save(student.state_dict(), save_path)
            print(f"[fold{fi}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"ce={run_ce/max(n,1):.4f} kl={run_kl/max(n,1):.4f} "
                  f"th_val={acc:.4f} best={best:.4f}", flush=True)
        print(f"== fold {fi} best thermal val = {best:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
