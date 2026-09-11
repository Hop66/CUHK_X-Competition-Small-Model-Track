#!/usr/bin/env python3
"""CUHK-X —— 跨模态对比学习（main Depth+IR ↔ thermal 双编码器 + InfoNCE）

目标：用 thermal（独立温度视角）的配对数据监督 main，让 main 学到跨模态不变
的动作语义 → 提升 cross-subject 泛化（论文 §6.1.3 证明对比学习有效）。

损失 = CE(main) + CE(thermal) + λ·InfoNCE(z_main, z_thermal)
（同 clip 特征为正对、batch 内其他 clip 为负对）

用法:
    python scripts/train_contrastive.py --backbone r2plus1d34 \
        --weights ig65m_r2plus1d34.pth --lr 1e-4 --epochs 60 --folds 3 \
        --lambda_contrast 0.1 --label_smoothing 0.1 \
        --save_dir outputs/contrastive --crop_cache bbox_train.json \
        --thermal_crop bbox_thermal.json
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
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex,
                         ThermalVideoDataset, build_balanced_sampler,
                         build_train_index)
from src.model import build_model
from src.skeleton_dataset import SkeletonClipIndex, SkeletonVideoDataset
from src.split import split_by_subject
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """main/thermal 同一 index 对齐（同一 clip 的两个模态）。"""

    def __init__(self, main_ds, th_ds):
        self.main_ds = main_ds
        self.th_ds = th_ds

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, i):
        xm, y, subj = self.main_ds[i]
        xt, _, _ = self.th_ds[i]
        return xm, xt, y, subj


def info_nce(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.T / temperature  # [B, B]
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def evaluate_main(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


class SkeletonPairedDataset(Dataset):
    """main 视频 ↔ skeleton 3D 同 index 配对（同一 clip 的两个模态）。"""

    def __init__(self, main_ds, skel_ds):
        self.main_ds = main_ds
        self.skel_ds = skel_ds

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, i):
        xm, y, subj = self.main_ds[i]
        xs, _, _ = self.skel_ds[i]
        return xm, xs, y, subj


class SkeletonEncoder(nn.Module):
    """轻量 3D 骨架编码器：[B,T,17,6] -> [B,D]（1D-CNN over time + GAP）。"""

    def __init__(self, in_dim: int = 17 * 6, hid: int = 256, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(hid, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(hid, out_dim, 5, padding=2), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B,T,17,6]
        B, T, V, C = x.shape
        x = x.reshape(B, T, V * C).permute(0, 2, 1)  # [B, 102, T]
        x = self.net(x)                              # [B, D, T//4]
        return self.pool(x).squeeze(-1)              # [B, D]


def build_skeleton_clips(root: Path, main_clips):
    """从 main clips 构造骨架路径（同 subject split 对齐）。"""
    skel = []
    for c in main_clips:
        action = c.depth_dir.parent.parent.name
        skel_dir = root / "Skeleton" / action / c.subject / c.sample / "predictions"
        skel.append(SkeletonClipIndex(c.action_id, c.subject, c.sample, skel_dir))
    return skel


def train_skeleton_contrastive(args, root, main_clips, device):
    """main↔skeleton 对比：CE(main) + λ·InfoNCE(z_main, z_skel)。

    只用骨架的配对数据监督 main（原生 3D，不投影），只保存 main —— 目标直接提升主线。
    """
    skel_clips = build_skeleton_clips(root, main_clips)
    main_crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    folds = split_by_subject(main_clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]

    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_main = [main_clips[i] for i in tr_idx]
        tr_skel = [skel_clips[i] for i in tr_idx]
        va_main = [main_clips[i] for i in va_idx]

        tr_main_ds = DepthIRVideoDataset(tr_main, args.num_frames, args.size, True, main_crop,
                                         use_frame_diff=False, aug_strength=args.aug_strength,
                                         seed=42)
        tr_skel_ds = SkeletonVideoDataset(tr_skel, args.num_frames, True, seed=42)
        va_main_ds = DepthIRVideoDataset(va_main, args.num_frames, args.size, False, main_crop,
                                         use_frame_diff=False)

        paired = SkeletonPairedDataset(tr_main_ds, tr_skel_ds)
        sampler = build_balanced_sampler([c.action_id for c in tr_main])
        tr_loader = DataLoader(paired, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_main_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        main_model = build_model(args.backbone, num_classes=40, in_channels=4,
                                 n_segment=args.num_frames,
                                 weights_path=args.weights or None).to(device)
        skel_enc = SkeletonEncoder(out_dim=512).to(device)
        params = list(main_model.parameters()) + list(skel_enc.parameters())
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best = 0.0
        save_path = None
        if args.save_dir:
            d = Path(args.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            save_path = d / f"{args.backbone}_contrastive_skel_fold{fi}.pth"

        for ep in range(args.epochs):
            main_model.train()
            skel_enc.train()
            run_loss = run_ce = run_nce = 0.0
            n = 0
            for xm, xs, ym, _ in tr_loader:
                xm, xs = xm.to(device), xs.to(device)
                ym = ym.to(device)
                opt.zero_grad()
                zm = main_model.encoder(xm.permute(0, 2, 1, 3, 4))  # [B,512]
                zs = skel_enc(xs)                                    # [B,512]
                ce = crit(main_model.head(zm), ym)
                nce = info_nce(zm, zs, args.temperature)
                loss = ce + args.lambda_contrast * nce
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(ym)
                run_ce += ce.item() * len(ym)
                run_nce += nce.item() * len(ym)
                n += len(ym)
            sched.step()
            acc = evaluate_main(main_model, va_loader, device)
            if acc > best:
                best = acc
                if save_path is not None:
                    torch.save({"main": main_model.state_dict(),
                                "skel_enc": skel_enc.state_dict()}, save_path)
            print(f"[fold{fi}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"ce={run_ce/max(n,1):.4f} nce={run_nce/max(n,1):.4f} "
                  f"main_val={acc:.4f} best={best:.4f}", flush=True)
        print(f"== fold {fi} best main val = {best:.4f} -> {save_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--backbone", default="r2plus1d34")
    ap.add_argument("--weights", default="", help="IG-65M 预训练权重（R2+1D-34 用）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal.json")
    ap.add_argument("--lambda_contrast", type=float, default=0.1, help="InfoNCE 权重 λ")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--save_dir", default="outputs/contrastive")
    ap.add_argument("--resume", default="", help="续跑：加载该 main/thermal checkpoint 继续训练")
    ap.add_argument("--start_epoch", type=int, default=0, help="续跑起点（scheduler 从该位置继续，cosine warm restart）")
    ap.add_argument("--aux", type=str, default="thermal", choices=["thermal", "skeleton"],
                    help="对比辅助模态：thermal=独立温度视角（旧）；skeleton=原生3D骨架（论文背书 cross-subject 有效）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()

    # main clips 权威（Depth_Color discovery），thermal/skeleton clips 从 main 构造（对齐）
    main_clips = build_train_index(root)
    if args.aux == "skeleton":
        print(f"== main↔skeleton 对比学习（λ={args.lambda_contrast} aug={args.aug_strength}）==", flush=True)
        train_skeleton_contrastive(args, root, main_clips, device)
        return
    th_clips = []
    for c in main_clips:
        th_dir = (root / "Thermal" / c.depth_dir.parent.parent.name / c.subject / c.sample)
        th_clips.append(ThermalClipIndex(c.action_id, c.subject, c.sample, th_dir))

    main_crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))

    folds = split_by_subject(main_clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]

    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_main = [main_clips[i] for i in tr_idx]
        tr_th = [th_clips[i] for i in tr_idx]
        va_main = [main_clips[i] for i in va_idx]
        va_th = [th_clips[i] for i in va_idx]

        tr_main_ds = DepthIRVideoDataset(tr_main, args.num_frames, args.size, True, main_crop,
                                         use_frame_diff=False, aug_strength=args.aug_strength, seed=42)
        tr_th_ds = ThermalVideoDataset(tr_th, args.num_frames, args.size, True, th_crop,
                                       use_frame_diff=False, seed=42)
        va_main_ds = DepthIRVideoDataset(va_main, args.num_frames, args.size, False, main_crop,
                                         use_frame_diff=False)

        paired = PairedDataset(tr_main_ds, tr_th_ds)
        sampler = build_balanced_sampler([c.action_id for c in tr_main])
        tr_loader = DataLoader(paired, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_main_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        main_model = build_model(args.backbone, num_classes=40, in_channels=4,
                                 n_segment=args.num_frames,
                                 weights_path=args.weights or None).to(device)
        th_model = build_model(args.backbone, num_classes=40, in_channels=3,
                               n_segment=args.num_frames,
                               weights_path=args.weights or None).to(device)

        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            main_model.load_state_dict(ckpt["main"])
            th_model.load_state_dict(ckpt["thermal"])
            print(f"resume from {args.resume}（从 ep{args.start_epoch} 续跑）", flush=True)

        params = list(main_model.parameters()) + list(th_model.parameters())
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        # 续跑 = cosine warm restart：T_max=args.epochs（新总数），scheduler 预步进到 start_epoch 位置
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for _ in range(args.start_epoch):
            sched.step()
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best = 0.0
        save_path = None
        if args.save_dir:
            d = Path(args.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            tag = "_cont" if args.resume else ""
            save_path = d / f"{args.backbone}_contrastive_fold{fi}{tag}.pth"

        for ep in range(args.start_epoch, args.epochs):
            main_model.train()
            th_model.train()
            run_loss = run_ce = run_nce = 0.0
            n = 0
            for xm, xt, ym, _ in tr_loader:
                xm, xt = xm.to(device), xt.to(device)
                ym = ym.to(device)
                opt.zero_grad()
                # R2Plus1D forward 内部有 permute，但 encoder 不会：需手动 [B,T,C,H,W]->[B,C,T,H,W]
                zm = main_model.encoder(xm.permute(0, 2, 1, 3, 4))
                zt = th_model.encoder(xt.permute(0, 2, 1, 3, 4))
                pm = main_model.head(zm)
                pt = th_model.head(zt)
                ce = crit(pm, ym) + crit(pt, ym)
                nce = info_nce(zm, zt, args.temperature)
                loss = ce + args.lambda_contrast * nce
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(ym)
                run_ce += ce.item() * len(ym)
                run_nce += nce.item() * len(ym)
                n += len(ym)
            sched.step()
            acc = evaluate_main(main_model, va_loader, device)
            if acc > best:
                best = acc
                if save_path is not None:
                    torch.save({"main": main_model.state_dict(),
                                "thermal": th_model.state_dict()}, save_path)
            print(f"[fold{fi}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"ce={run_ce/max(n,1):.4f} nce={run_nce/max(n,1):.4f} "
                  f"main_val={acc:.4f} best={best:.4f}", flush=True)
        print(f"== fold {fi} best main val = {best:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
