#!/usr/bin/env python3
"""Thermal 双流训练：S 静态热像流（大框 per-frame logits mean）+ M 骨架运动流（1D conv 时序）。

架构（用户定案）:
  S 流:  热像帧大框 crop(bbox_thermal_train.json margin1.4) → FrameNet / resnet18/34 → per-frame logits mean
  M 流:  骨架运动缓存 motion_cache.pkl [N,29] → resample [T,29] → MotionNet(1D conv) → 运动 logits
  融合:  fused = α·softmax(S) + (1-α)·softmax(M)，α 可学/固定(默认 learn)
           loss = CE(S) + CE(M) + CE(fused)（各自学好再融合，防偷懒）
对照:  S-only ≈ 0.3863(crop 版已有) / thermal 3D 0.6339
判读:  SM > S → 骨架运动有真增益；M-only >> 0 → 运动流本身有判别力；
       SM 逼近/超过 0.6339 → thermal 双流成立 → 并入 0.73 融合冲 0.8
用法:  python scripts/train_thermal_dual.py --fold 0 --arch resnet18 --use_static --use_motion ...
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

from src.dataset import build_balanced_sampler, build_thermal_index
from src.split import split_by_subject
from src.framenet import FrameNet, build_frame_net_pretrained
from src.skeleton_motion import K_DIM
from src.thermal_dual_dataset import ThermalDualDataset


class MotionNet(nn.Module):
    """骨架运动流：motion[T,K] → 1D conv 保留时序 → 时间聚合 → logits。"""

    def __init__(self, in_dim: int = K_DIM, hid: int = 64, num_classes: int = 40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hid, 5, padding=2, bias=False), nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid, hid, 5, padding=2, bias=False), nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True))
        self.head = nn.Linear(hid, num_classes)

    def forward(self, m):                    # m: [B,T,29]
        z = self.conv(m.permute(0, 2, 1))    # [B,hid,T]
        return self.head(z.mean(dim=2))      # [B,40]


class ThermalDual(nn.Module):
    def __init__(self, arch: str, use_static: bool, use_motion: bool, alpha_mode: str = "learn"):
        super().__init__()
        self.use_static = use_static
        self.use_motion = use_motion
        self.static = (build_frame_net_pretrained(arch, 40)
                       if arch != "framenet" else FrameNet(40)) if use_static else None
        self.motion = MotionNet() if use_motion else None
        self.a = nn.Parameter(torch.tensor(0.0)) if (use_static and use_motion
                                                     and alpha_mode == "learn") else None

    def forward(self, x, t):
        s = self.static(x) if self.use_static else None          # [B,40] 或 None
        m = self.motion(t) if self.use_motion else None          # [B,40] 或 None
        if s is not None and m is not None:
            alpha = 0.5 if self.a is None else torch.sigmoid(self.a)
            p = alpha * torch.softmax(s, -1) + (1 - alpha) * torch.softmax(m, -1)
            return torch.log(p.clamp_min(1e-8)), dict(s=s, m=m)  # fused logits
        if s is not None:
            return s, dict(s=s)
        return m, dict(m=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_cache", default="bbox_thermal_train.json")
    ap.add_argument("--motion_cache", default="outputs/motion_cache.pkl")
    ap.add_argument("--arch", default="resnet18", choices=["framenet", "resnet18", "resnet34"])
    ap.add_argument("--use_static", action="store_true")
    ap.add_argument("--use_motion", action="store_true")
    ap.add_argument("--gray_norm", action="store_true")
    ap.add_argument("--alpha", default="learn", choices=["learn", "fixed"])
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--save_dir", default="outputs/thermal_dual")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.train_root).expanduser()
    clips = build_thermal_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_c, va_c = [clips[i] for i in tr_idx], [clips[i] for i in va_idx]

    crop_cache = {}
    if args.crop_cache and Path(args.crop_cache).expanduser().exists():
        import json as _json
        crop_cache = _json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    motion_cache = {}
    if args.use_motion:
        p = Path(args.motion_cache).expanduser()
        if p.exists():
            with open(p, "rb") as f:
                motion_cache = pickle.load(f)
            keys = {f"{c.action_id}/{c.subject}/{c.sample}" for c in clips}
            cov = sum(1 for k in keys if k in motion_cache) / max(len(keys), 1)
            print(f"[dual] motion_cache clips={len(motion_cache)} thermal覆盖={cov:.3f}", flush=True)
        else:
            print(f"⚠️ 缺运动缓存 {p}（先跑 extract_motion_cache.py），M 流将全 valid=0", flush=True)

    ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
    tr_ds = ThermalDualDataset(tr_c, args.num_frames, args.size, True, crop_cache, motion_cache,
                               sample_mode="segment", mean_std=ms, seed=args.seed)
    va_ds = ThermalDualDataset(va_c, args.num_frames, args.size, False, crop_cache, motion_cache,
                               sample_mode="segment", mean_std=ms, seed=args.seed)
    print(f"[dual] train覆盖={tr_ds.coverage():.3f} val覆盖={va_ds.coverage():.3f}", flush=True)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    tr_loader = DataLoader(tr_ds, sampler=build_balanced_sampler([c.action_id for c in tr_c]),
                           drop_last=True, **loader_kw)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_kw)

    model = ThermalDual(args.arch, args.use_static, args.use_motion, args.alpha).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{args.arch}_{'S' if args.use_static else ''}{'M' if args.use_motion else ''}"
    print(f"[dual] {tag} params={n_params / 1e6:.2f}M | static={args.use_static} "
          f"motion={args.use_motion} alpha={args.alpha}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def forward_batch(b):
        x = b[0].to(device)
        t = b[1].to(device)
        y = b[3].to(device).long()
        out, comp = model(x, t)
        return out, comp, y

    def step_loss(out, comp, y):
        if "s" in comp and "m" in comp:
            return crit(out, y) + crit(comp["s"], y) + crit(comp["m"], y)
        return crit(out, y)

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for b in va_loader:
                out, _, y = forward_batch(b)
                correct += (out.argmax(-1) == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{tag}_fold{args.fold}.pth"
    best, no_impr = 0.0, 0
    for ep in range(args.epochs):
        model.train()
        run_loss = n = 0
        for b in tr_loader:
            out, comp, y = forward_batch(b)
            opt.zero_grad()
            loss = step_loss(out, comp, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate()
        if acc > best:
            best, no_impr = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_impr += 1
        print(f"[fold{args.fold}] ep{ep + 1}/{args.epochs} loss={run_loss / max(n, 1):.4f} "
              f"val={acc:.4f} best={best:.4f}", flush=True)
        if no_impr >= 15:
            print(f"[fold{args.fold}] early stop @ ep{ep + 1}", flush=True)
            break
    print(f"== ThermalDual[{tag}] fold{args.fold} best = {best:.4f} "
          f"（对照 S-only≈0.386 / thermal 3D 0.6339）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
