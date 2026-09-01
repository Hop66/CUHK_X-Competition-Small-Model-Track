#!/usr/bin/env python3
"""Thermal 3D 双流训练：S=热像 R2+1D-34(IG65M 3ch) + M=骨架运动（同源物理补充，3D 侧待补空白）

已做对照：thermal 2D 静态+骨架=0.4429(+4.2pt) / main 3D+骨架=0.6652(+0.4pt) / thermal 3D(无骨架)=0.6083。
本脚本补上"thermal 3D + 骨架"（按强宿主规律预期 +0.4pt 级）+ th3D-only 复核。
架构：S=R2+1D-34(3ch,Kinetics norm, uniform16,128) | M=MotionNet(motion[T,29])
      融合 fused=α·softmax(S)+(1-α)·softmax(M)，α 可学；loss=CE(S)+CE(M)+CE(fused)
判读：[SM] > [S] → 骨架辅助 thermal 3D 真增益（并入 0.73 融合）；否则骨架只留 main 侧
用法: python scripts/train_thermal3d_dual.py --fold 0 [--no_motion] ...
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.backbone_compare import build_backbone
from src.dataset import build_balanced_sampler, build_thermal_index
from src.motion_net import MotionNet
from src.split import split_by_subject
from src.thermal_dual_dataset import ThermalDualDataset


class Thermal3DDual(nn.Module):
    def __init__(self, weights_path, use_motion=True, alpha_mode="learn"):
        super().__init__()
        self.static = build_backbone("r2plus1d34", num_classes=40, in_channels=3,
                                     weights_path=weights_path, traj_dim=0)
        self.motion = MotionNet() if use_motion else None
        self.a = nn.Parameter(torch.tensor(0.0)) if (use_motion and alpha_mode == "learn") else None

    def forward(self, x, t):                    # x:[B,T,3,H,W] | t:[B,T,29]
        s = self.static(x)                      # [B,40]
        if self.motion is None:
            return s, dict(s=s)
        m = self.motion(t)                      # [B,40]
        alpha = 0.5 if self.a is None else torch.sigmoid(self.a)
        p = alpha * torch.softmax(s, -1) + (1 - alpha) * torch.softmax(m, -1)
        return torch.log(p.clamp_min(1e-8)), dict(s=s, m=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_cache", default="bbox_thermal_train.json")
    ap.add_argument("--motion_cache", default="outputs/motion_cache.pkl")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--no_motion", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--save_dir", default="outputs/thermal3d_dual")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.train_root).expanduser()
    clips = build_thermal_index(root)                      # thermal 2891
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_c, va_c = [clips[i] for i in tr_idx], [clips[i] for i in va_idx]

    crop_cache = {}
    if args.crop_cache and Path(args.crop_cache).expanduser().exists():
        import json as _json
        crop_cache = _json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    motion_cache = {}
    if not args.no_motion:
        p = Path(args.motion_cache).expanduser()
        if p.exists():
            with open(p, "rb") as f:
                motion_cache = pickle.load(f)
            print(f"[th3d_dual] motion_cache={len(motion_cache)}", flush=True)
        else:
            print(f"⚠️ 缺运动缓存 {p}", flush=True)

    tr_ds = ThermalDualDataset(tr_c, args.num_frames, args.size, True, crop_cache, motion_cache,
                               sample_mode="uniform", seed=args.seed)
    va_ds = ThermalDualDataset(va_c, args.num_frames, args.size, False, crop_cache, motion_cache,
                               sample_mode="uniform", seed=args.seed)
    print(f"[th3d_dual] train覆盖={tr_ds.coverage():.3f} val覆盖={va_ds.coverage():.3f}", flush=True)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    tr_loader = DataLoader(tr_ds, sampler=build_balanced_sampler([c.action_id for c in tr_c]),
                           drop_last=True, **loader_kw)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_kw)

    model = Thermal3DDual(weights_path=args.weights, use_motion=not args.no_motion).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tag = "S" if args.no_motion else "SM"
    print(f"[th3d_dual] {tag} params={n_params / 1e6:.1f}M | motion={not args.no_motion}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def fwd(b):
        x = b[0].to(device)
        t = b[1].to(device)
        y = b[3].to(device).long()
        return model(x, t), y

    def step_loss(out, comp, y):
        if "s" in comp and "m" in comp:
            return crit(out, y) + crit(comp["s"], y) + crit(comp["m"], y)
        return crit(out, y)

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for b in va_loader:
                (out, _), y = fwd(b)
                correct += (out.argmax(-1) == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"th3d_{'SM' if not args.no_motion else 'S'}_fold{args.fold}.pth"
    best, no_impr = 0.0, 0
    for ep in range(args.epochs):
        model.train()
        run_loss = n = 0
        for b in tr_loader:
            (out, comp), y = fwd(b)
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
    print(f"== Thermal3DDual[{tag}] fold{args.fold} best = {best:.4f} "
          f"（对照 th3D 基线 0.6083 — 0.6339）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
