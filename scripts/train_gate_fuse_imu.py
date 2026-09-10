#!/usr/bin/env python3
"""IMU 特征门控融合(骨架门控基础上加 IMU 分支, 三路 gate)。
f_v(512) + fs(SkeletonBiGRU) + fi(IMU-CNN) ; g_s=σ(Ws[cv,fs]), g_i=σ(Wi[cv,fi])
f = fv + g_s⊙proj_s(fs) + g_i⊙proj_i(fi) → head
fold0 main nf16 + skel + IMU(门控) vs 0.6695。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import build_train_index, DepthIRVideoDataset
from src.imu_dataset import time_align, load_imu_sequence
from src.model import build_model
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject


class IMUBranch(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(30, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())  # [B,128]

    def forward(self, x):            # [B,T,30]
        f = self.cnn(x.permute(0, 2, 1))       # [B,128,T]
        return self.head(f)                    # [B,128]


def imu_seq(imu_dir, T=64):
    dev = load_imu_sequence(Path(imu_dir))
    x = time_align(dev, T=T)
    if x is None or not np.any(x):
        return None
    return x.astype(np.float32)


def skel_seq(pred_dir, T=16):
    from train_gate_fuse import skel_seq as _s
    return _s(pred_dir, T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--save_dir", default="outputs/main_gate_bi")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    from src.dataset import build_train_index as _b
    from src.imu_dataset import build_imu_index
    clips = _b(root)
    folds = split_by_subject(clips, 3)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    skel_map = {f"{c.action_id}/{c.subject}/{c.sample}": str(c.pred_dir)
                for c in build_skeleton_index(root)}
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(root, clips)}
    imu_key = {f"{c.action_id}/{c.subject}/{c.sample}": str(c.imu_dir)
               for c in build_imu_index(root, clips)}
    print(f"fold{args.fold} tr={len(tr_clips)} va={len(va_clips)}", flush=True)

    outdir = Path(args.save_dir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)
    crop_cache = json.loads(Path("bbox_train.json").read_text(encoding="utf-8"))
    tr_ds = DepthIRVideoDataset(tr_clips, args.num_frames, 128, True, crop_cache,
                                aug_strength=2, return_key=True)
    va_ds = DepthIRVideoDataset(va_clips, args.num_frames, 128, False, crop_cache,
                                return_key=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    main_model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                             weights_path=args.weights or None).to(device)
    skel_branch = __import__("train_gate_fuse").SkeletonBranch().to(device)
    imu_branch = IMUBranch().to(device)
    gate_s = nn.Sequential(nn.Linear(512 + 128, 1), nn.Sigmoid()).to(device)
    gate_i = nn.Sequential(nn.Linear(512 + 128, 1), nn.Sigmoid()).to(device)
    proj_s = nn.Sequential(nn.Linear(128, 512), nn.ReLU()).to(device)
    proj_i = nn.Sequential(nn.Linear(128, 512), nn.ReLU()).to(device)
    head = nn.Linear(512, 40).to(device)
    params = (list(main_model.parameters()) + list(skel_branch.parameters())
              + list(imu_branch.parameters()) + list(gate_s.parameters())
              + list(gate_i.parameters()) + list(proj_s.parameters())
              + list(proj_i.parameters()) + list(head.parameters()))
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    feats = {}
    def hook(mod, i, o):
        feats["fv"] = o
    main_model.encoder.register_forward_hook(hook)
    sk_cache, im_cache = {}, {}

    def get_sk(keys):
        out = []
        for k in keys:
            if k not in sk_cache:
                s = skel_seq(skel_map.get(k, ""))
                sk_cache[k] = torch.from_numpy(s) if s is not None else torch.zeros(args.num_frames, 17, 6)
            out.append(sk_cache[k])
        return torch.stack(out).to(device)

    def get_im(keys):
        out = []
        for k in keys:
            if k not in im_cache:
                s = imu_seq(imu_key.get(k, ""))
                im_cache[k] = torch.from_numpy(s) if s is not None else torch.zeros(64, 30)
            out.append(im_cache[k])
        return torch.stack(out).to(device)

    def fused(x, s, im):
        feats.clear()
        _ = main_model(x)
        fv = feats["fv"]
        fs = skel_branch(s)
        fi = imu_branch(im)
        gs = gate_s(torch.cat([fv, fs], -1))
        gi = gate_i(torch.cat([fv, fi], -1))
        f = fv + gs * proj_s(fs) + gi * proj_i(fi)
        return head(f)

    def valid():
        for m in (main_model, skel_branch, imu_branch, gate_s, gate_i, proj_s, proj_i):
            m.eval()
        c = t = 0
        with torch.no_grad():
            for x, y, _, keys in va_loader:
                x = x.to(device)
                lg = fused(x, get_sk(keys), get_im(keys))
                c += (lg.argmax(-1) == y.to(device)).sum().item()
                t += len(y)
        return c / max(t, 1)

    best, best_sd = 0.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        for m in (main_model, skel_branch, imu_branch, gate_s, gate_i, proj_s, proj_i):
            m.train()
        run = n = 0
        for x, y, _, keys in tr_loader:
            x, y = x.to(device), y.to(device)
            lg = fused(x, get_sk(keys), get_im(keys))
            loss = crit(lg, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * y.numel(); n += y.numel()
        sched.step()
        va = valid()
        if va > best:
            best = va
            best_sd = {
                name: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
                for name, mod in [("main", main_model), ("skel", skel_branch),
                                  ("imu", imu_branch), ("gs", gate_s), ("gi", gate_i),
                                  ("ps", proj_s), ("pi", proj_i), ("head", head)]
            }
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run/max(n,1):.4f} "
              f"val={va:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"state": best_sd, "best_acc": best}, outdir / f"gate_bi_fold{args.fold}.pth")
    print(f"== fold{args.fold} best={best:.4f} (对照 main16f=0.6695) ===", flush=True)


if __name__ == "__main__":
    main()
