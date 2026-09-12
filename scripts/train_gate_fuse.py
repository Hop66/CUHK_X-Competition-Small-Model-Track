#!/usr/bin/env python3
"""骨架特征门控融合(main R2+1D + 骨架 BiGRU 分支 + gate) —— 单模型内、特征层融合。
不是概率层叠加(已证 -0.5): 视觉 f_v(512) 与 骨架 f_s(GRU) 经门控 g=σ(W[·]) 逐通道调制融合。
fold0 main nf16(+骨架特征门控) vs 0.6695(单 main)。推理需骨架输入(官方提供/轻 pose)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import build_train_index, DepthIRVideoDataset
from src.model import build_model
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject


class SkeletonBranch(nn.Module):
    """[T,17,6](norm+vel) → BiGRU+attn → 256。"""
    def __init__(self, hidden=128, T=16):
        super().__init__()
        self.gru = nn.GRU(17 * 6, hidden, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.proj = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU())

    def forward(self, s):          # [B,T,17,6]
        B, T, J, C = s.shape
        s = s.reshape(B, T, J * C)
        g, _ = self.gru(s)          # [B,T,256]
        w = torch.softmax(self.attn(g), dim=1)
        z = (g * w).sum(dim=1)      # [B,256]
        return self.proj(z)         # [B,128]


GATE = None


def skel_seq(pred_dir, T=16):
    fs = sorted(Path(pred_dir).glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    if len(fs) < 2:
        return None
    idx = np.linspace(0, len(fs) - 1, T).round().astype(int)
    kps = []
    for i in idx:
        try:
            o = json.loads(fs[i].read_text("utf-8"))
            fr = o if isinstance(o, dict) else o[0]
            kps.append(np.asarray(fr["keypoints"], np.float32).reshape(17, 3))
        except Exception:
            return None
    kp = np.stack(kps)                 # [T,17,3]
    rel = kp - kp[:, 0:1, :]
    vel = np.zeros_like(rel)
    vel[1:] = rel[1:] - rel[:-1]
    s = np.concatenate([rel, vel], -1)  # [T,17,6]
    sc = np.abs(s).max() + 1e-6
    return (s / sc).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--save_dir", default="outputs/main_gate_skel")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, 3)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    skel_map = {f"{c.action_id}/{c.subject}/{c.sample}": str(c.pred_dir)
                for c in build_skeleton_index(root)}
    print(f"fold{args.fold} tr={len(tr_clips)} va={len(va_clips)}", flush=True)

    outdir = Path(args.save_dir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)
    crop_cache = json.loads(Path("bbox_train.json").read_text(encoding="utf-8"))
    tr_ds = DepthIRVideoDataset(tr_clips, args.num_frames, 128,
                                True, crop_cache, aug_strength=2, return_key=True)
    va_ds = DepthIRVideoDataset(va_clips, args.num_frames, 128, False, crop_cache,
                                return_key=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    main_model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                             weights_path=args.weights or None).to(device)
    skel_branch = SkeletonBranch().to(device)
    # 门控融合: f_v(512)+f_s(128) → g(512); f = (1-g)*f_v + g*proj(f_s->512)
    gate = nn.Sequential(nn.Linear(512 + 128, 512), nn.Sigmoid()).to(device)
    proj_s = nn.Sequential(nn.Linear(128, 512), nn.ReLU()).to(device)
    head = nn.Linear(512, 40).to(device)
    (gate, proj_s, head, skel_branch, main_model).__class__  # noqa
    params = (list(main_model.parameters()) + list(skel_branch.parameters())
              + list(gate.parameters()) + list(proj_s.parameters()) + list(head.parameters()))
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    feats = {}
    def hook(mod, inp, o):
        feats["fv"] = o
    main_model.encoder.register_forward_hook(hook)
    skel_cache_ref = {}

    def get_skel(keys):
        out = []
        for k in keys:
            if k not in skel_cache_ref:
                s = skel_seq(skel_map.get(k, ""))
                skel_cache_ref[k] = (torch.from_numpy(s) if s is not None else
                                     torch.zeros(args.num_frames, 17, 6))
            out.append(skel_cache_ref[k])
        return torch.stack(out).to(device)

    def fused_logits(x, s):
        feats.clear()
        _ = main_model(x)             # 触发 hook → feats['fv']
        fv = feats["fv"]              # [B,512]
        fs = skel_branch(s)           # [B,128]
        g = gate(torch.cat([fv, fs], -1))
        f = (1 - g) * fv + g * proj_s(fs)
        return head(f), fs

    def valid():
        main_model.eval(); skel_branch.eval(); gate.eval(); proj_s.eval()
        c = t = 0
        with torch.no_grad():
            for x, y, _, keys in va_loader:
                x = x.to(device)
                s = get_skel(keys)
                lg, _ = fused_logits(x, s)
                c += (lg.argmax(-1) == y.to(device)).sum().item()
                t += len(y)
        return c / max(t, 1)

    best, best_sd = 0.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        main_model.train(); skel_branch.train(); gate.train(); proj_s.train()
        run = n = 0
        for x, y, _, keys in tr_loader:
            x, y = x.to(device), y.to(device)
            s = get_skel(keys)
            lg, _ = fused_logits(x, s)
            loss = crit(lg, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * y.numel(); n += y.numel()
        sched.step()
        va = valid()
        if va > best:
            best = va
            # ⚠️ P0修复(09-12 审计): 原版只快照 main_model, skel/gate/proj_s/head 用最后epoch
            #    → "best main + final adapter" 混搭，非一致 best checkpoint。
            #    现在五个组件在 best 时全部深拷贝。
            import copy as _copy
            best_sd = {
                "main": _copy.deepcopy(main_model.state_dict()),
                "skel": _copy.deepcopy(skel_branch.state_dict()),
                "gate": _copy.deepcopy(gate.state_dict()),
                "proj_s": _copy.deepcopy(proj_s.state_dict()),
                "head": _copy.deepcopy(head.state_dict()),
            }
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run/max(n,1):.4f} "
              f"val={va:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if best_sd is None:
        import copy as _copy
        best_sd = {
            "main": _copy.deepcopy(main_model.state_dict()),
            "skel": _copy.deepcopy(skel_branch.state_dict()),
            "gate": _copy.deepcopy(gate.state_dict()),
            "proj_s": _copy.deepcopy(proj_s.state_dict()),
            "head": _copy.deepcopy(head.state_dict()),
        }
    torch.save({"main": best_sd["main"], "best_acc": best,
                "skel": best_sd["skel"], "gate": best_sd["gate"],
                "proj_s": best_sd["proj_s"], "head": best_sd["head"]},
               outdir / f"gate_skel_fold{args.fold}.pth")
    print(f"== fold{args.fold} best={best:.4f} (对照 main16f=0.6695) ===", flush=True)


if __name__ == "__main__":
    main()
