#!/usr/bin/env python3
"""挖掘被埋没判决：DualFusion(特征级 end-to-end 融合, main4ch+thermal3ch) fold0 验证。

产出:
  outputs/oof/dualfusion_oof_fold0.pkl = { "<action>/<subj>/<sample>": logits[40] }
打印 fold0 val acc 及与 main/thermal 单流样本数对齐的集成判定线索。
对比: extract_oof_logits 产出的 main_oof.pkl / thermal_oof.pkl 的 fold0 部分。
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    ClipIndex, DepthIRVideoDataset, ThermalClipIndex, ThermalVideoDataset,
    build_train_index,
)
from src.split import split_by_subject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dual_fusion import DualFusion, DualDataset  # noqa: E402


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    keys, logits, labels = [], [], []
    for xm, xt, y in loader:
        xm, xt = xm.to(device), xt.to(device)
        lg = model(xm, xt).cpu().numpy().astype(np.float32)
        xs = xm.shape[0]
        logits.append(lg)
        labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    return np.concatenate(logits, 0), np.concatenate(labels, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--ckpt", default="outputs/dual_fusion/dualfusion_fold0.pth")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--out", default="outputs/oof/dualfusion_oof_fold0.pkl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    va_clips = [clips[i] for i in va_idx]
    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))

    va_ds = DualDataset(va_clips, args.num_frames, args.size, False, main_crop, th_crop)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    print(f"val clips={len(va_ds)}", flush=True)

    model = DualFusion(args.weights, mod_drop=0.0).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"ckpt best_acc(训练时记)={ck.get('best_acc', '?')}", flush=True)

    t0 = time.time()
    logits, labels = collect_logits(model, va_loader, device)
    acc = float((logits.argmax(-1) == labels).mean())
    print(f"== DualFusion fold{args.fold} val acc = {acc:.4f} ({time.time()-t0:.0f}s) "
          f"对比 main fold0 ~0.6609 / thermal 0.6339 / R(1m1th) fold0 需同clip集 ==", flush=True)

    keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in va_clips]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({args.fold: dict(zip(keys, logits))}, fh, protocol=4)
    print(f"[save] {args.out}  fold{args.fold} {len(keys)} clips", flush=True)


if __name__ == "__main__":
    main()
