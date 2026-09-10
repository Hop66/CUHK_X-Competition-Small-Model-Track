#!/usr/bin/env python3
"""new thermal (nf32 R2+1D, GRU+Attn) fold0 复核 + OOF logits 收集。

产出:
  outputs/oof/th_nf32_oof_fold0.pkl / th_gru_oof_fold0.pkl
  { "<action>/<subj>/<sample>": logits[40] }（与 main_oof/thermal_oof 同 key）
并打印每模型 fold0 acc（复核推理链）。
用法:
  python scripts/collect_th_oof.py --th_nf32 outputs/th_nf32/r2plus1d34_thermal_fold0.pth \
      --th_gru outputs/th_gru/th_gru_fold0.pth --fold 0
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

from src.dataset import ThermalClipIndex, ThermalVideoDataset, build_train_index
from src.model import build_model
from src.split import split_by_subject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_thermal_gru_attn import ThermalGRUNet  # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    out = np.zeros((len(loader.dataset), 40), dtype=np.float32)
    s = 0
    for x, _, _ in loader:
        b = x.shape[0]
        out[s:s + b] = torch.softmax(model(x.to(device)), -1).cpu().numpy()
        s += b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--th_nf32", default="outputs/th_nf32/r2plus1d34_thermal_fold0.pth")
    ap.add_argument("--th_gru", default="outputs/th_gru/th_gru_fold0.pth")
    ap.add_argument("--nf32_frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs/oof")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=3)
    _, va_idx = folds[args.fold]
    va_clips = [clips[i] for i in va_idx]
    th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                 root / "Thermal" / c.depth_dir.parent.parent.name /
                                 c.subject / c.sample) for c in va_clips]
    crop = json.loads(Path("bbox_thermal_train.json").read_text(encoding="utf-8"))
    keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in va_clips]
    labels = np.array([c.action_id for c in va_clips])

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def run(name, model, nf):
        ds = ThermalVideoDataset(th_clips, nf, args.size, False, crop)
        loader = DataLoader(ds, batch_size=16, shuffle=False,
                            num_workers=args.workers, pin_memory=True, timeout=300)
        t0 = time.time()
        lg = collect(model, loader, device)
        acc = float((lg.argmax(-1) == labels).mean())
        print(f"[{name}] fold{args.fold} acc={acc:.4f} nf={nf} ({time.time()-t0:.0f}s)", flush=True)
        with open(Path(args.out_dir) / f"{name}_oof_fold{args.fold}.pkl", "wb") as fh:
            pickle.dump({args.fold: dict(zip(keys, lg.astype(np.float32)))}, fh, protocol=4)
        print(f"[save] {name}_oof_fold{args.fold}.pkl", flush=True)

    # th_nf32: R2+1D 3ch（train_step1 裸 state_dict）
    if args.th_nf32 and Path(args.th_nf32).exists():
        m = build_model("r2plus1d34", num_classes=40, in_channels=3,
                        n_segment=args.nf32_frames).to(device).eval()
        sd = torch.load(args.th_nf32, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        m.load_state_dict(sd)
        run("th_nf32", m, args.nf32_frames)

    # th_gru
    if args.th_gru and Path(args.th_gru).exists():
        m = ThermalGRUNet().to(device).eval()
        ck = torch.load(args.th_gru, map_location=device)
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"[th_gru] ckpt best_acc(训练)={ck.get('best_acc', '?')}", flush=True)
        run("th_gru", m, 16)


if __name__ == "__main__":
    main()
