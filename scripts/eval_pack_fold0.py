#!/usr/bin/env python3
"""int4 vs int5 折级等价性验证 + Int4 可提交 A 形态集成判定。

对 fold0 val 推理多个 pack(int5 与 int4),算单模型 acc 与 2main+1th prob-avg。
产出: outputs/oof/packint_fold0.pkl = { "<pack名>": { "<key>": logits[40] } }
判定: int4 单模型 acc 与 int5 差距 ≤0.5pt 且 2m1th int4 集成 acc 不显著 < int5 → int4 可提交。
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

from src.dataset import DepthIRVideoDataset, ThermalVideoDataset, ThermalClipIndex, build_train_index
from src.split import split_by_subject
from src.model import build_model
from src.quantize import load_quantized


def load_sd(path, device):
    ck = torch.load(path, map_location=device)
    return load_quantized(path, device) if "q_sd" in ck else ck


def infer_pack(model, loader, device):
    model.eval()
    out = np.zeros((0, 40), dtype=np.float32)
    with torch.no_grad():
        for x, _, _ in loader:
            b = x.shape[0]
            out = np.concatenate([out, model(x.to(device)).cpu().numpy()], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--packs", nargs="+", default=[
        "main_s42_fold0_int5", "main_s42_fold0_int4",
        "main_s777_fold0_int5", "main_s777_fold0_int4",
        "thermal_s42_fold0_int5", "thermal_s42_fold0_int4",
        "thermal_s777_fold0_int5", "thermal_s777_fold0_int4",
    ])
    ap.add_argument("--pack_dir", default="outputs/pack")
    ap.add_argument("--out", default="outputs/oof/packint_fold0.pkl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=3)
    tr_idx, va_idx = folds[args.fold]
    va_clips = [clips[i] for i in va_idx]
    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
    keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in va_clips]

    results = {}
    dsm = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, main_crop)
    dst = ThermalVideoDataset(
        [ThermalClipIndex(c.action_id, c.subject, c.sample,
                          root / "Thermal" / c.depth_dir.parent.parent.name / c.subject / c.sample)
         for c in va_clips], args.num_frames, args.size, False, th_crop)
    loadm = DataLoader(dsm, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    loadt = DataLoader(dst, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    m_model = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=args.num_frames).to(device).eval()
    t_model = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=args.num_frames).to(device).eval()

    for name in args.packs:
        p = Path(args.pack_dir) / f"{name}.pth"
        if not p.exists():
            print(f"[skip] {name} 缺", flush=True)
            continue
        is_th = name.startswith("thermal")
        model, dl = (t_model, loadt) if is_th else (m_model, loadm)
        sd = load_sd(p, device)
        try:
            model.load_state_dict(sd)
        except Exception as e:
            print(f"[load-err] {name}: {e}", flush=True)
            continue
        t0 = time.time()
        lg = infer_pack(model, dl, device)
        acc = float((lg.argmax(-1) == np.array([int(k.split('/')[0]) for k in keys])).mean())
        print(f"fold{args.fold} {name}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"fold{args.fold} {name}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results[name] = dict(zip(keys, lg.astype(np.float32)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(results, fh, protocol=4)
    print(f"[save] {args.out}  n_packs={len(results)}", flush=True)


if __name__ == "__main__":
    main()
