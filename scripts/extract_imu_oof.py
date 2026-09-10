#!/usr/bin/env python3
"""IMU fold{0,1,2} OOF logits 提取 —— 补「低权 α 插值」验证(骨架同为法, 之前只按单模态0.33判弱漏测)。

产物: outputs/oof/imu_fold{i}.pkl = { "<action>/<subj>/<sample>": logits[40] } (与 main/thermal OOF 同 key)
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import build_train_index
from src.imu_dataset import IMUDataset, build_imu_index
from src.split import split_by_subject
from train_imu import IMUCNN


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ckpt_base", default="outputs/imu/imu_fold{i}.pth")
    ap.add_argument("--out_base", default="outputs/oof/imu_fold{i}.pkl")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--fold", type=int, default=-1, help="-1=全部 0/1/2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    clips = build_imu_index(root, main_clips)
    folds = split_by_subject(clips, n_folds=3)
    targets = [args.fold] if args.fold >= 0 else [0, 1, 2]

    model = IMUCNN().to(device).eval()
    for fi in targets:
        _, va_idx = folds[fi]
        va_clips = [clips[i] for i in va_idx]
        ds = IMUDataset(va_clips, args.T, False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        ckpt = Path(args.ckpt_base.replace("{i}", str(fi))).expanduser()
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"])
        print(f"loaded {ckpt} (best_acc={sd.get('best_acc','?')})", flush=True)
        out = {}
        s = 0
        for x, y, _ in loader:
            x = x.to(device)
            lg = model(x).cpu().numpy().astype(np.float32)
            for j in range(len(x)):
                c = va_clips[s + j]
                out[f"{c.action_id}/{c.subject}/{c.sample}"] = lg[j]
            s += len(x)
        acc = float((np.argmax(np.stack(list(out.values())), -1) ==
                     np.array([int(k.split('/')[0]) for k in out])).mean())
        print(f"  fold{fi}: n={len(out)} val_acc={acc:.4f}", flush=True)
        outf = Path(args.out_base.replace("{i}", str(fi))).expanduser()
        outf.parent.mkdir(parents=True, exist_ok=True)
        with open(outf, "wb") as fh:
            pickle.dump(out, fh)
        print(f"  saved {outf}", flush=True)


if __name__ == "__main__":
    main()
