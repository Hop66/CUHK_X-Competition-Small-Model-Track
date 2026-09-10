#!/usr/bin/env python3
"""IMU test logits 提取 —— 3 折 IMUCNN(imu_fold{0,1,2}.pth) 各推一遍 Testing 取平均。
产物: outputs/oof/imu_test_logits.npy, 形状 (N_smtest, 40), 行序 = sorted(SM_test_* dirs)。
用法(在 ensemble_inference 里以 --imu_logits 载入, --w_imu 定权重, --prob_avg 融合)。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.imu_dataset import IMUDataset, IMUClipIndex
from train_imu import IMUCNN


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root",
                    default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--ckpts", nargs="+", default=[
        "outputs/imu/imu_fold0.pth", "outputs/imu/imu_fold1.pth",
        "outputs/imu/imu_fold2.pth"])
    ap.add_argument("--model", default="cnn", choices=["cnn", "gru"],
                    help="cnn=IMUCNN(3conv+GAP), gru=IMUGRUNet(Conv1DBlock+BiGRU+attn)")
    ap.add_argument("--out", default="outputs/oof/imu_test_logits.npy")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "cnn":
        from train_imu import IMUCNN as ModelCls
    else:
        from train_imu_gru import IMUGRUNet as ModelCls
    root = Path(args.test_root).expanduser()
    clip_ids = [d.name for d in sorted(root.iterdir())
                if d.is_dir() and d.name.startswith("SM_test_")]
    clips = [IMUClipIndex(-1, cid, cid, root / cid / "IMU") for cid in clip_ids]
    ds = IMUDataset(clips, args.T, False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"test clips={len(clip_ids)} device={device}", flush=True)

    acc = np.zeros((len(clip_ids), 40), np.float32)
    t0 = time.time()
    for ck in args.ckpts:
        m = ModelCls().to(device).eval()
        sd = torch.load(Path(ck).expanduser(), map_location=device)
        m.load_state_dict(sd["model"])
        lg = []
        for x, _, _ in loader:
            lg.append(m(x.to(device)).cpu().numpy())
        lg = np.concatenate(lg, 0)
        best = sd.get("best_acc", "?")
        print(f"{Path(ck).name}: best_acc={best} ({time.time()-t0:.0f}s)", flush=True)
        acc += lg / len(args.ckpts)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, acc)
    print(f"saved {out} {acc.shape} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
