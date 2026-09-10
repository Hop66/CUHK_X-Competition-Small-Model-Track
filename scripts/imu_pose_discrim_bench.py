#!/usr/bin/env python3
"""CUHK-X —— IMU 姿态积分特征的难对判别基准（无泄漏：train fold 训 → 其他 fold 测）。

目的：验证「IMU 修复(5dev 30ch) + 姿态角积分」是否提升难对二分类判别力。
此前 raw IMU CNN 难对判别 = 0.623（各模态中仅次于 main 0.662）。

特征：imu_pose_integ(imu_dir) → [T,30]（5dev × [gyro3, 积分姿态角3]）
判别器：轻 1D-CNN（同 IMUCNN 结构）二分类 per-pair。

用法：
  python scripts/imu_pose_discrim_bench.py --pairs "6-37,6-7,11-14,7-19,13-12,22-21"
输出：每 pair 跨折平均 acc（对 main baseline 0.66 有增益才算真判别）。
"""
import argparse
import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import build_train_index
from src.split import split_by_subject
from scripts.imu_pose_integ import imu_pose_integ
from src.imu_dataset import build_imu_index


class TwinCNN(nn.Module):
    def __init__(self, in_ch=30):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(48, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Linear(96, 2)

    def forward(self, x):
        x = self.enc(x.permute(0, 2, 1))      # [B,30,T] -> [B,96,T/4]
        return self.head(x.mean(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="6-37,6-7,11-14,7-19,13-12,22-21,8-10,18-17")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # 缓存路径（按 feat 名分流，防污染）
    CACHE = Path("outputs/imu_pose_feats_bench.pkl")
    if CACHE.exists():
        print(f"load cache {CACHE}")
        data = pickle.load(open(CACHE, "rb"))
        keys, feats, labels = data["keys"], data["feats"], data["labels"]
    else:
        root = Path("data/Training/HAR")
        main_clips = build_train_index(root)
        clips = build_imu_index(root, main_clips)
        keys, feats, labels = [], [], []
        for i, c in enumerate(clips):
            if c.imu_dir is None or not c.imu_dir.exists():
                feats.append(np.zeros((args.T, 30), np.float32))
            else:
                try:
                    feats.append(imu_pose_integ(c.imu_dir, args.T))
                except Exception:
                    feats.append(np.zeros((args.T, 30), np.float32))
            keys.append(f"{c.action_id}/{c.subject}/{c.sample}")
            labels.append(c.action_id)
        feats = np.stack(feats).astype(np.float32)
        print(f"feats {feats.shape} cache -> {CACHE}")
        pickle.dump({"keys": keys, "feats": feats, "labels": labels}, open(CACHE, "wb"))

    # folds（按 subject）
    n = len(keys)
    subj = [k.split("/")[1] for k in keys]
    us = sorted(set(subj))
    np.random.seed(0)
    perm = np.random.permutation(us)
    folds = [set(perm[i::3].tolist()) for i in range(3)]  # 3 折按 subject

    device = torch.device(args.device)
    PAIRS = [tuple(map(int, p.split("-"))) for p in args.pairs.split(",")]
    print(f"\n=== 姿态积分 IMU 难对判别（无泄漏跨折） ===")
    for a, b in PAIRS:
        idx_a = [i for i in range(n) if labels[i] == a]
        idx_b = [i for i in range(n) if labels[i] == b]
        if len(idx_a) < 5 or len(idx_b) < 5:
            print(f"  ({a},{b}) 样本不足 n_a={len(idx_a)} n_b={len(idx_b)} skip")
            continue
        accs = []
        xa = feats[idx_a]; ya = np.ones(len(idx_a), np.int64)
        xb = feats[idx_b]; yb = np.zeros(len(idx_b), np.int64)
        for f in range(3):
            tra = [i for i, gi in enumerate(idx_a) if subj[gi] not in folds[f]]
            trb = [i for i, gi in enumerate(idx_b) if subj[gi] not in folds[f]]
            vaa = [i for i, gi in enumerate(idx_a) if subj[gi] in folds[f]]
            vab = [i for i, gi in enumerate(idx_b) if subj[gi] in folds[f]]
            if len(tra) == 0 or len(trb) == 0 or len(vaa) == 0 or len(vab) == 0:
                continue
            Xtr = np.concatenate([xa[tra], xb[trb]])
            Ytr = np.concatenate([ya[tra], yb[trb]])
            Xva = np.concatenate([xa[vaa], xb[vab]])
            Yva = np.concatenate([ya[vaa], yb[vab]])
            model = TwinCNN(30).to(device)
            opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
            lossf = nn.CrossEntropyLoss()
            tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
            tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
            for _ in range(args.epochs):
                model.train()
                for xb, yb_ in tr_ld:
                    xb, yb_ = xb.to(device), yb_.to(device)
                    opt.zero_grad()
                    lossf(model(xb), yb_).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(Xva).to(device)
                acc = (model(xv).argmax(-1).cpu().numpy() == Yva).mean()
            accs.append(acc)
        if accs:
            m = np.mean(accs)
            mark = "  <-- 主要" if m > 0.68 else ""
            print(f"  ({a},{b}) 跨折acc={m:.3f} (n={len(idx_a)}+{len(idx_b)}) {mark}")
        else:
            print(f"  ({a},{b}) 无有效折")


if __name__ == "__main__":
    main()
