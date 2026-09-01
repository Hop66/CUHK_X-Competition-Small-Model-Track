#!/usr/bin/env python3
"""CUHK-X —— IMU 特征工程 + RF 摸底（诊断 IMU 数据是否有规律）。

思路（用户提供）：手工统计特征（均值/方差/峰值/频谱能量等）→ 随机森林，
小数据下稳健不过拟合。回答"IMU 数据到底有没有信号"：
  acc > 0.5 → IMU 有规律 → 值得深入（简单 CNN / 加入融合）
  acc ~0.3 → IMU 噪声大/弱 → 不投入

特征：5 设备 × 6 通道(acc3+gyro3) = 30 通道 × 8 统计量 = 240 维
用法:
    python scripts/imu_feat_rf.py --folds 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset import build_train_index
from src.imu_dataset import load_imu_sequence, time_align, build_imu_index
from src.split import split_by_subject


def extract_feats(feat: np.ndarray) -> np.ndarray:
    """[T,30] → [240] 统计特征（每通道 8 统计量）。"""
    stats = []
    for c in range(feat.shape[1]):
        x = feat[:, c]
        stats += [x.mean(), x.std(), x.min(), x.max(),
                  x.ptp(), float(np.sum(x ** 2)),
                  float(np.mean(np.abs(np.diff(x)) > 1e-6)),  # 过零/变化率
                  float(np.median(x))]
    return np.array(stats, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=None)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier

    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    clips = build_imu_index(root, main_clips)
    print(f"IMU clips={len(clips)} classes={len(set(c.action_id for c in clips))}", flush=True)

    # 提取全部特征（一次性，内存 OK）
    t0 = time.time()
    X, y, subj = [], [], []
    for c in clips:
        dev = load_imu_sequence(c.imu_dir)
        feat = time_align(dev, args.T)          # [T,30]
        X.append(extract_feats(feat))
        y.append(c.action_id)
        subj.append(c.subject)
    X = np.array(X); y = np.array(y)
    print(f"特征矩阵 {X.shape}（30ch×8=240 维）耗时 {time.time()-t0:.1f}s", flush=True)

    folds = split_by_subject(clips, n_folds=args.folds)
    accs = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr_mask = np.zeros(len(clips), bool); tr_mask[tr_idx] = True
        va_mask = np.zeros(len(clips), bool); va_mask[va_idx] = True
        rf = RandomForestClassifier(n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    n_jobs=-1, random_state=42)
        rf.fit(X[tr_mask], y[tr_mask])
        acc = (rf.predict(X[va_mask]) == y[va_mask]).mean()
        accs.append(acc)
        print(f"[fold{fi}] RF acc = {acc:.4f}", flush=True)

    mean_acc = float(np.mean(accs))
    print(f"\n==== IMU RF 摸底: mean acc = {mean_acc:.4f} ====", flush=True)
    print("判读: >0.5 → IMU 有规律，值得深入; 0.3-0.5 → 弱但有信息; <0.3 → 噪声大/无信号", flush=True)


if __name__ == "__main__":
    main()
