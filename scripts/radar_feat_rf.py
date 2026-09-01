#!/usr/bin/env python3
"""CUHK-X —— Radar 特征工程 + RF 摸底（诊断 Radar 数据是否有规律）。

Radar：radar_output_*.csv，8 列稀疏点云 (timestamp,frame,DetObj#,x,y,z,v,snr,noise)
每 clip 聚合点云统计特征 → 随机森林（小数据稳健，不过拟合）。
回答"Radar 到底有没有信号"：
  acc > 0.4 → Radar 有规律，值得深入（加入融合）
  acc ~0.2 → 点云太稀疏/噪声大 → 关闭

用法:
    python scripts/radar_feat_rf.py --folds 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.dataset import build_train_index
from src.split import split_by_subject


def clip_feats(radar_dir: Path) -> np.ndarray:
    """聚合一个 clip 的 Radar 点云为统计特征向量。"""
    csvs = sorted(radar_dir.glob("radar_output_*.csv"))
    xs, ys, zs, vs, snrs = [], [], [], [], []
    n_frames, n_points = 0, 0
    for f in csvs:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        n_frames += 1
        n_points += len(df)
        if "x" in df.columns:
            xs.append(df["x"].to_numpy()); ys.append(df["y"].to_numpy())
            zs.append(df["z"].to_numpy()); vs.append(df["v"].to_numpy())
        if "snr" in df.columns:
            snrs.append(df["snr"].to_numpy())
    if not xs:
        return np.zeros(20, np.float32)
    x = np.concatenate(xs); y = np.concatenate(ys)
    z = np.concatenate(zs); v = np.concatenate(vs)
    s = np.concatenate(snrs) if snrs else np.zeros_like(x)
    feats = [
        n_frames, n_points,                      # 规模
        float(len(csvs)),                        # 文件数
        x.mean(), x.std(), float(x.ptp()),       # x 分布
        y.mean(), y.std(), float(y.ptp()),       # y 分布
        z.mean(), z.std(), float(z.ptp()),       # z 分布
        v.mean(), v.std(), float(np.sum(v ** 2)),  # 速度/能量
        s.mean(), float(s.max()),                 # SNR
        float(np.mean(np.abs(v) > 0.1)),          # 运动点占比
    ]
    return np.array(feats[:20], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n_estimators", type=int, default=300)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier

    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)  # 复用 main 的 clip 索引，Radar 目录同结构
    print(f"clips={len(clips)}", flush=True)

    t0 = time.time()
    X, y = [], []
    n_empty = 0
    for c in clips:
        radar_dir = root / "Radar" / c.depth_dir.parent.parent.name / c.subject / c.sample
        f = clip_feats(radar_dir)
        if f.sum() == 0:
            n_empty += 1
        X.append(f); y.append(c.action_id)
    X = np.array(X); y = np.array(y)
    print(f"特征矩阵 {X.shape} 空 clip {n_empty} 耗时 {time.time()-t0:.1f}s", flush=True)

    folds = split_by_subject(clips, n_folds=args.folds)
    accs = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr_mask = np.zeros(len(clips), bool); tr_mask[tr_idx] = True
        va_mask = np.zeros(len(clips), bool); va_mask[va_idx] = True
        rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=42)
        rf.fit(X[tr_mask], y[tr_mask])
        acc = (rf.predict(X[va_mask]) == y[va_mask]).mean()
        accs.append(acc)
        print(f"[fold{fi}] Radar RF acc = {acc:.4f}", flush=True)

    print(f"\n==== Radar RF 摸底: mean acc = {np.mean(accs):.4f} ====", flush=True)
    print("判读: >0.4 → Radar 有规律值得深入; 0.2-0.4 → 弱; <0.2 → 点云噪声大关闭", flush=True)


if __name__ == "__main__":
    main()
