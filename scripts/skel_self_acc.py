#!/usr/bin/env python3
"""骨架自身 acc 评估（Task 3，纯 CPU，不抢 GPU）:
全量 40 类 LOSO(GPB) 下，骨架手工 v3 特征(+IMU) 的 GBDT 单模态 acc。
对照: BiGRU 原始 [T,17,6] 单模态 fold0 ≈ 0.5318；train_main_aux 的 aux 基线 0.6695 是 main。
目的: 判断"骨架自身优化 acc"的最大手柄 —— 若 v3+IMU GBDT 显著 > 0.53，说明手工运动学特征是骨架
单模态的最优表示，值得给 gate 融合的 SkeletonBranch 用 v3 风格特征而非原始 rel+vel。

用法: python scripts/skel_self_acc.py --cpu --features v3_imu|v3|imu
"""
import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.dataset import build_train_index
from src.imu_dataset import build_imu_index
from src.skeleton_dataset import build_skeleton_index
from src.split import split_by_subject
from twin_v3 import feats_v3, imu_traj, load_kp

ROOT = Path("data/Training/HAR")


def cache_path(features):
    return Path("outputs") / f"skel_v3_full_cache_{features}.pkl"


def build_full_feats(features):
    sk = build_skeleton_index(ROOT)
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(ROOT, build_train_index(ROOT))}
    for c in sk:
        c.imu_dir = imu_map[f"{c.subject}/{c.sample}"].imu_dir
    CACHE = cache_path(features)
    if CACHE.exists():
        FE = pickle.load(open(CACHE, "rb"))
        use = {f"{c.action_id}/{c.subject}/{c.sample}" for c in sk}
        FE = {k: v for k, v in FE.items() if k in use}
    else:
        FE = {}
        for i, c in enumerate(sk):
            k = load_kp(c)
            if k is None:
                continue
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            parts = []
            if "v3" in features:
                parts.append(feats_v3(k))
            if "imu" in features:
                parts.append(imu_traj(c.imu_dir))
            FE[key] = np.concatenate(parts).astype(np.float32)
            if i % 500 == 0:
                print(f"  feat {i}/{len(sk)} ({key})", flush=True)
        pickle.dump(FE, open(CACHE, "wb"))
    return sk, FE


def run_fold(sk, FE, fold):
    folds = split_by_subject(sk, 3)
    tr_idx, va_idx = folds[fold]
    tr_c = [sk[i] for i in tr_idx]
    va_c = [sk[i] for i in va_idx]
    Xtr, ytr = [], []
    for c in tr_c:
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        if key in FE:
            Xtr.append(FE[key]); ytr.append(c.action_id)
    Xva, yva = [], []
    for c in va_c:
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        if key in FE:
            Xva.append(FE[key]); yva.append(c.action_id)
    Xtr = np.stack(Xtr); Xva = np.stack(Xva)
    ytr = np.array(ytr); yva = np.array(yva)
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    return accuracy_score(yva, pred), len(yva), Counter(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="v3_imu", choices=["v3_imu", "v3", "imu"])
    args = ap.parse_args()

    print(f"== 骨架自身 acc (CPU) features={args.features} | 对照 BiGRU fold0≈0.5318 ==", flush=True)
    t0 = time.time()
    sk, FE = build_full_feats(args.features)
    print(f"feats n={len(FE)} dim={len(next(iter(FE.values())))} ({time.time()-t0:.0f}s)", flush=True)
    accs = []
    for fold in range(3):
        a, n, cnt = run_fold(sk, FE, fold)
        accs.append(a)
        print(f"fold{fold} acc={a:.4f} n={n} pred类数={len(cnt)}", flush=True)
    print(f"== 骨架单模态 3折 mean={np.mean(accs):.4f} | 对照 BiGRU fold0≈0.5318 ==", flush=True)


if __name__ == "__main__":
    main()
