#!/usr/bin/env python3
"""CUHK-X —— 评估手工运动特征 CSV 的分类能力（GBDT，规则允许非 DL）

判读：3 折（按 subject 分层）mean acc 显著 > 随机(2.5% for 40 类) 才算有信号；
若单独 acc ≥ 0.3 且与 main 误差互补 → 参与决策级融合（Stacking）。

用法:
    python scripts/eval_feats_gbdt.py --csv outputs/feats/bbox_feats_train.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def load_csv(path: Path):
    rows = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        feats = [k for k in rdr.fieldnames if k not in ("sample", "label")]
        for r in rdr:
            rows.append((r["sample"], int(r["label"]),
                         [float(r[k]) for k in feats]))
    return rows, feats


def subject_of(sample: str) -> str:
    return sample.split("/")[1] if "/" in sample else sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception:
        from sklearn.ensemble import RandomForestClassifier as GradientBoostingClassifier
        print("[eval] 无 GBDT，用 RandomForest 替代", flush=True)

    rows, feats = load_csv(Path(args.csv))
    if not rows:
        print("空 CSV"); return
    X = np.asarray([r[2] for r in rows], np.float64)
    y = np.asarray([r[1] for r in rows], np.int64)
    subjects = [subject_of(r[0]) for r in rows]
    uni = sorted(set(subjects))
    print(f"samples={len(rows)} 特征={len(feats)} subjects={len(uni)} "
          f"随机基线={1/len(set(y))*100:.1f}%", flush=True)

    # 按 subject 分层 3 折（严格跨被试，不泄漏）
    rng = np.random.default_rng(0)
    rng.shuffle(uni)
    folds = np.array_split(uni, args.folds)
    accs = []
    for fi in range(args.folds):
        va_sub = set(folds[fi])
        tr_mask = np.array([s not in va_sub for s in subjects])
        va_mask = ~tr_mask
        if tr_mask.sum() < 30 or va_mask.sum() < 10:
            print(f"fold{fi} 样本过少，跳过"); continue
        clf = GradientBoostingClassifier(n_estimators=args.n_estimators,
                                         learning_rate=0.05, max_depth=3)
        clf.fit(X[tr_mask], y[tr_mask])
        acc = (clf.predict(X[va_mask]) == y[va_mask]).mean()
        accs.append(acc)
        print(f"  fold{fi}: train={tr_mask.sum()} val={va_mask.sum()} acc={acc:.4f}", flush=True)
    if accs:
        print(f"\n== 特征单独 GBDT mean acc = {np.mean(accs):.4f} "
              f"(随机基线 {1/len(set(y))*100:.1f}%) ==")
        print("判读：显著 > 随机才有信号；≥0.3 且与 main 误差互补 → 参与决策级融合")


if __name__ == "__main__":
    main()
