#!/usr/bin/env python3
"""骨架手工运动特征在 main+th 已错的孪生对样本上的条件纠错 fold 验证。
若 GBDT 纠对率显著 >50%, 骨架孪生对 overlay 值得进链(只在主链犹豫样本借骨架)。
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_kp(k):
    fs = sorted(k.pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    kps = []
    for f in fs:
        try:
            d = json.loads(f.read_text("utf-8"))
            fr = d if isinstance(d, dict) else d[0]
            kps.append(np.asarray(fr["keypoints"], np.float32).reshape(17, 3))
        except Exception:
            pass
    return np.stack(kps, 0) if len(kps) >= 3 else None


def feats(kp):
    """含 Global 骨盆轨迹 + 局部身体动/静比 的手工运动特征(10 维)。"""
    p = kp[:, 0]
    dp = np.diff(p, axis=0)
    rel = kp - kp[:, 0:1, :]
    hand = rel[:, [4, 7]]
    foot = rel[:, [10, 13]]
    torso = rel[:, [1, 2, 8, 11, 14]]
    hv = np.abs(np.diff(hand, axis=0)).mean()
    fv = np.abs(np.diff(foot, axis=0)).mean()
    tv = np.abs(np.diff(torso, axis=0)).mean()
    return np.array([
        np.abs(dp).mean(), np.abs(dp).max(), np.std(kp[:, 0, 2]),
        np.abs(np.diff(rel, axis=0)).mean(),
        hv, fv, tv, hv / max(tv, 1e-6), fv / max(tv, 1e-6), np.abs(p).mean(),
    ])


def main():
    TWINS = [(13, 12), (17, 26)]
    OOF = Path("outputs/oof")
    root = Path("data/Training/HAR")
    clips = build_skeleton_index(root)
    twin_set = {a for t in TWINS for a in t}

    per_t = {}
    tot = [0, 0]  # n_mainerr, gbdt_correct
    for fold in [0, 1, 2]:
        km = pickle.load(open(OOF / "main_oof.pkl", "rb"))[fold]
        kt = pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[fold]
        folds = split_by_subject(clips, 3)
        tr_idx, va_idx = folds[fold]
        tr_c = [clips[i] for i in tr_idx]
        va_c = [clips[i] for i in va_idx]
        # GBDT 训练 (train folds 的孪生样本)
        X, L = [], []
        for c in tr_c:
            if c.action_id not in twin_set:
                continue
            k = load_kp(c)
            if k is not None:
                X.append(feats(k))
                L.append(c.action_id)
        X = np.array(X); L = np.array(L)
        clfs = {}
        for a, b in TWINS:
            m = [i for i, l in enumerate(L) if l in (a, b)]
            clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
            clf.fit(X[m], np.array([1 if L[i] == a else 0 for i in m]))
            clfs[(a, b)] = clf
        # val 孪生样本: 只看 main+th 已判错的
        for c in va_c:
            if c.action_id not in twin_set:
                continue
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in km or key not in kt:
                continue
            p = (sf(km[key]) + sf(kt[key])).argmax()
            if p == c.action_id:
                continue  # main 对的不动
            k = load_kp(c)
            if k is None:
                continue
            for a, b in TWINS:
                if c.action_id in (a, b):
                    g = clfs[(a, b)].predict([feats(k)])[0]
                    pred = a if g == 1 else b
                    per_t.setdefault((a, b), [0, 0])
                    per_t[(a, b)][0] += 1
                    per_t[(a, b)][1] += int(pred == c.action_id)
                    tot[0] += 1
                    tot[1] += int(pred == c.action_id)

    for t in per_t:
        n, c = per_t[t]
        print(f"孪生对{t}: main已错 n={n} → 骨架GBDT纠对率 {100*c/n:.1f}%")
    if tot[0]:
        print(f"合计: main已错 n={tot[0]}, GBDT纠对率={100*tot[1]/tot[0]:.1f}%  (挽回≈{tot[1]}样本)")


if __name__ == "__main__":
    main()
