#!/usr/bin/env python3
"""test overlay: 主链(main+th soft) top-2 落在 (17,26)/(13,12) 时, 用骨架v2+IMU GBDT 在该对换头。
产出 outputs/sub_chain_skeloverlay.csv (与锚同格式)。LB 一次实测。
"""
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.imu_dataset import IMUClipIndex, IMUDataset, build_imu_index, load_imu_sequence, time_align
from src.skeleton_dataset import MotionBertSkeletonDataset, SkeletonClipIndex, build_skeleton_index, frame_num_of
from src.split import split_by_subject

TWINS = [(17, 26), (13, 12)]


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_kp(pred_dir):
    fs = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    kps = []
    for f in fs:
        try:
            d = json.loads(f.read_text("utf-8"))
            fr = d if isinstance(d, dict) else d[0]
            kps.append(np.asarray(fr["keypoints"], np.float32).reshape(17, 3))
        except Exception:
            pass
    return np.stack(kps, 0) if len(kps) >= 8 else None


def feats_v2(kp):
    p = kp[:, 0]
    dp = np.diff(p, axis=0)
    rel = kp - kp[:, 0:1, :]
    rel_d = rel - rel[0:1]
    lb = rel_d[:, [4, 7]] - rel_d[:, [3, 6]]
    dlb = np.diff(lb, axis=0)
    wr = rel_d[:, [4, 7]]; dwr = np.diff(wr, axis=0)
    ft = rel_d[:, [10, 13]]; dft = np.diff(ft, axis=0)
    ts = rel_d[:, [2, 5, 8, 11, 14]]; dts = np.diff(ts, axis=0)
    ax = kp[:, [8]] - kp[:, [0]]
    tilt = np.abs(np.diff(ax, axis=0)).mean()
    st = lambda d: [np.abs(d).mean(), np.abs(d).max(), np.abs(d).std()]
    gv = st(dp); hv = st(dwr); fv = st(dft); tv = st(dts); lv = st(dlb)
    return np.array(gv + hv + fv + tv + lv +
                    [hv[0] / max(tv[0], 1e-6), fv[0] / max(tv[0], 1e-6),
                     lv[0] / max(tv[0], 1e-6), tilt, np.std(kp[:, 0, 2])])


def imu_feat_dir(imu_dir):
    dev = load_imu_sequence(imu_dir)
    x = time_align(dev, T=64)
    if x is None or not np.any(x):
        return np.zeros(12)
    f = []
    for k in range(0, 30, 6):
        for b in (k, k + 3):
            a = x[:, b:b + 3]
            f += [np.abs(a).mean(), np.abs(np.diff(a, axis=0)).mean(), np.abs(a).std()]
    return np.array(f[:12], np.float32)


def main():
    root = Path("data/Training/HAR")
    troot = Path("data/Testing/data/small_model_track_test")

    # ---- 训练 GBDT (全量 train) ----
    sk = build_skeleton_index(root)
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(root, build_train_index(root))}
    X, L = [], []
    for c in sk:
        if c.action_id not in {a for t in TWINS for a in t}:
            continue
        k = load_kp(c.pred_dir)
        if k is None or f"{c.subject}/{c.sample}" not in imu_map:
            continue
        X.append(np.concatenate([feats_v2(k), imu_feat_dir(imu_map[f"{c.subject}/{c.sample}"].imu_dir)]))
        L.append(c.action_id)
    X = np.array(X); L = np.array(L)
    clfs = {}
    for a, b in TWINS:
        m = [i for i, l in enumerate(L) if l in (a, b)]
        clf = GradientBoostingClassifier(n_estimators=110, max_depth=3, random_state=0)
        clf.fit(X[m], (L[m] == a).astype(int))
        clfs[(a, b)] = clf
        print(f"GBDT ({a},{b}): n_train={len(m)}", flush=True)

    # ---- test 主链 soft (0.75锚 teacher probs) ----
    raw = pickle.load(open("outputs/test_teacher_avg_probs.pkl", "rb"))
    teacher = {re.sub(r"^test:", "", k): np.asarray(v, np.float32) for k, v in raw.items()}
    clip_ids = sorted(teacher.keys())
    P = {c: teacher[c] for c in clip_ids}  # 主链 soft prob

    # ---- test 特征 ----
    import collections
    flips = []
    rows = []
    for cid in clip_ids:
        p = P[cid]
        main_top = int(p.argmax())
        top2 = set(np.argsort(-p)[:2].tolist())
        pred = main_top
        k = load_kp(troot / cid / "Skeleton" / "predictions"
                    if (troot / cid / "Skeleton" / "predictions").is_dir()
                    else troot / cid / "Skeleton")
        if k is not None:
            xf = np.concatenate([feats_v2(k), imu_feat_dir(troot / cid / "IMU")])
            for a, b in TWINS:
                if a in top2 and b in top2:
                    g = clfs[(a, b)].predict([xf])[0]
                    pred = a if g == 1 else b
                    if pred != main_top:
                        flips.append((cid, main_top, pred))
        rows.append((f"small_model_track_test/{cid}/", int(pred)))
    df = pd.DataFrame(rows, columns=["path", "prediction"])
    df.to_csv("outputs/sub_chain_skeloverlay.csv", index=False)
    print(f"overlay flips={len(flips)}: {flips[:12]}")

    # 与锚对比
    a = pd.read_csv("outputs/sub_chain_nf32_flip.csv")
    f = lambda x: re.search(r"(SM_test_\d+)", str(x)).group(1)
    amap = dict(zip(a["path"].astype(str).map(f), a["prediction"]))
    dmap = dict(zip(df["path"].astype(str).map(f), df["prediction"]))
    nflip = sum(1 for k in amap if amap[k] != dmap[k])
    print(f"vs 锚 flips={nflip}/405")


if __name__ == "__main__":
    main()
