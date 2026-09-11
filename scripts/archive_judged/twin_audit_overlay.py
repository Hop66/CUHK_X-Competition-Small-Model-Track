#!/usr/bin/env python3
"""升级版 audit + 扩大 overlay:
 1) 对全部 9 孪生对: 报 main+th 整体 vs 骨架v2+IMU GBDT 整体(3折聚合) → 选出 main 弱、弱模态强的对(weak-twin set)
 2) 端到端 overlay: 对 weak-set 对, 凡"主链对该 clip 属于这对"(top-1 或 top-2 ∈对)即换头 → 3折全局 Δ
各处特征一次性缓存(骨架kp+imu), 加快。
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.imu_dataset import build_imu_index, load_imu_sequence, time_align
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject

PAIRS = [(13, 12), (22, 21), (8, 9), (8, 10), (37, 6), (18, 17), (7, 6), (36, 32), (26, 24)]
FEAT_CACHE = Path("/tmp/twin_feat_cache.pkl")


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
    sk = build_skeleton_index(root)
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(root, build_train_index(root))}
    sk = [c for c in sk if c.action_id in {a for t in PAIRS for a in t}
          and f"{c.subject}/{c.sample}" in imu_map]
    for c in sk:
        c.imu_dir = imu_map[f"{c.subject}/{c.sample}"].imu_dir

    # 一次性特征缓存
    feats = {}
    for i, c in enumerate(sk):
        k = load_kp(c)
        if k is None:
            continue
        feats[f"{c.action_id}/{c.subject}/{c.sample}"] = (
            np.concatenate([feats_v2(k), imu_feat_dir(c.imu_dir)]))
        if i % 500 == 0:
            print(f"feat cache {i}/{len(sk)}", flush=True)
    print("cached", len(feats), "dims", len(next(iter(feats.values()))))

    folds = split_by_subject(sk, 3)
    OOF = Path("outputs/oof")
    MK = {f: pickle.load(open(OOF / "main_oof.pkl", "rb"))[f] for f in range(3)}
    TK = {f: pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f] for f in range(3)}
    twin_id = {a for t in PAIRS for a in t}

    # ---- per-pair 整体 acc(main vs GBDT) audit ----
    mp = pickle.load(open("outputs/test_teacher_avg_probs.pkl", "rb"))  # noqa # 仅确认可用
    audit = {}
    for fold in range(3):
        tr_idx, va_idx = folds[fold]
        tr_c = [sk[i] for i in tr_idx]
        va_c = [sk[i] for i in va_idx]
        Xtr = [(f"{c.action_id}/{c.subject}/{c.sample}", feats[f"{c.action_id}/{c.subject}/{c.sample}"])
               for c in tr_c if f"{c.action_id}/{c.subject}/{c.sample}" in feats]
        Xva = va_c
        for a, b in PAIRS:
            mti = [(k, f) for k, f in Xtr if int(k.split('/')[0]) in (a, b)]
            clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
            clf.fit(np.array([f for _, f in mti]), [1 if int(k.split('/')[0]) == a else 0 for k, _ in mti])
            for c in Xva:
                key = f"{c.action_id}/{c.subject}/{c.sample}"
                if key not in feats or key not in MK[fold] or key not in TK[fold]:
                    continue
                if c.action_id not in (a, b):
                    continue
                pk = (sf(MK[fold][key]) + sf(TK[fold][key])).argmax()
                g = a if clf.predict([feats[key]])[0] == 1 else b
                audit.setdefault((a, b), [0, 0, 0])
                audit[(a, b)][0] += 1
                audit[(a, b)][1] += int(pk == c.action_id)
                audit[(a, b)][2] += int(g == c.action_id)
    print("\n== 孪生对 audit: main+th 整体 vs 弱模态GBDT 整体 ==")
    weak = []
    for t in PAIRS:
        n, m, g = audit[t]
        if n == 0:
            continue
        tag = "WEAK" if (m / n < g / n - 0.03) else ""
        if tag: weak.append(t)
        print(f"  {t}: n={n} main={100*m/n:.1f}%  GBDT={100*g/n:.1f}%  {tag}")
    print("weak-twin set:", weak)

    # ---- 端到端 overlay(weak-set, 门控=主链 top-1 ∈对) ----
    deltas = []
    for fold in range(3):
        tr_idx, va_idx = folds[fold]
        tr_c = [sk[i] for i in tr_idx]
        va_c = [sk[i] for i in va_idx]
        Xtr = [(f"{c.action_id}/{c.subject}/{c.sample}", feats[f"{c.action_id}/{c.subject}/{c.sample}"])
               for c in tr_c if f"{c.action_id}/{c.subject}/{c.sample}" in feats]
        clfs = {}
        for a, b in weak:
            mti = [(k, f) for k, f in Xtr if int(k.split('/')[0]) in (a, b)]
            clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
            clf.fit(np.array([f for _, f in mti]), [1 if int(k.split('/')[0]) == a else 0 for k, _ in mti])
            clfs[(a, b)] = clf
        yt, yb, yo = [], [], []
        for c in va_c:
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in MK[fold] or key not in TK[fold]:
                continue
            p = (sf(MK[fold][key]) + sf(TK[fold][key])).argmax()
            yt.append(c.action_id); yb.append(p)
            pend = p
            for a, b in weak:
                if p in (a, b) and key in feats:   # 主链 top-1 ∈ weak对
                    g = a if clfs[(a, b)].predict([feats[key]])[0] == 1 else b
                    pend = g
            yo.append(pend)
        deltas.append(np.mean(np.array(yo) == np.array(yt)) - np.mean(np.array(yb) == np.array(yt)))
        print(f"fold{fold} overlay Δ={deltas[-1]:+.4f}", flush=True)
    print(f"\n3折全局 Δ(main top-1 ∈ weak对 换头) mean={np.mean(deltas):+.4f} +- {np.std(deltas):.4f}")


if __name__ == "__main__":
    main()
