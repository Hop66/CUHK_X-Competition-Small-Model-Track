#!/usr/bin/env python3
"""快速补跑: 骨架v2 + IMU 双流(预计算缓存), 对你点名的难对 (17,26)(13,12)(22,21) 条件纠错。"""
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

TWINS = [(17, 26), (13, 12), (22, 21)]
CACHE = Path("/tmp/imu_coro_cache.pkl")


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
    drel = np.diff(rel_d, axis=0)
    lb = (rel_d[:, [4, 7]] - rel_d[:, [3, 6]])
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


def imu_feat(dev):
    x = time_align(dev, T=64)
    if x is None or not np.any(x):
        return np.zeros(14)
    mag = np.abs(x)
    f = [mag.mean(), mag.std(), np.abs(np.diff(x, axis=0)).mean(),
         np.abs(np.diff(x, axis=0)).max()]
    for k in range(0, 30, 6):
        f += [mag[:, k:k + 6].mean(), mag[:, k:k + 6].std()]
    return np.array(f[:14])


def main():
    root = Path("data/Training/HAR")
    sk = build_skeleton_index(root)
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(root, build_train_index(root))}
    sk = [c for c in sk if f"{c.subject}/{c.sample}" in imu_map]
    for c in sk:
        c.imu_dir = imu_map[f"{c.subject}/{c.sample}"].imu_dir

    # 预计算一次 IMU 特征缓存
    if CACHE.exists():
        imu_cache = pickle.load(open(CACHE, "rb"))
    else:
        imu_cache = {}
        for i, c in enumerate(sk):
            try:
                imu_cache[f"{c.subject}/{c.sample}"] = imu_feat(load_imu_sequence(c.imu_dir))
            except Exception:
                imu_cache[f"{c.subject}/{c.sample}"] = np.zeros(14)
            if i % 200 == 0:
                print(f"imu cache {i}/{len(sk)}", flush=True)
        pickle.dump(imu_cache, open(CACHE, "wb"))

    folds = split_by_subject(sk, 3)
    OOF = Path("outputs/oof")
    MK = {f: pickle.load(open(OOF / "main_oof.pkl", "rb"))[f] for f in range(3)}
    TK = {f: pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f] for f in range(3)}

    per_t = {}
    tot = [0, 0]
    for f in range(3):
        tr_idx, va_idx = folds[f]
        tr_c = [sk[i] for i in tr_idx]
        va_c = [sk[i] for i in va_idx]
        X, L = [], []
        for c in tr_c:
            k = load_kp(c)
            if k is None:
                continue
            X.append(np.concatenate([feats_v2(k), imu_cache[f"{c.subject}/{c.sample}"]]))
            L.append(c.action_id)
        X = np.array(X); L = np.array(L)
        for a, b in TWINS:
            m = [i for i, l in enumerate(L) if l in (a, b)]
            clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
            clf.fit(X[m], (L[m] == a).astype(int))
            for c in va_c:
                if c.action_id not in (a, b):
                    continue
                key = f"{c.action_id}/{c.subject}/{c.sample}"
                if key not in MK[f] or key not in TK[f]:
                    continue
                if (sf(MK[f][key]) + sf(TK[f][key])).argmax() == c.action_id:
                    continue
                k = load_kp(c)
                if k is None:
                    continue
                xf = np.concatenate([feats_v2(k), imu_cache[f"{c.subject}/{c.sample}"]])
                pr = a if clf.predict([xf])[0] == 1 else b
                per_t.setdefault((a, b), [0, 0])
                per_t[(a, b)][0] += 1
                per_t[(a, b)][1] += int(pr == c.action_id)
                tot[0] += 1
                tot[1] += int(pr == c.action_id)
    print("== 骨架v2+IMU 双流: main已错孪生样本条件纠对 ==")
    for t in per_t:
        n, cc = per_t[t]
        print(f"  {t}: n={n} 纠对率={100*cc/n:.1f}%")
    print(f"  合计 n={tot[0]} 纠对率={100*tot[1]/max(tot[0],1):.1f}%")


if __name__ == "__main__":
    main()
