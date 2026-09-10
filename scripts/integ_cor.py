#!/usr/bin/env python3
"""骨架v2 + IMU 去偏积分轨迹特征(速度/位移幅度) —— 用户问的"积分算轨迹"。
实现(惯性导航标准处理的轻量版):
  1) acc 去序列均值(估/减零偏+姿态无关地削掉低频) 
  2) 速度 = cumsum(net_acc) → 再加 detrend 去积分漂移
  3) 位移 = cumsum(v) → 幅度 max-min
  特征: 每设备(腕/踝/腰) 速度std、位移幅度 + 腰/踝比率 类"整体活动量/平移度"
对孪生对(main+th 已错样本)条件纠错 fold 判定。
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import detrend
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.imu_dataset import build_imu_index, load_imu_sequence, time_align
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject

TWINS = [(17, 26), (13, 12), (22, 21)]


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


def integ_feat(clip):
    dev = load_imu_sequence(clip.imu_dir)
    x = time_align(dev, T=128)  # [T,30] acc3+gyro3 ×5dev
    if x is None or not np.any(x):
        return np.zeros(12)
    f = []
    T = x.shape[0]
    for d in range(5):
        a = x[:, d * 6:d * 6 + 3]          # 各设备 acc(近似重力+动态)
        a = a - a.mean(0)                   # 去零偏(粗略, 去重力常量部分)
        v = np.cumsum(a - a.mean(0), axis=0)   # 速度(带残余漂移)
        v = detrend(v, axis=0)              # 去线性漂移(标准做法)
        disp = np.cumsum(v, axis=0)
        f += [np.abs(v).std(), (disp.max(0) - disp.min(0)).sum()]
    # 腰(4) vs 踝(0,1): 整体平移度 ratio
    waist = f[8]  # speed std waist
    ank = (f[0] + f[2]) / 2
    f += [waist / max(ank, 1e-6), ank]
    return np.array(f[:12], np.float32)


def main():
    root = Path("data/Training/HAR")
    sk = build_skeleton_index(root)
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(root, build_train_index(root))}
    sk = [c for c in sk if f"{c.subject}/{c.sample}" in imu_map]
    for c in sk:
        c.imu_dir = imu_map[f"{c.subject}/{c.sample}"].imu_dir

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
            X.append(np.concatenate([feats_v2(k), integ_feat(c)]))
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
                xf = np.concatenate([feats_v2(k), integ_feat(c)])
                pr = a if clf.predict([xf])[0] == 1 else b
                per_t.setdefault((a, b), [0, 0])
                per_t[(a, b)][0] += 1
                per_t[(a, b)][1] += int(pr == c.action_id)
                tot[0] += 1
                tot[1] += int(pr == c.action_id)
    print("== 骨架v2 + IMU 积分轨迹(速度/位移幅度): main已错孪生样本条件纠对 ==")
    for t in per_t:
        n, cc = per_t[t]
        print(f"  {t}: n={n} 纠对率={100*cc/n:.1f}%")
    print(f"  合计 n={tot[0]} 纠对率={100*tot[1]/max(tot[0],1):.1f}%")


if __name__ == "__main__":
    main()
