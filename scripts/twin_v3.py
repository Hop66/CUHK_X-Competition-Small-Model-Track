#!/usr/bin/env python3
"""骨架v3(体动门控分层) + IMU完整轨迹(部位/方向/距离) + 欧拉角姿态角 → 孪生对 audit + top-1 overlay。

骨架v3: g=σ(骨盆位移速度) 门控; 特征=[g·全局(骨盆/脚), (1-g)·局部(下臂/腕), g, 分支速度]
IMU轨迹(∫acc): 每设备速度=Σnet_acc, 位移=Σv; 方向直方(象限), 净位移幅度, 弧长, 垂直占比
欧拉角: 板载姿态角变化范围/主旋转轴/正反转比/往返(相对初始朝向)
"""
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.imu_dataset import build_imu_index, load_imu_sequence, time_align
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject

PAIRS = [(13, 12), (22, 21), (8, 9), (8, 10), (37, 6), (18, 17), (7, 6), (36, 32), (26, 24)]
ANGLE = ["角度X(°)", "角度Y(°)", "角度Z(°)"]
CACHE = Path("/tmp/twin_v3_cache.pkl")


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


def feats_v3(kp):
    """体动门控分层: 先算全局体力 g, 再分支加权。"""
    p = kp[:, 0]
    dp = np.diff(p, axis=0)
    rel = kp - kp[:, 0:1, :]
    rel_d = rel - rel[0:1]
    lb = rel_d[:, [4, 7]] - rel_d[:, [3, 6]]
    wr = rel_d[:, [4, 7]]
    ft = rel_d[:, [10, 13]]
    dlb = np.diff(lb, axis=0)
    dwr = np.diff(wr, axis=0)
    dft = np.diff(ft, axis=0)
    dts = np.diff(rel_d[:, [2, 5, 8, 11, 14]], axis=0)
    ax = kp[:, [8]] - kp[:, [0]]
    st = [np.abs(dp).mean(), np.abs(dp).max()]            # 全局体速
    g_speed = np.abs(dp).mean()
    g = float(1 / (1 + math.exp(-(math.log(1 + g_speed) - 1.0))))
    fs_hand = [np.abs(dwr).mean(), np.abs(dwr).max(), np.abs(dlb).mean()]
    fs_foot = [np.abs(dft).mean(), np.abs(dft).max()]
    fs_glo = st + [np.std(kp[:, 0, 2]), np.abs(np.diff(ax, axis=0)).mean()]
    fu = (1 - g) * np.array(fs_hand)
    fw = g * np.array(fs_foot)
    fg = g * np.array(fs_glo)
    return np.concatenate([fg, fu, fw, [g, 1 - g, fs_hand[0], fs_foot[0]]]).astype(np.float32)


def imu_traj(imu_dir):
    """IMU: 轨迹方向/距离 + 欧拉角姿态角(板载). 返回拼接特征 ~ 40 维。"""
    dev = load_imu_sequence(imu_dir)      # {dev:(t,feat[N,6])} acc+gyro
    x = time_align(dev, T=128)            # [T,30]
    f = []
    if x is not None and np.any(x):
        T = x.shape[0]
        for d in range(5):
            a = x[:, d * 6:d * 6 + 3]
            net = a - a.mean(0)
            v = detrend(np.cumsum(net, axis=0), axis=0)
            disp = np.cumsum(v, axis=0)
            # 方向(水平): atan2 of disp; 垂直分量占比
            h = np.arctan2(disp[:, 0], disp[:, 1])       # 水平朝向
            quad = np.histogram(np.abs(h), bins=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], weights=None)[0]
            q = np.histogram(np.abs(h), bins=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])[0]
            quad = q / max(q.sum(), 1)
            net_disp = np.abs(disp[-1])                   # 净位移
            arc = np.abs(np.diff(disp, axis=0)).sum()     # 弧长
            vert = np.abs(disp[:, 2]).mean() / max(np.abs(disp).mean(), 1e-6)
            f += [quad[0], quad[1], quad[2], quad[3], net_disp.mean(), arc.mean(), vert]
    else:
        f += [0.0] * 7 * 5
    # 欧拉角姿态角
    e = []
    for name in ("down(LL+RL).csv", "up(LA+RA+C).csv"):
        fh = imu_dir / name
        if fh.is_file():
            try:
                df = pd.read_csv(fh, encoding="utf-8-sig")
            except Exception:
                df = None
            if df is not None and not df.empty:
                allangles = []
                for col in ANGLE:
                    if col in df.columns:
                        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
                        v = v[np.isfinite(v)]
                        if len(v) >= 2:
                            dv = np.diff(v)
                            rng = float(np.abs(v.max() - v.min()))
                            rate = float(np.abs(dv).mean())
                            pos = float((dv > 0).mean())
                            retro = float(np.abs(np.abs(dv.sum())) / max(np.abs(dv).sum(), 1e-6))
                            s = [rng, rate, pos, retro]
                        else:
                            s = [0.0, 0.0, 0.5, 1.0]
                    else:
                        s = [0.0, 0.0, 0.5, 1.0]
                    allangles.append(s)
                if allangles:
                    e += [np.mean([a[0] for a in allangles]), np.mean([a[1] for a in allangles]),
                          np.mean([a[2] for a in allangles]), np.mean([a[3] for a in allangles])]
                    continue
        e += [0.0, 0.0, 0.5, 1.0]
    return np.concatenate([np.array(f), np.array(e)]).astype(np.float32)


def main():
    root = Path("data/Training/HAR")
    sk = build_skeleton_index(root)
    imu_map = {f"{c.subject}/{c.sample}": c for c in build_imu_index(root, build_train_index(root))}
    sk = [c for c in sk if c.action_id in {a for t in PAIRS for a in t}
          and f"{c.subject}/{c.sample}" in imu_map]
    for c in sk:
        c.imu_dir = imu_map[f"{c.subject}/{c.sample}"].imu_dir

    if CACHE.exists():
        FE = pickle.load(open(CACHE, "rb"))
    else:
        FE = {}
        for i, c in enumerate(sk):
            k = load_kp(c)
            if k is None:
                continue
            FE[f"{c.action_id}/{c.subject}/{c.sample}"] = np.concatenate([feats_v3(k), imu_traj(c.imu_dir)])
            if i % 400 == 0:
                print(f"v3 cache {i}/{len(sk)}", flush=True)
        pickle.dump(FE, open(CACHE, "wb"))
    print("v3 dim:", len(next(iter(FE.values()))), "n:", len(FE))

    folds = split_by_subject(sk, 3)
    OOF = Path("outputs/oof")
    MK = {f: pickle.load(open(OOF / "main_oof.pkl", "rb"))[f] for f in range(3)}
    TK = {f: pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f] for f in range(3)}
    audit = {}
    for fold in range(3):
        tr_idx, va_idx = folds[fold]
        tr_c = [sk[i] for i in tr_idx]
        va_c = [sk[i] for i in va_idx]
        Xtr = [(f"{c.action_id}/{c.subject}/{c.sample}", FE[f"{c.action_id}/{c.subject}/{c.sample}"])
               for c in tr_c if f"{c.action_id}/{c.subject}/{c.sample}" in FE]
        for a, b in PAIRS:
            mti = [(k, fx) for k, fx in Xtr if int(k.split('/')[0]) in (a, b)]
            clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
            clf.fit(np.array([fx for _, fx in mti]), [1 if int(k.split('/')[0]) == a else 0 for k, _ in mti])
            for c in va_c:
                key = f"{c.action_id}/{c.subject}/{c.sample}"
                if key not in FE or key not in MK[fold] or key not in TK[fold]:
                    continue
                if c.action_id not in (a, b):
                    continue
                pk = (sf(MK[fold][key]) + sf(TK[fold][key])).argmax()
                g = a if clf.predict([FE[key]])[0] == 1 else b
                audit.setdefault((a, b), [0, 0, 0])
                audit[(a, b)][0] += 1
                audit[(a, b)][1] += int(pk == c.action_id)
                audit[(a, b)][2] += int(g == c.action_id)
    print("\n== 骨架v3+IMU完整: 孪生对整体 main vs v3 ==")
    weak = []
    for t in PAIRS:
        n, m, g = audit[t]
        if n == 0:
            continue
        tag = "WEAK" if (m / n < g / n - 0.03) else ""
        if tag:
            weak.append(t)
        print(f"  {t}: n={n} main={100*m/n:.1f}%  v3={100*g/n:.1f}%  {tag}")
    print("weak:", weak)


if __name__ == "__main__":
    main()
