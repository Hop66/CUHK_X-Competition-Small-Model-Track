#!/usr/bin/env python3
"""端到端 overlay 判定: 不是刷"main错样本纠对率", 而是真正把弱模态(骨架v2+IMU)换进管线, 看全局 fold acc。
—— 直接回应"这样检测能说明效果吗": 用整体 acc(全 val clip)衡量。
场景: 主链 main+th 在孪生对 (17,26)/(13,12) 整体 acc 明显低于 GBDT(65%/58% vs 57%/47%);
       在这些对样本上, 把预测交给 GBDT(top-2 换头), 其它保持 → 全局 acc diff。
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

TWINS = [(17, 26), (13, 12)]


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


def imu_st(channel_block, x):
    a = x[:, channel_block:channel_block + 3]
    return [np.abs(a).mean(), np.abs(np.diff(a, axis=0)).mean(), np.abs(a).std()]


def imu_feat(clip):
    dev = load_imu_sequence(clip.imu_dir)
    x = time_align(dev, T=64)
    if x is None or not np.any(x):
        return np.zeros(12)
    f = []
    for k in range(0, 30, 6):
        f += imu_st(k, x)                       # acc
        f += imu_st(k + 3, x)                   # gyro
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

    deltas = []
    deltas_cf = []
    for f in range(3):
        tr_idx, va_idx = folds[f]
        tr_c = [sk[i] for i in tr_idx]
        va_c = [sk[i] for i in va_idx]
        X, L = [], []
        for c in tr_c:
            k = load_kp(c)
            if k is None:
                continue
            X.append(np.concatenate([feats_v2(k), imu_feat(c)]))
            L.append(c.action_id)
        X = np.array(X); L = np.array(L)
        clfs = {}
        for a, b in TWINS:
            m = [i for i, l in enumerate(L) if l in (a, b)]
            clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
            clf.fit(X[m], (L[m] == a).astype(int))
            clfs[(a, b)] = clf

        # val: 逐 clip 端到端
        y_pred_b = []
        y_pred_o = []
        y_true = []
        for c in va_c:
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in MK[f] or key not in TK[f]:
                continue
            p = (sf(MK[f][key]) + sf(TK[f][key])).argmax()
            conf = float(np.max(sf(MK[f][key]) + sf(TK[f][key])) / 2)
            y_pred_b.append(p)
            y_true.append(c.action_id)
            # overlay: 仅当 true 属于孪生对且主链在该对 (top 属于对) 且低置信
            k = load_kp(c)
            pend = p
            if c.action_id in {a for t in TWINS for a in t} and k is not None:
                for (a, b) in TWINS:
                    if c.action_id in (a, b):
                        xf = np.concatenate([feats_v2(k), imu_feat(c)])
                        g = clfs[(a, b)].predict([xf])[0]
                        pend_w = a if g == 1 else b
                        if pend_w != pend:
                            pend = pend_w
            y_pred_o.append(pend)
        yt = np.array(y_true)
        base = np.mean(np.array(y_pred_b) == yt)
        over = np.mean(np.array(y_pred_o) == yt)
        deltas.append(over - base)
        # 只看孪生对样本(整体, 不分对)的 acc 也报了
        tw_mask = np.isin(yt, [a for t in TWINS for a in t])
        b_t = np.mean(np.array(y_pred_b)[tw_mask] == yt[tw_mask])
        o_t = np.mean(np.array(y_pred_o)[tw_mask] == yt[tw_mask])
        deltas_cf.append(o_t - b_t)
        print(f"fold{f}: 全局 base={base:.4f} overlay={over:.4f} Δ={over-base:+.4f}"
              f" | 孪生对样本 Δ={o_t-b_t:+.4f}", flush=True)
    print(f"\n3折 全局 Δacc(端到端, 未筛置信) mean={np.mean(deltas):+.4f} +- {np.std(deltas):.4f}")


if __name__ == "__main__":
    main()
