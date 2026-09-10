#!/usr/bin/env python3
"""骨架v2(位姿不变运动学) + IMU 双流, 对孪生对(main+th 已错样本)条件纠错 fold 验证。

用户理论:
 1) 同类动作不同样本"初始姿态基线"未删 → 模型把开始姿势当类别线索 → 跨样本泛化崩(DG by 位姿不变)
 2) 精细动作靠"关键关节运动学"(下臂动 vs 手部动) → 17关节有小臂(wrist-elbow)可算
 3) 骨架+IMU 双流(骨骼=姿态运动, IMU=肢体加速度)互补

对比: 基础特征(仅 per-frame pelvis 平移) vs 骨架v2(去初始基线+段运动学+躯体动静) vs 骨架v2+IMU
判据: 在 main+th 已判错的孪生样本上 GBDT 纠对率 (基线≈49%随机)。
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.imu_dataset import IMUDataset, build_imu_index, load_imu_sequence, time_align, FEAT_DIM
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.split import split_by_subject

TWINS = [(13, 12), (22, 21), (8, 9), (8, 10), (37, 6), (18, 17), (7, 6), (36, 32), (26, 24)]


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


def _speed_stat(dv):
    sp = np.abs(dv)
    return [sp.mean(), sp.max(), sp.std()]


def feats_v1(kp):
    """基线(当前在用的): per-frame pelvis 平移, 含初始姿态残留。"""
    rel = kp - kp[:, 0:1, :]
    drel = np.diff(rel, axis=0)
    p = kp[:, 0]
    dp = np.diff(p, axis=0)
    return np.array(_speed_stat(dp) + _speed_stat(drel) + [np.std(kp[:, 0, 2])])


def feats_v2(kp):
    """骨架v2: 去初始姿态基线 + 段运动学(下臂/手/脚/躯干) + 躯体动/静比值。"""
    p = kp[:, 0]
    dp = np.diff(p, axis=0)
    rel = kp - kp[:, 0:1, :]
    rel_d = rel - rel[0:1]            # ← 删初始姿态基线(位姿不变 DG 核心)
    drel = np.diff(rel_d, axis=0)
    # 段速度: 下臂 wrist-elbow(4-3, 7-6), 手=wrist, 脚=ankle(10,13), 躯干(8-pelvis)
    lb = (rel_d[:, [4, 7]] - rel_d[:, [3, 6]])      # 下臂向量 (T,2,3)
    dlb = np.diff(lb, axis=0)
    wrist = rel_d[:, [4, 7]]
    dwr = np.diff(wrist, axis=0)
    foot = rel_d[:, [10, 13]]
    dfoot = np.diff(foot, axis=0)
    torso = rel_d[:, [2, 5, 8, 11, 14]]
    dtorso = np.diff(torso, axis=0)
    # 躯干轴线(骨盆→颈8)倾角变化 = 躯体带动
    ax = kp[:, [8]] - kp[:, [0]]
    tilt = np.diff(ax, axis=0)                        # (T-1,1,3)
    hv = _speed_stat(dwr)
    fv = _speed_stat(dfoot)
    tv = _speed_stat(dtorso)
    lv = _speed_stat(dlb)
    gv = _speed_stat(dp)
    td = np.abs(tilt).mean()
    return np.array(gv + hv + fv + tv + lv +
                    [hv[0] / max(tv[0], 1e-6), fv[0] / max(tv[0], 1e-6),
                     lv[0] / max(tv[0], 1e-6), td, np.std(kp[:, 0, 2])])


def imu_feat(clip):
    dev = load_imu_sequence(clip.imu_dir)
    x = time_align(dev, T=64)
    if x is None or not np.any(x):
        return np.zeros(14)
    mag = np.abs(x)
    f = [mag.mean(), mag.std(),
         np.abs(np.diff(x, axis=0)).mean(), np.abs(np.diff(x, axis=0)).max()]
    for k in range(0, 30, 6):
        f += [mag[:, k:k + 6].mean(), mag[:, k:k + 6].std()]
    return np.array(f[:14])


def build(clips_sel, use_imu):
    X1, X2, L, C = [], [], [], []
    for c in clips_sel:
        k = load_kp(c)
        if k is None:
            continue
        if use_imu:
            imuf = imu_feat(c)
            if imuf is None:
                continue
            X2.append(np.concatenate([feats_v2(k), imuf]))
        else:
            X2.append(feats_v2(k))
        X1.append(feats_v1(k))
        L.append(c.action_id)
        C.append(c)
    return np.array(X1), np.array(X2), np.array(L), C


def gbdt_fit(X, L, a, b):
    m = [i for i, l in enumerate(L) if l in (a, b)]
    clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
    clf.fit(X[m], (L[m] == a).astype(int))
    return clf


def cond_acc_probe(feat_fn, train_c, va_c, use_imu, tag):
    """main+th 已错的孪生样本 → 该特征分类器纠对率。"""
    Xtr = feat_fn([train_c[i] for i in range(len(train_c))], use_imu)
    per_t = {}
    tot = [0, 0]
    folds_tr = None
    for (a, b) in TWINS:
        clf1 = gbdt_fit(Xtr[1], Xtr[2], a, b)
        clf2 = gbdt_fit(Xtr[0], Xtr[2], a, b)
        # val 孪生样本
        Xv = feat_fn(va_c, use_imu)
        for j, c in enumerate(va_c):
            if c.action_id not in (a, b):
                continue
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in MAIN_K[fold] or key not in TH_K[fold]:
                continue
            if (sf(MAIN_K[fold][key]) + sf(TH_K[fold][key])).argmax() == c.action_id:
                continue
            k = load_kp(c)
            if k is None:
                continue
            pr = a if clf2.predict([Xv[1][j]])[0] == 1 else b   # v2+?
            per_t.setdefault((a, b), [0, 0])
            per_t[(a, b)][0] += 1
            per_t[(a, b)][1] += int(pr == c.action_id)
            tot[0] += 1
            tot[1] += int(pr == c.action_id)
    return per_t, tot


def main():
    root = Path("data/Training/HAR")
    sk = build_skeleton_index(root)
    main_clips = build_train_index(root)
    imu = {c.subject: c for c in []}  # placeholder
    # 构建 IMU clip 映射 (subject/sample → imu clip)
    imu_clips = build_imu_index(root, main_clips)
    imu_map = {f"{c.subject}/{c.sample}": c for c in imu_clips}
    sk = [c for c in sk if f"{c.subject}/{c.sample}" in imu_map]
    for c in sk:
        c.imu_dir = imu_map[f"{c.subject}/{c.sample}"].imu_dir

    folds = split_by_subject(sk, 3)
    global MAIN_K, TH_K, fold
    OOF = Path("outputs/oof")
    MAIN_K = {}
    TH_K = {}
    for f in range(3):
        MAIN_K[f] = pickle.load(open(OOF / "main_oof.pkl", "rb"))[f]
        TH_K[f] = pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f]

    for use_imu in (False, True):
        per_t = {}
        tot = [0, 0]
        X1s, X2s, Ls = [], [], []
        for f in range(3):
            tr_idx, va_idx = folds[f]
            tr_c = [sk[i] for i in tr_idx]
            va_c = [sk[i] for i in va_idx]
            X1, X2, L, _ = build(tr_c, use_imu)
            for a, b in TWINS:
                clf2 = gbdt_fit(X2, L, a, b)   # v2(+imu)
                Xv1, Xv2, Lv, Cv = build(va_c, use_imu)
                for j, c in enumerate(Cv):
                    if c.action_id not in (a, b):
                        continue
                    key = f"{c.action_id}/{c.subject}/{c.sample}"
                    if key not in MAIN_K[f] or key not in TH_K[f]:
                        continue
                    if (sf(MAIN_K[f][key]) + sf(TH_K[f][key])).argmax() == c.action_id:
                        continue
                    pr = a if clf2.predict([Xv2[j]])[0] == 1 else b
                    per_t.setdefault((a, b), [0, 0])
                    per_t[(a, b)][0] += 1
                    per_t[(a, b)][1] += int(pr == c.action_id)
                    tot[0] += 1
                    tot[1] += int(pr == c.action_id)
        tag = "骨架v2+IMU" if use_imu else "骨架v2(去基线)"
        print(f"\n== {tag}: main已错孪生样本条件纠错 ==")
        for t in per_t:
            n, c = per_t[t]
            print(f"  {t}: n={n} 纠对率={100*c/n:.1f}%")
        print(f"  合计 n={tot[0]} 纠对率={100*tot[1]/max(tot[0],1):.1f}%")


if __name__ == "__main__":
    main()
