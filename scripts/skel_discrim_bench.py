#!/usr/bin/env python3
"""骨架难对判别方法基准：v1/v2/v3 及组合，在每对难对上 3 折 GBDT 二分类 acc。
回答"升级骨架哪个方法/组合能更好分类所有难对"。
- v1: 基线(含初始姿态残留 + 速度)
- v2: 去初始姿态基线 + 段运动学(下臂/手/脚/躯干) + 动/静比
- v3: v2 + 体动门控分层(g) 加权
- 组合: v1+v2 / v2+v3 / v1+v2+v3 (concat) → 判别力是否互补
- 另加「角度特征 v4」: 关节角度(臂肘角/躯干倾角)变化率 = 细动作判别(对付同幅难对)
用法: python scripts/skel_discrim_bench.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.skeleton_dataset import build_skeleton_index
from src.split import split_by_subject
from skel_imu_twin_v2 import feats_v1, feats_v2
from twin_v3 import feats_v3, load_kp

HARD = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24), (9, 10), (6, 7), (11, 14)]
CACHE = Path("outputs/skel_feats_bench.pkl")


def feats_v4(kp):
    """角度/姿态变化率: 肘角(wrist-elbow-shoulder)、肩-肘、躯干倾角 → 细动作判别。"""
    a = kp  # [T,17,3]
    # 肘角: 肩(3,6)-肘(4,7)-腕(5,8) 夹角
    def ang(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6
        c = np.clip(np.sum(v1 * v2, axis=-1) / n, -1, 1)
        return np.arccos(c)
    # 左臂: 肩3 肘4 腕5 ; 右臂: 肩6 肘7 腕8
    elb_l = ang(a[:, 3], a[:, 4], a[:, 5])
    elb_r = ang(a[:, 6], a[:, 7], a[:, 8])
    # 躯干倾角(骨盆0→颈8) 与世界 z
    ax = a[:, 8] - a[:, 0]
    axn = ax / (np.linalg.norm(ax, axis=-1, keepdims=True) + 1e-6)
    z = np.array([0, 0, 1.0])
    tilt = np.arccos(np.clip(np.dot(axn, z), -1, 1))
    # 膝角(hip-1,2 / knee-9,12 / ankle-10,13)
    kne_l = ang(a[:, 1], a[:, 9], a[:, 10])
    kne_r = ang(a[:, 2], a[:, 12], a[:, 13])
    dels = [np.diff(x, axis=0) for x in (elb_l, elb_r, tilt, kne_l, kne_r)]
    def st(x):  # speed stat: mean/max
        return [np.abs(x).mean(), np.abs(x).max(), np.std(x)]
    return np.concatenate([st(x) for x in dels]).astype(np.float32)


def build_feats():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    root = Path("data/Training/HAR")
    sk = build_skeleton_index(root)
    FE = {}
    for i, c in enumerate(sk):
        k = load_kp(c)
        if k is None:
            continue
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        FE[key] = {"v1": feats_v1(k), "v2": feats_v2(k), "v3": feats_v3(k), "v4": feats_v4(k)}
        if i % 800 == 0:
            print(f"  feats {i}/{len(sk)}", flush=True)
    pickle.dump(FE, open(CACHE, "wb"))
    return FE


def main():
    root = Path("data/Training/HAR")
    FE = build_feats()
    print(f"feats n={len(FE)} v1d={len(next(iter(FE.values()))['v1'])} "
          f"v2d={len(next(iter(FE.values()))['v2'])} v3d={len(next(iter(FE.values()))['v3'])} "
          f"v4d={len(next(iter(FE.values()))['v4'])}")

    METHODS = {
        "v1": lambda d: d["v1"], "v2": lambda d: d["v2"], "v3": lambda d: d["v3"],
        "v4": lambda d: d["v4"],
        "v12": lambda d: np.concatenate([d["v1"], d["v2"]]),
        "v23": lambda d: np.concatenate([d["v2"], d["v3"]]),
        "v24": lambda d: np.concatenate([d["v2"], d["v4"]]),
        "v234": lambda d: np.concatenate([d["v2"], d["v3"], d["v4"]]),
        "ALL": lambda d: np.concatenate([d["v1"], d["v2"], d["v3"], d["v4"]]),
    }
    allsk = [c for c in build_skeleton_index(root) if c.action_id in {x for p in HARD for x in p}]
    folds = split_by_subject(allsk, 3)
    print(f"\n{'对':<10}{'n':<5}" + "".join(f"{m:<18}" for m in METHODS))
    for pair in HARD:
        accs = {m: [] for m in METHODS}
        ns = []
        for f in range(3):
            tr, va = folds[f]
            for m, fn in METHODS.items():
                Xtr = []; ytr = []; Xva = []; yva = []
                for c in [allsk[i] for i in tr]:
                    k = f"{c.action_id}/{c.subject}/{c.sample}"
                    if k in FE and c.action_id in pair:
                        Xtr.append(fn(FE[k])); ytr.append(1 if c.action_id == pair[0] else 0)
                for c in [allsk[i] for i in va]:
                    k = f"{c.action_id}/{c.subject}/{c.sample}"
                    if k in FE and c.action_id in pair:
                        Xva.append(fn(FE[k])); yva.append(1 if c.action_id == pair[0] else 0)
                if len(Xtr) < 8 or len(set(ytr)) < 2 or len(Xva) < 4:
                    continue
                clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
                clf.fit(np.array(Xtr), ytr)
                accs[m].append(np.mean(np.array(clf.predict(np.array(Xva))) == np.array(yva)))
            ns.append(len([c for c in [allsk[i] for i in va] if c.action_id in pair]))
        row = f"{pair[0]}-{pair[1] if len(str(pair[1]))==2 else pair[1]:<2}"
        row = f"{pair}: "
        nline = f"{pair}: n={int(np.mean(ns)) if ns else 0:<4}".replace("n=inf", "")
        line = f"{str(pair):<10}{int(np.mean(ns)) if ns else 0:<5}"
        for m in METHODS:
            a = np.mean(accs[m]) if accs[m] else float("nan")
            line += f"{a if not np.isnan(a) else 0.0:<10.3f}acentered "
            #line += f"{m}:{a:.3f} "
        print(line)


if __name__ == "__main__":
    main()
