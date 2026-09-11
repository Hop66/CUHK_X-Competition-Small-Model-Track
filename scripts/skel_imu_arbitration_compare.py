#!/usr/bin/env python3
"""四仲裁方案对比（骨架主/IMU主/双一致/动态强判别）—— 锚同源 test 侧, 真实类映射。

判别器全部用全体 train 训练（与锚 main_s42 同源），test 用 test_v3_cache 前13维 / test_imu_pose_agg 60d。
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.split import split_by_subject

FE = pickle.load(open("outputs/skel_v3_extra_cache.pkl", "rb"))
IMU = pickle.load(open("outputs/imu_pose_agg_bench.pkl", "rb"))
TVC = pickle.load(open("outputs/test_v3_cache.pkl", "rb"))
TIMU = pickle.load(open("outputs/test_imu_pose_agg.pkl", "rb"))
probs = pickle.load(open("outputs/oof/anchor_softprobs.pkl", "rb"))
KEYS = sorted(probs.keys())
PA = np.stack([probs[k] for k in KEYS])

PAIRS = [(5, 31), (6, 37), (7, 19), (7, 20), (11, 14), (17, 26), (17, 18),
         (25, 24), (6, 20), (8, 10), (13, 12), (22, 21), (26, 24)]


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    soft = sf(PA)
    pred = soft.argmax(-1)
    top2 = []
    for i in range(len(KEYS)):
        arr = np.argsort(soft[i])[::-1][:2]
        top2.append((int(arr[0]), int(arr[1])))

    clips = build_train_index(Path("data/Training/HAR"))
    folds = split_by_subject(clips, 3)
    tr_s = [{clips[i].subject for i in folds[f][0]} for f in range(3)]
    subjS = {k: k.split("/")[1] for k in FE}
    subjI = {k: k.split("/")[1] for k in IMU}

    def mk(pair, fd, sm, tr=None):
        a, b = pair
        sel = [k for k in fd if int(k.split("/")[0]) in (a, b)]
        if tr is not None:
            sel = [k for k in sel if sm.get(k) in tr]
        if len(set(int(k.split("/")[0]) for k in sel)) < 2 or len(sel) < 10:
            return None
        X = np.stack([np.asarray(fd[k], np.float32) for k in sel])
        y = np.array([a if int(k.split("/")[0]) == a else b for k in sel])
        return GradientBoostingClassifier(n_estimators=140, max_depth=3, random_state=0).fit(X, y)

    def pdir(clf, x):
        if clf is None:
            return None
        xx = np.asarray(x, np.float32).reshape(1, -1)
        cls = list(clf.classes_)
        return cls[int(clf.predict_proba(xx)[0].argmax())]

    # 跨折判别力
    fp = {}
    for p in PAIRS:
        sa, ia = [], []
        for f in range(3):
            for c, acc, sm, fd in (("s", sa, subjS, FE), ("i", ia, subjI, IMU)):
                c2 = mk(p, fd, sm, tr_s[f])
                if c2 is None:
                    continue
                va = [k for k in fd if int(k.split("/")[0]) in p and sm.get(k) not in tr_s[f]]
                if len(va) < 3 or len(set(int(k.split("/")[0]) for k in va)) < 2:
                    continue
                yv = np.array([int(k.split("/")[0]) for k in va])
                acc.append((c2.predict(np.stack([np.asarray(fd[k], np.float32) for k in va])) == yv).mean())
        fp[p] = (np.mean(sa) if sa else -1, np.mean(ia) if ia else -1)
    print("=== 对判别力（跨折）===")
    for p, (s, i) in fp.items():
        print(f"  {p}: skel={s:.3f} imu={i:.3f}")

    full = {p: (mk(p, FE, subjS), mk(p, IMU, subjI)) for p in PAIRS}

    def build(mode):
        out = []
        for i, k in enumerate(KEYS):
            a, b = top2[i]
            p = (a, b) if (a, b) in full else ((b, a) if (b, a) in full else None)
            if p is None:
                continue
            cs, ci = full[p]
            p0 = int(pred[i])
            ds = pdir(cs, TVC[k][:13]) if k in TVC else None
            dI = pdir(ci, TIMU[k]) if k in TIMU else None
            s_acc, i_acc = fp[p]
            if mode == "skel":
                if ds is not None and ds != p0:
                    out.append((k, int(ds), p))
            elif mode == "imu":
                if dI is not None and dI != p0:
                    out.append((k, int(dI), p))
            elif mode == "agree":
                if ds is not None and dI is not None and ds == dI and ds != p0:
                    out.append((k, int(ds), p))
            elif mode == "dyn":
                if s_acc > i_acc:
                    mn, oth, mnacc = ds, dI, s_acc
                else:
                    mn, oth, mnacc = dI, ds, i_acc
                if mn is not None and mn != p0 and mnacc >= 0.65:
                    if oth is None or oth == mn or min(s_acc, i_acc) < 0.55:
                        out.append((k, int(mn), p))
        return out

    print()
    for mode, name in [("skel", "骨架主"), ("imu", "IMU主"), ("agree", "双一致"), ("dyn", "动态(强判别主)")]:
        fs = build(mode)
        print(f"{name}: {len(fs)} flip")
        for k, d, p in fs:
            print(f"    {k}: pair{p} 判→{d}  (锚={int(pred[KEYS.index(k)])})")
        print()


if __name__ == "__main__":
    main()
