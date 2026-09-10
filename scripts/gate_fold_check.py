#!/usr/bin/env python3
"""难对 margin-gate override 的 fold 复刻(无 oracle, 与 test gate 同机制)。

用 OOF 的 main+th fused soft: pred = argmax(fused); margin = soft[pred]-soft[2nd]。
当 pred 落入难对类且 margin<tau 时, 由难对骨架 GBDT 判定换头。
扫 tau 报整体/难对区 acc diff —— 逻辑与 make_test_gate_csv.py 完全一致
(基于 pred 而非 true∈对), 用于选定 test 用 tau 并验证"只救不伤"在无 oracle 下成立。
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.skeleton_dataset import build_skeleton_index
from src.split import split_by_subject

HARD = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
HPID = {x for p in HARD for x in p}


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    root = Path("data/Training/HAR")
    sk = [c for c in build_skeleton_index(root) if c.action_id in HPID]
    folds = split_by_subject(sk, 3)
    FE = pickle.load(open("outputs/twin_v3_cache.pkl", "rb"))
    OOF = Path("outputs/oof")
    MK = {f: pickle.load(open(OOF / "main_oof.pkl", "rb"))[f] for f in range(3)}
    TK = {f: pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f] for f in range(3)}

    taus = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
    agg = {t: [] for t in taus}
    for f in range(3):
        tr_idx, va_idx = folds[f]
        tr_c = [sk[i] for i in tr_idx]
        va_c = [sk[i] for i in va_idx]
        clfs = {}
        for a, b in HARD:
            mti = []
            for c in tr_c:
                k = f"{c.action_id}/{c.subject}/{c.sample}"
                if k in FE:
                    mti.append((k, FE[k]))
            if len(mti) < 12:
                continue
            clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
            clf.fit(np.array([v for _, v in mti]),
                    [1 if int(k.split('/')[0]) == a else 0 for k, _ in mti])
            clfs[(a, b)] = clf
        pred_b, margin, gb_side, yt, inhp = [], [], [], [], []
        for c in va_c:
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in MK[f] or key not in TK[f] or key not in FE:
                continue
            soft = sf(MK[f][key]) + sf(TK[f][key])
            soft = soft / soft.sum()
            pred = int(np.argmax(soft))
            srt = np.sort(soft)[::-1]
            margin.append(float(srt[0] - srt[1]))
            pred_b.append(pred)
            yt.append(int(c.action_id))
            inhp.append(pred in HPID)
            gb = pred
            if pred in HPID:
                for a, b in HARD:
                    if pred not in (a, b):
                        continue
                    clf = clfs.get((a, b))
                    if clf is None:
                        break
                    cls = list(clf.classes_)
                    ia, ib = cls.index(1), cls.index(0)
                    pab = clf.predict_proba([FE[key]])[0]
                    gb = a if pab[ia] >= pab[ib] else b
                    break
            gb_side.append(gb)
        pred_b = np.array(pred_b); margins = np.array(margin)
        yt = np.array(yt); inhp = np.array(inhp); gb_side = np.array(gb_side)
        base = (pred_b == yt).mean()
        hp_mask = inhp
        base_hp = (pred_b[hp_mask] == yt[hp_mask]).mean() if hp_mask.any() else 0.0
        for t in taus:
            pred_g = np.where((inhp & (margins < t)), gb_side, pred_b)
            agg[t].append(((pred_g == yt).mean() - base,
                           (pred_g[hp_mask] == yt[hp_mask]).mean() - base_hp
                           if hp_mask.any() else 0.0))
    print("== 难对 margin-gate(无 oracle) fold 复刻 ==")
    print("  tau   Δ整体(3fold mean±std)      Δ难对区")
    for t in taus:
        d_all = np.mean([x[0] for x in agg[t]]); s_all = np.std([x[0] for x in agg[t]])
        d_hp = np.mean([x[1] for x in agg[t]])
        print(f"  {t:.2f}  {d_all:+.4f}±{s_all:.4f}           {d_hp:+.4f}")


if __name__ == "__main__":
    main()
