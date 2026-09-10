#!/usr/bin/env python3
"""骨架-only 难对 margin-gate fold 复刻（继承 gate_fold_check 逻辑, 去掉 IMU 偏移源）。

动机: sub_hardpair_gate LB-0.5 而 fold+1.73 的根因被疑为 v3 特征里含 IMU 轨迹
(test IMU 角速度=train×1.63 域偏移 → GBDT 在 test 被误导)。骨架 train/test 结构
100% 一致(无域偏移) → 纯骨架 feats_v3 难对 GBDT 可能跨域更稳。

用 OOF 的 main+th fused soft: pred=argmax; margin=soft[pred]-soft[2nd]。
当 pred 落入难对类且 conf 在 [0.5, 0.85] 且 margin<tau → 骨架-only GBDT 换头。
扫 tau 报整体/难对区 acc diff（无 oracle, 与 test 同机制）。
用法: python scripts/gate_fold_check_skel_only.py
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
from twin_v3 import feats_v3, load_kp

HARD = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
HPID = {x for p in HARD for x in p}

CACHE = Path("outputs/skel_v3_only_cache.pkl")


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    import argparse as _ap
    _ap_ = _ap.ArgumentParser()
    _ap_.add_argument("--pairs", type=str, default="",
                      help="额外难对 a-b,c-d（追加 HARD）")
    _aa = _ap_.parse_args()
    global HARD, HPID
    if _aa.pairs.strip():
        for token in _aa.pairs.split(","):
            try:
                a, b = (int(x) for x in token.strip().split("-"))
                if (a, b) not in HARD and (b, a) not in HARD:
                    HARD = HARD + [(a, b)]
            except Exception:
                pass
        HPID = {x for p in HARD for x in p}
        print(f"HARD pairs (含额外): {HARD}", flush=True)

    root = Path("data/Training/HAR")
    sk = [c for c in build_skeleton_index(root) if c.action_id in HPID]
    folds = split_by_subject(sk, 3)
    OOF = Path("outputs/oof")
    MK = {f: pickle.load(open(OOF / "main_oof.pkl", "rb"))[f] for f in range(3)}
    TK = {f: pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f] for f in range(3)}

    if CACHE.exists():
        FE = pickle.load(open(CACHE, "rb"))
        print(f"load skel-only cache n={len(FE)}", flush=True)
    else:
        FE = {}
        for i, c in enumerate(sk):
            k = load_kp(c)
            if k is None:
                continue
            FE[f"{c.action_id}/{c.subject}/{c.sample}"] = feats_v3(k)
            if i % 500 == 0:
                print(f"  feats_v3 {i}/{len(sk)}", flush=True)
        pickle.dump(FE, open(CACHE, "wb"))
    print("skel-only dim:", len(next(iter(FE.values()))), "n:", len(FE), flush=True)

    taus = [0.05, 0.1, 0.2]
    conf_grid = [(0.0, 1.0), (0.5, 0.85), (0.5, 0.95), (0.6, 0.9), (0.4, 0.8)]
    agg = {f"{t}:{cl}-{ch}": [] for t in taus for cl, ch in conf_grid}
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
        pred_b, margin, conf, gb_side, yt, inhp = [], [], [], [], [], []
        for c in va_c:
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in MK[f] or key not in TK[f] or key not in FE:
                continue
            soft = sf(MK[f][key]) + sf(TK[f][key])
            soft = soft / soft.sum()
            pred = int(np.argmax(soft))
            srt = np.sort(soft)[::-1]
            margin.append(float(srt[0] - srt[1]))
            conf.append(float(srt[0]))
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
        conf_arr = np.array(conf)
        yt = np.array(yt); inhp = np.array(inhp); gb_side = np.array(gb_side)
        base = (pred_b == yt).mean()
        hp_mask = inhp
        base_hp = (pred_b[hp_mask] == yt[hp_mask]).mean() if hp_mask.any() else 0.0
        for t in taus:
            for cl, ch in conf_grid:
                # 触发: pred 落难对 + margin<tau + conf∈[cl,ch]
                gate = inhp & (margins < t) & (conf_arr >= cl) & (conf_arr <= ch)
                pred_g = np.where(gate, gb_side, pred_b)
                agg[f"{t}:{cl}-{ch}"].append(((pred_g == yt).mean() - base,
                                               (pred_g[hp_mask] == yt[hp_mask]).mean() - base_hp
                                               if hp_mask.any() else 0.0,
                                               int(gate.sum())))
    print("== 骨架-only 难对 margin-gate(无 oracle) fold 复刻 ==")
    print("  tau:conf            Δ整体(3fold mean±std)      Δ难对区   触发")
    for t in taus:
        for cl, ch in conf_grid:
            key = f"{t}:{cl}-{ch}"
            d_all = np.mean([x[0] for x in agg[key]]); s_all = np.std([x[0] for x in agg[key]])
            d_hp = np.mean([x[1] for x in agg[key]])
            n_tr = np.mean([x[2] for x in agg[key]])
            print(f"  {key:<18} {d_all:+.4f}±{s_all:.4f}           {d_hp:+.4f}   {n_tr:.0f}")


if __name__ == "__main__":
    main()
