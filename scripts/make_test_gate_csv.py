#!/usr/bin/env python3
"""CUHK-X —— test 难对门控 override(推理侧, 只救不伤) → LB 实测版。

对比对象: 训练侧特征融合(hardpair-aux, main 真正学到)。
本脚本 = 推理侧 override: main top 预测落入难对类集合 且 难对骨架 GBDT 高置信 才换头。
数据驱动难对集(与 L2-hardpairs 相同): HARD=[(13,12),(22,21),(8,10),(18,17),(26,24)]。

用法:
  python scripts/make_test_gate_csv.py \
    --anchor outputs/oof/anchor_softprobs.pkl \
    --out outputs/sub_hardpair_gate.csv
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

HARD = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
HPID = {x for p in HARD for x in p}


class Clip:
    __slots__ = ("pred_dir", "imu_dir")


def main():
    global HARD, HPID
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="outputs/oof/anchor_softprobs.pkl")
    ap.add_argument("--train_cache", default="outputs/twin_v3_cache.pkl")
    ap.add_argument("--test_cache", default="outputs/test_v3_cache.pkl")
    ap.add_argument("--pairs", type=str, default="",
                    help="额外难对: 逗号分隔 'a-b,c-d'，追加到基础 5 对（如 '17-18'=键盘↔写字）")
    ap.add_argument("--pairs_keep", type=str, default="",
                    help="只对指定对启用 gate（fold 净正对），如 '13-12,22-21'；空=全部 HARD")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="主链 margin 阈值: soft[top1]-soft[top2] < tau 才触发难对 override"
                         "(由 gate_fold_check.py 在 fold 上选定)")
    ap.add_argument("--skel_only", action="store_true",
                    help="只用骨架 v3 特征(前13维 feats_v3, 去掉 IMU 域偏移 v3+imu 混合特征)。"
                         "test IMU 幅值 1.63× 域偏移是 gate fold+1.73→LB-0.5 反噬根因")
    ap.add_argument("--conf_lo", type=float, default=0.5,
                    help="触发仅当主链置信(soft top1) ∈ [conf_lo, conf_hi]")
    ap.add_argument("--conf_hi", type=float, default=0.85,
                    help="排除高置信(>conf_hi): main 大概率对, 避免翻转")
    ap.add_argument("--out", default="outputs/sub_hardpair_gate.csv")
    args = ap.parse_args()

    # 额外难对（语义外）：'a-b,c-d' → 追加 HARD
    if args.pairs.strip():
        for token in args.pairs.split(","):
            try:
                a, b = (int(x) for x in token.strip().split("-"))
                if (a, b) not in HARD and (b, a) not in HARD:
                    HARD = HARD + [(a, b)]
            except Exception:
                pass
        HPID = {x for p in HARD for x in p}
        print(f"HARD pairs (含额外): {HARD}", flush=True)

    # 只保留 fold 净正对
    KEEP = set()
    if args.pairs_keep.strip():
        for token in args.pairs_keep.split(","):
            try:
                a, b = (int(x) for x in token.strip().split("-"))
                KEEP.add(tuple(sorted((a, b))))
            except Exception:
                pass
        print(f"gate 仅启用对: {sorted(KEEP)}", flush=True)

    # 1) train 特征(全体难对样本) -> 每难对 GBC
    FE = pickle.load(open(args.train_cache, "rb"))
    from twin_v3 import load_kp, feats_v3, imu_traj
    DIM = 13 if args.skel_only else None   # feats_v3 骨架-only 前13维 / None=全56(v3+imu)
    def _feat(v):
        return v[:DIM] if DIM else v
    clfs = {}
    for a, b in HARD:
        mti = [(k, _feat(v)) for k, v in FE.items() if int(k.split('/')[0]) in (a, b)]
        if len(mti) < 12:
            print(f"pair({a},{b}) 样本不足 {len(mti)}", flush=True)
            continue
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
        clf.fit(np.array([v for _, v in mti]),
                [1 if int(k.split('/')[0]) == a else 0 for k, _ in mti])
        clfs[(a, b)] = clf
        print(f"pair({a},{b}): fitted on {len(mti)} train clips (dim={len(mti[0][1])})", flush=True)

    # 2) test v3+imu 特征(缓存)
    test_root = Path("data/Testing/data/small_model_track_test")
    tcache = Path(args.test_cache)
    if tcache.exists():
        TFE = pickle.load(open(tcache, "rb"))
    else:
        TFE = {}
        for d in sorted(test_root.iterdir()):
            if not d.is_dir() or not d.name.startswith("SM_test_"):
                continue
            c = Clip()
            c.pred_dir = d / "Skeleton" / "predictions"
            c.imu_dir = d / "IMU"
            k = load_kp(c)
            if k is None:
                continue
            TFE[d.name] = np.concatenate([feats_v3(k), imu_traj(c.imu_dir)]).astype(np.float32)
        pickle.dump(TFE, open(tcache, "wb"))
    print(f"test v3 feats: {len(TFE)}", flush=True)

    # 3) anchor soft
    anchor = pickle.load(open(args.anchor, "rb"))
    test_df = pd.read_csv(Path("~/Multimodal/data/Testing/test_file/test.csv").expanduser())
    preds, overrides = [], 0
    for pth in test_df["path"]:
        m = re.search(r"(SM_test_\d+)", str(pth))
        if not m or m.group(1) not in anchor:
            preds.append(0)
            continue
        cid = m.group(1)
        soft = np.asarray(anchor[cid], dtype=float)
        soft = soft / soft.sum()          # 归一化(与 gate_fold_check 同口径；anchor 是两成员 soft 和)
        pred = int(np.argmax(soft))
        srt = np.sort(soft)[::-1]
        margin = float(srt[0] - srt[1])
        conf = float(srt[0])
        if pred in HPID and margin < args.tau and cid in TFE \
                and args.conf_lo <= conf <= args.conf_hi:
            xf = _feat(TFE[cid])
            # 若指定 keep 集，仅当该样本所在对在 keep 内才换头
            pair_of = None
            for a, b in HARD:
                if pred in (a, b):
                    pair_of = tuple(sorted((a, b)))
                    break
            if KEEP and pair_of is not None and pair_of not in KEEP:
                preds.append(pred)
                continue
            for a, b in HARD:
                if pred not in (a, b):
                    continue
                clf = clfs.get((a, b))
                if clf is None:
                    break
                cls = list(clf.classes_)
                ia, ib = cls.index(1), cls.index(0)   # ia=>a 侧
                pa, pb = clf.predict_proba([xf])[0][ia], clf.predict_proba([xf])[0][ib]
                g = a if pa >= pb else b
                if g != pred:
                    pred = g
                    overrides += 1
                break
        preds.append(pred)
    # 写 CSV(与 submit 链同样格式)
    test_df["prediction"] = preds
    out = Path(args.out).expanduser()
    test_df[["path", "prediction"]].to_csv(out, index=False)
    print(f"override={overrides} clips; total={len(preds)}; -> {out}", flush=True)
    hist = {c: int((np.array(preds) == c).sum()) for c in sorted(set(preds))}
    print("pred 分布(前10):", dict(list(sorted(hist.items(), key=lambda x: -x[1]))[:10]))
    # 对比原 anchor argmax 改动了多少
    base = []
    for pth in test_df["path"]:
        m = re.search(r"(SM_test_\d+)", str(pth))
        base.append(int(np.argmax(anchor[m.group(1)])) if m and m.group(1) in anchor else 0)
    diff = int(sum(a != b for a, b in zip(base, preds)))
    print(f"与 anchor-argmax 相比改动 {diff} clips", flush=True)


if __name__ == "__main__":
    main()
