#!/usr/bin/env python3
"""折级网格搜主/热融合权重(fold OOF 上, flip-only) —— 判断"非等权概率平均"是否优于等权冠军。

用法: python scripts/grid_fusion_weights.py
输出: 各 fold 在(等权 vs 最优搜索权重)下的 acc; 报告一致性。
"""
import pickle

import numpy as np


def softmax(lg):
    lg = np.asarray(lg, np.float32)
    lg = lg - lg.max()
    e = np.exp(lg)
    return e / e.sum()


def probs_by_fold(path):
    d = pickle.load(open(path, "rb"))
    out = {}
    for f, kv in d.items():
        p = {k: softmax(v) for k, v in kv.items()}
        out[f] = p
    return out


def joint(folds_streams, fold):
    """返回 sorted keys + 各方 probs 矩阵。"""
    d = [folds_streams[s][fold] for s in folds_streams]
    common = set(d[0].keys())
    for dd in d[1:]:
        common &= set(dd.keys())
    keys = sorted(common)
    P = [np.stack([dd[k] for k in keys]) for dd in d]  # 每流 [N,40]
    y = np.array([int(k.split("/")[0]) for k in keys])
    return P, y


def acc_from_weights(P, y, w):
    fused = sum(wi * Pi for wi, Pi in zip(w, P))
    return (fused.argmax(1) == y).mean()


def main():
    pm = probs_by_fold("outputs/oof/main_oof.pkl")          # aug2 main
    p1 = probs_by_fold("outputs/oof/main_oof_aug1.pkl")     # aug1 main
    th = probs_by_fold("outputs/oof/thermal_oof.pkl")       # th

    for tag, streams in [("1main+1th", {"m": pm, "t": th}),
                         ("2main+1th", {"m": pm, "m1": p1, "t": th})]:
        print(f"\n=== {tag}: 等权 vs 网格搜权 ===")
        per_eq, per_best = [], []
        for f in range(3):
            P, y = joint(streams, f)
            S = len(P)
            eq = acc_from_weights(P, y, [1.0 / S] * S)
            # 网格: 每个流 w in [0.4..0.8 step .1], 归一化(简化: 只搜第一流权重 w1, 其余均分)
            best_w, best_a = [1.0 / S] * S, eq
            for w0 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                rest = (1.0 - w0) / (S - 1)
                w = [w0] + [rest] * (S - 1)
                a = acc_from_weights(P, y, w)
                if a > best_a:
                    best_a, best_w = a, w
            per_eq.append(eq); per_best.append(best_a)
        print(f"  等权 fold accs={[round(x,4) for x in per_eq]} mean={np.mean(per_eq):.4f}")
        print(f"  最优 fold accs={[round(x,4) for x in per_best]} mean={np.mean(per_best):.4f} Δ={np.mean(per_best)-np.mean(per_eq):+.4f}")


if __name__ == "__main__":
    main()
