#!/usr/bin/env python3
"""3 路融合折级判定（fold0）：main + th_R2+1D(s42,16f) + th_nf32 + th_gru prob-avg。

读 outputs/oof/{main_oof,thermal_oof,th_nf32_oof_fold0,th_gru_oof_fold0}.pkl 的 fold0。
判定: 新增 th 成员是否在共同 clip 上把 R0(main+th16) acc 拉高。
"""
import pickle

import numpy as np

FOLD = 0


def loadf(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    m = loadf("outputs/oof/main_oof.pkl")[FOLD]
    t16 = loadf("outputs/oof/thermal_oof.pkl")[FOLD]
    try:
        nf32 = loadf("outputs/oof/th_nf32_oof_fold0.pkl")[FOLD]
    except Exception:
        nf32 = None
    try:
        gru = loadf("outputs/oof/th_gru_oof_fold0.pkl")[FOLD]
    except Exception:
        gru = None

    streams = {"main": m, "th16": t16}
    if nf32: streams["th_nf32"] = nf32
    if gru: streams["th_gru"] = gru
    keys = set.intersection(*[set(v) for v in streams.values()])
    ks = sorted(keys)
    labels = np.array([int(k.split("/")[0]) for k in ks])
    P = {name: softmax(np.stack([v[k] for k in ks])) for name, v in streams.items()}
    print(f"共同 clip={len(ks)} 流={list(streams)}")

    def acc(combo):
        p = np.sum([P[x] for x in combo], 0)
        return (p.argmax(-1) == labels).mean()

    for combo in [["main", "th16"],
                  ["main", "th_nf32"], ["main", "th_gru"],
                  ["main", "th16", "th_nf32"],
                  ["main", "th16", "th_gru"],
                  ["main", "th16", "th_nf32", "th_gru"],
                  ["main", "th_nf32", "th_gru"]]:
        if all(c in P for c in combo):
            print(f"  prob-avg {combo} -> acc={acc(combo):.4f}")


if __name__ == "__main__":
    main()
