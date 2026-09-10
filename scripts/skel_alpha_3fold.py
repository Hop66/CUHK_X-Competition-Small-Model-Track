#!/usr/bin/env python3
"""骨架低权 α: 3 折稳健性验证 + KD vs native (CPU)。

每折: main_oof[i] + thermal_oof[i] 基准 vs +skel_fold{i} α=0.075
f0 另测 KD(teacher=main+th) 的融合, 对比 native。
"""
import pickle
import sys
from pathlib import Path

import numpy as np

OOF = Path("outputs/oof")


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_main(i):
    d = pickle.load(open(OOF / "main_oof.pkl", "rb"))
    return d[i] if isinstance(d, dict) and i in d else d


def load_th(i):
    d = pickle.load(open(OOF / "thermal_oof.pkl", "rb"))
    return d[i] if isinstance(d, dict) and i in d else d


def skel_alpha(fold, alpha=0.075, skel_file=None):
    M = load_main(fold)
    T = load_th(fold)
    S = pickle.load(open(OOF / (skel_file or f"skel_fold{fold}.pkl"), "rb"))
    keys = [k for k in M if k in T and k in S]
    pM = softmax(np.stack([M[k] for k in keys]))
    pT = softmax(np.stack([T[k] for k in keys]))
    pS = softmax(np.stack([S[k] for k in keys]))
    y = np.array([int(k.split("/")[0]) for k in keys])
    mt = 0.5 * (pM + pT)
    a = (mt.argmax(-1) == y).mean()
    fused = (1 - alpha) * mt + alpha * pS
    f = (fused.argmax(-1) == y).mean()
    return len(keys), a, f, f - a


print("== native 骨架 3折 fixed α=0.075 ==")
res = []
for i in [0, 1, 2]:
    n, a, f, d = skel_alpha(i)
    res.append((a, f, d))
    print(f"  fold{i}: n={n:d}  main+th={a:.4f}  +skel={f:.4f}  Δ={d:+.4f}")
if res:
    ga = np.mean([r[0] for r in res]); gf = np.mean([r[1] for r in res])
    gd = np.mean([r[2] for r in res]); sd = np.std([r[2] for r in res])
    print(f"\n 3折: 基线 mean={ga:.4f}  +skel mean={gf:.4f}  Δmean={gd:+.4f} (±{sd:.4f})")

print("\n== KD fold0 vs native fold0 (standalone 0.5318 vs 0.4984) 融合对比 ==")
n, a, f, d = skel_alpha(0, skel_file="skel_kd_fold0.pkl")
print(f"  KD fold0: base={a:.4f} +KDskel={f:.4f} Δ={d:+.4f} (native was +1.65pt @0.6740)")
