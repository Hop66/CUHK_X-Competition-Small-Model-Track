#!/usr/bin/env python3
"""骨架融合伙伴诊断 (CPU, fold0):
1) 骨架对 main 还是 thermal 增益大 (两两 α vs 三路)
2) per-class 骨架纠错统计 (为「类级/样本级加权」Q2 提供证据)
3) 固定 α vs 骨架置信门控(per-sample α) 对比
"""
import pickle
from pathlib import Path

import numpy as np

OOF = Path("outputs/oof")


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_fold0(p):
    d = pickle.load(open(p, "rb"))
    return d[0] if isinstance(d, dict) and 0 in d else d


main = load_fold0(OOF / "main_oof.pkl")
therm = load_fold0(OOF / "thermal_oof.pkl")
skel = pickle.load(open(OOF / "skel_fold0.pkl", "rb"))

keys = [k for k in main if k in therm and k in skel]
M = softmax(np.stack([main[k] for k in keys]))
T = softmax(np.stack([therm[k] for k in keys]))
S = softmax(np.stack([skel[k] for k in keys]))
y = np.array([int(k.split("/")[0]) for k in keys])
n = len(keys)


def acc(P):
    return (P.argmax(-1) == y).mean()


def join(*Ps, alpha=0.075, w=None):
    base = sum(Ps[:-1]) / len(Ps[:-1])
    if w is None:
        w = np.full(len(Ps[0]), alpha)
    return (1 - w[:, None]) * base + w[:, None] * Ps[-1]


mt = 0.5 * (M + T)
print(f"n={n}  main={acc(M):.4f} thermal={acc(T):.4f} skel={acc(S):.4f} main+th={acc(mt):.4f}")

print("\n=== Q4: 骨架对谁好 (两两 α=0.075) ===")
for name, pair in [("main+skel", (M, S)), ("thermal+skel", (T, S)), ("m+t+skel", (mt, S))]:
    print(f"  {name:<12} {acc(join(*pair)):.4f}")

# 骨架 vs main/th 的纠错对象
print("\n=== Q2: 谁被骨架纠错 (固定 α=0.075 三路 vs main+th) ===")
pred_mt = mt.argmax(-1)
pred_3 = join(mt, S).argmax(-1)
flip = (pred_3 != pred_mt)
stay = (pred_3 == pred_mt)
print(f"  三路相对 main+th: 改动样本 {flip.sum()}/{n} ({flip.mean()*100:.1f}%)")
fix = (pred_mt != y) & (pred_3 == y)     # 由错变对
brk = (pred_mt == y) & (pred_3 != y)     # 由对变错
print(f"  纠错 fix={fix.sum()}  破坏 brk={brk.sum()}  净Δ={fix.sum()-brk.sum()}")

# per-class 纠错
print("\n=== per-class: main+th 错但骨架对 (骨架强项类) ===")
mt_err = pred_mt != y
skel_ok = S.argmax(-1) == y
per_class = {}
for c in np.unique(y):
    m = (y == c)
    if m.sum() > 0:
        per_class[int(c)] = ((m & mt_err & skel_ok).sum(), int(m.sum()))
    if False:
        pass
srt = sorted(per_class.items(), key=lambda kv: -kv[1][0])
for c, (res, tot) in srt[:10]:
    if res > 0:
        print(f"  class {c:2d}: 骨架挽救 {res:3d}/{tot:3d}")
print(f"  (其余类挽救=0 或 <阈值略过; 总挽救 {sum(v[0] for v in per_class.values())})")

# 置信门控: 骨架高置信样本给高 α
print("\n=== Q2-b: 固定 α vs per-sample 骨架置信门控 ===")
conf = S.max(-1)
for alpha0 in [0.05, 0.075, 0.10]:
    base_acc = acc(join(mt, S, alpha=alpha0))
    # 门控: 骨架 top-50% 置信 → α=alpha0*1.5, 低置信 → α=alpha0*0.5
    thr = np.median(conf)
    w = np.where(conf >= thr, alpha0 * 1.5, alpha0 * 0.5)
    gated = (1 - w[:, None]) * mt + w[:, None] * S
    print(f"  α0={alpha0:.3f}: 固定 {base_acc:.4f}  | 门控 {acc(gated):.4f} (Δ{acc(gated)-base_acc:+.4f})")
