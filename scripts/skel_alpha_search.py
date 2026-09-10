#!/usr/bin/env python3
"""骨架「低权重 α 插值」闭环验证 (CPU, fold0)。

输入: outputs/oof/{skel_fold0,main_oof,thermal_oof}.pkl (dict: key->logits[40])
扫描 α ∈ [0,0.3]: pred = argmax( (1-α)·prob_avg(main,th) + α·softmax(skel) )
回答: 骨架以低权重加入是否有一点点正边际; 若无 → 骨架线彻底关。
"""
import pickle
from pathlib import Path

import numpy as np


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


OOF = Path("outputs/oof")
skel = load(OOF / "skel_fold0.pkl")
# main/thermal OOF 顶层按 fold 组织 {0: {key: logits}, 1:..., 2:...} → 取 fold0
_main = load(OOF / "main_oof.pkl")
_therm = load(OOF / "thermal_oof.pkl")
main = _main[0] if isinstance(_main, dict) and 0 in _main else _main
therm = _therm[0] if isinstance(_therm, dict) and 0 in _therm else _therm

keys = [k for k in main if k in therm and k in skel]
print(f"对齐 samples: {len(keys)}  (main/thermal/skel 交集)")

M = np.stack([main[k] for k in keys])
T = np.stack([therm[k] for k in keys])
S = np.stack([skel[k] for k in keys])
y = np.array([int(k.split("/")[0]) for k in keys])

pM = softmax(M)
pT = softmax(T)
pS = softmax(S)
mt = 0.5 * (pM + pT)  # 双模态 prob-avg 基线(与 0.75 链同法)
base_acc = (mt.argmax(-1) == y).mean()
print(f"\n基线 main+th prob-avg acc = {base_acc:.4f}")
print(f"骨架 standalone acc         = {(pS.argmax(-1) == y).mean():.4f}")
print(f"\n{'alpha':>6} | {'main+th+skel':>14} | delta vs base")
best = (0.0, base_acc)
for a in np.arange(0.0, 0.301, 0.025):
    pred = ((1 - a) * mt + a * pS).argmax(-1)
    acc = (pred == y).mean()
    flag = ""
    if acc > base_acc + 1e-4:
        flag = "  <== +"
        if acc > best[1]:
            best = (a, acc)
    print(f"{a:6.3f} | {acc:14.4f} | {acc - base_acc:+.4f}{flag}")
print(f"\n最优 α={best[0]:.3f} acc={best[1]:.4f} (Δ={best[1]-base_acc:+.4f})")
print("结论:", "骨架低权有正边际, 值得进链" if best[1] > base_acc + 0.003 else
      "骨架低权无正边际(或 ≤0.3pt), 骨架线正式关闭")
