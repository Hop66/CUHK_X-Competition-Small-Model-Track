#!/usr/bin/env python3
"""dual_fusion fold0 挖掘第二步：与 main/thermal OOF 对齐后做 fold0 集成判定。

用法: python scripts/compare_dual_fusion_fold0.py
判定:
  - dual 单独 vs main/thermal 单流 fold0
  - 2流 & 3流 prob-avg 集成 vs R(1m+1th) 同clip集 fold0
"""
import pickle
from pathlib import Path

import numpy as np

OOF_DIR = Path("outputs/oof")
FOLD = 0


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def per_fold(d, f):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and str(f) in d:
            out = d[f]
            break
        out = v  # single-fold pkl: {key: logits}
    return out


def acc_of(logits, labels):
    return float((np.asarray(logits).argmax(-1) == np.asarray(labels)).mean())


def main():
    main_oof = load(OOF_DIR / "main_oof.pkl")
    th_oof = load(OOF_DIR / "thermal_oof.pkl")
    du = load(OOF_DIR / "dualfusion_oof_fold0.pkl")

    m0 = main_oof[FOLD] if isinstance(main_oof, dict) and FOLD in main_oof else main_oof
    t0 = th_oof[FOLD] if isinstance(th_oof, dict) and FOLD in th_oof else th_oof
    d0 = du[FOLD] if isinstance(du, dict) and FOLD in du else du

    common_mt = set(m0) & set(t0)
    common_all = common_mt & set(d0)
    print(f"fold{FOLD}: main={len(m0)} th={len(t0)} dual={len(d0)} 共同(m,t)={len(common_mt)} 三流共同={len(common_all)}")

    def build(srcs):
        keys = set.intersection(*[set(s) for s in srcs])
        mats = []
        for s in srcs:
            mats.append(np.stack([s[k] for k in sorted(keys)]))
        return [m for m in mats], sorted(keys)

    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    # 1) 单流 (main/th/dual) —— 各自自验证(即每流自己 fold 内 acc)
    for name, s in [("main", m0), ("thermal", t0), ("dual_fusion", d0)]:
        keys = sorted(s)
        lg = np.stack([s[k] for k in keys])
        lbl = np.array([int(k.split("/")[0]) for k in keys])
        print(f"单流 {name} acc = {acc_of(lg, lbl):.4f}  (n={len(keys)})")

    # 2) 三流共同的 clip 上：R(1m+1th) vs 2m1th(+dual) vs 3流
    msub = {k: m0[k] for k in common_all}
    tsub = {k: t0[k] for k in common_all}
    dsub = {k: d0[k] for k in common_all}
    msub2 = {k: main_oof["fold0_aug1"][k] if isinstance(main_oof, dict) and "fold0_aug1" in main_oof else m0[k] for k in common_all}
    keys = sorted(common_all)
    labels = np.array([int(k.split("/")[0]) for k in keys])
    Mm = np.stack([msub2[k] for k in keys])
    Mt = np.stack([tsub[k] for k in keys])
    Md = np.stack([dsub[k] for k in keys])
    Ms = np.stack([msub[k] for k in keys]) if "fold0_aug1" not in (main_oof if isinstance(main_oof, dict) else {}) else None

    combos = {
        "R(1main_aug2+1th)": softmax(Mm) + softmax(Mt),
        "R'(1main_base+1th)": (softmax(Ms) if Ms is not None else softmax(Mm)) + softmax(Mt),
        "A(2main+1th)": softmax(Mm) + (softmax(Ms) if Ms is not None else softmax(Mm)) + softmax(Mt),
        "C(2main+2th:=+dual)": softmax(Mm) + (softmax(Ms) if Ms is not None else softmax(Mm)) + softmax(Mt) + softmax(Md),
        "dual+1th": softmax(Md) + softmax(Mt),
        "dual+1main": softmax(Md) + softmax(Mm),
        "D(all3)": softmax(Md) + softmax(Mm) + softmax(Mt),
    }
    print(f"\n=== 共同 {len(common_all)} clip 的 fold{FOLD} 集成判定 ===")
    for name, p in combos.items():
        s = p.argmax(-1)
        print(f"{name}: acc={ (s==labels).mean():.4f}")


if __name__ == "__main__":
    main()
