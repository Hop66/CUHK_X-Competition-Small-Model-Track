#!/usr/bin/env python3
"""推理侧高阶扫描(方向4, 纯 OOF CPU, 不占 GPU):
main_oof + thermal_oof(3折) 上扫:
  1. arith vs geomean
  2. 温度缩放 T on each member softmax
  3. per-class 权重(用折 oracle) → 评估"先验提出难类"
  4. 置信缝合(conf-gate pick max-conf)
对照: fix0.5 prob_avg(已知折叠 0.6777) → 找 fold 上 ≥+0.5pt 的形态。
"""
import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

OOF = Path("outputs/oof")


def load_modal(name):
    out = {}
    if name == "main":
        d = pickle.load(open(OOF / "main_oof.pkl", "rb"))
        for f, kv in d.items():
            out[f] = kv
    elif name == "thermal":
        d = pickle.load(open(OOF / "thermal_oof.pkl", "rb"))
        for f, kv in d.items():
            out[f] = kv
    elif name == "th_nf32":
        for f in range(3):
            p = OOF / f"th_nf32_oof_fold{f}.pkl"
            out[f] = pickle.load(open(p, "rb")) if p.exists() else {}
    return out


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def acc_of(probs, y):
    return (probs.argmax(-1) == y).mean()


def main():
    M = load_modal("main")
    T = load_modal("thermal")
    TN = load_modal("th_nf32")
    print(f"modals: main folds={sorted(M)} thermal={sorted(T)} th_nf32={sorted(TN)}")
    base_accs = {}
    for fold in range(3):
        mk = M[fold]; tk = T[fold]
        keys = [k for k in mk if k in tk]
        y = np.array([int(k.split("/")[0]) for k in keys])
        pM = softmax(np.stack([mk[k] for k in keys]))
        pT = softmax(np.stack([tk[k] for k in keys]))
        base = 0.5 * (pM + pT)
        a = (base.argmax(-1) == y).mean()
        base_accs[fold] = (keys, y, pM, pT, base, a)
        print(f"  fold{fold}: main={acc_of(pM,y):.4f} thermal={acc_of(pT,y):.4f} "
              f"0.5avg={a:.4f}")

    # 1) 温度扫描(全局 T)
    print("\n== 温度 T (均作用于两成员 softmax 后，仍 0.5avg) ==")
    for T_ in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        accs = []
        for fold in range(3):
            keys, y, pM, pT, base, a = base_accs[fold]
            pM2 = softmax(np.stack([M[fold][k] for k in keys]), T_)
            pT2 = softmax(np.stack([T[fold][k] for k in keys]), T_)
            fused = 0.5 * (pM2 + pT2)
            accs.append((fused.argmax(-1) == y).mean())
        print(f"  T={T_} 3折mean={np.mean(accs):.4f}  (folds {[f'{x:.4f}' for x in accs]})")

    # 2) geomean (log space): 0.5*logpM + 0.5*logpT → argmax
    print("\n== geomean(logit平均) vs arith(prob平均) ==")
    for T_ in [1.0, 2.0]:
        accs = []
        for fold in range(3):
            keys, y, pM, pT, base, a = base_accs[fold]
            lM = np.log(softmax(np.stack([M[fold][k] for k in keys]), T_) + 1e-9)
            lT = np.log(softmax(np.stack([T[fold][k] for k in keys]), T_) + 1e-9)
            fused = lM + lT  # unnormalized, argmax 等效
            accs.append((fused.argmax(-1) == y).mean())
        print(f"  geomean T={T_} 3折mean={np.mean(accs):.4f}  (folds {[f'{x:.4f}' for x in accs]})")

    # 3) per-class 权重(oracle, 用 val acc per class 对成员加权)
    print("\n== per-class 权重(oracle: 用 val 折类别acc 当成员权重) ==")
    accs = []
    for fold in range(3):
        keys, y, pM, pT, base, a = base_accs[fold]
        # 每个类: 用该折 main/th 各自对该类 val acc 决定权重
        wm = np.zeros(40); wt = np.zeros(40)
        for c in range(40):
            sel = y == c
            if sel.sum() == 0:
                wm[c] = wt[c] = 0.5
                continue
            am = (pM[sel].argmax(-1) == y[sel]).mean()
            at = (pT[sel].argmax(-1) == y[sel]).mean()
            wm[c] = am / (am + at + 1e-9); wt[c] = 1 - wm[c]
        w = wm[None, :]  # (1,40) → 广播到 (N,40)
        fused = w * pM + (1 - w) * pT
        accs.append((fused.argmax(-1) == y).mean())
        print(f"    fold{fold} per-class={accs[-1]:.4f} (base {base_accs[fold][5]:.4f})")
    print(f"  per-class 3折mean={np.mean(accs):.4f} vs base 0.5avg mean="
          f"{np.mean([base_accs[f][5] for f in range(3)]):.4f}")

    # 4) 3 成员(main+thermal+th_nf32) 0.33avg, 若 nf32 该 fold 有
    print("\n== main+thermal+th_nf32 三分(若该折 nf32 有 OOF) ==")
    accs = []
    for fold in range(3):
        mk = M[fold]; tk = T[fold]; nk = TN[fold]
        keys = [k for k in mk if k in tk and k in nk]
        if len(keys) < 100:
            print(f"    fold{fold}: nf32 OOF n={len(keys)} 跳过")
            continue
        y = np.array([int(k.split("/")[0]) for k in keys])
        pM = softmax(np.stack([mk[k] for k in keys]))
        pT = softmax(np.stack([tk[k] for k in keys]))
        pN = softmax(np.stack([nk[k] for k in keys]))
        fused = (pM + pT + pN) / 3
        accs.append((fused.argmax(-1) == y).mean())
        # 还测 2成员(main+th) 同 keys 对照
        base = 0.5 * (pM + pT)
        bacc = (base.argmax(-1) == y).mean()
        print(f"    fold{fold}: 2avg={bacc:.4f} 3avg={accs[-1]:.4f} Δ={accs[-1]-bacc:+.4f}")
    if accs:
        print(f"  3成员 mean={np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
