#!/usr/bin/env python3
"""混淆矩阵分析（找"语义外难对" + 长尾诊断）——
用 main+thermal OOF 融合在 3 折上的完整错误模式：
1. 所有混淆对 (A,B): 真实=A 但被分到 B（或反之）的样本数
2. 每类 per-class acc / 支持数（长尾）
3. 已知难对(5对)之外的 Top 混淆对
4. 每对二分类内"主链 margin/conf"分布 → 找可 gate 的潜在难对
"""
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OOF = Path("outputs/oof")


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    # 融合 main+th 3折 OOF
    MK = pickle.load(open(OOF / "main_oof.pkl", "rb"))
    TK = pickle.load(open(OOF / "thermal_oof.pkl", "rb"))
    allrk = []
    for f in range(3):
        mk, tk = MK[f], TK[f]
        for k in mk:
            if k in tk:
                soft = sf(mk[k]) + sf(tk[k])
                soft = soft / soft.sum()
                pred = int(np.argmax(soft))
                label = int(k.split("/")[0])
                srt = np.sort(soft)[::-1]
                allrk.append((label, pred, float(srt[0] - srt[1]), float(srt[0]),
                              float(soft[pred] - soft[label]) if pred != label else float("nan")))
    print(f"总样本 n={len(allrk)} acc={(np.mean([1 if l==p else 0 for l,p,_,_,_ in allrk])):.4f}")

    # 1) 每类 per-class acc + 支持数（长尾）
    cls = defaultdict(list)
    for l, p, m, c, _ in allrk:
        cls[l].append(1 if l == p else 0)
    print("\n== per-class acc / 支持数（长尾诊断） ==")
    print(f"{'类':<4}{'n':<6}{'acc':<10}")
    for l in sorted(cls):
        v = cls[l]
        print(f"{l:<4}{len(v):<6}{np.mean(v):.3f}")
    low = [(l, np.mean(v), len(v)) for l, v in cls.items() if np.mean(v) < 0.60]
    print(f"\nacc<0.60 类: {len(low)}个 → {[(l, f'{n}样本', f'{a:.2f}') for l, a, n in low]}")

    # 2) 混淆对（无向）
    conf = Counter()
    conf_dir = Counter()
    for l, p, m, c, d in allrk:
        if l != p:
            pair = tuple(sorted((l, p)))
            conf[pair] += 1
            conf_dir[(l, p)] += 1
    print("\n== Top 混淆对（无向, 错误数≥3） ==")
    HARD = {(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)}
    for pair, n in sorted(conf.items(), key=lambda x: -x[1])[:25]:
        known = "★已知" if pair in HARD else ""
        print(f"  {pair} err={n}  {known}")
    print("\n== 已知难对之外的混淆对 ==")
    for pair, n in sorted(conf.items(), key=lambda x: -x[1]):
        if pair in HARD or n < 3:
            continue
        print(f"  {pair} err={n}")

    # 3) 每混淆对的可 gate 性: 融合 margin 均值（小 margin=可 gate 潜力）
    print("\n== 新难对候选（错误≥3 且 margin 小=融合难分） ==")
    pair_info = defaultdict(list)
    for l, p, m, c, d in allrk:
        if l != p:
            pair_info[tuple(sorted((l, p)))].append((m, c))
    for pair, ms in sorted(pair_info.items(), key=lambda x: -len(x[1])):
        if len(ms) < 3 or pair in HARD:
            continue
        marg = np.mean([x[0] for x in ms]); confm = np.mean([x[1] for x in ms])
        print(f"  {pair} err={len(ms)} margin_mean={marg:.3f} conf_mean={confm:.3f}")


if __name__ == "__main__":
    main()
