#!/usr/bin/env python3
"""融合 α 扫描 + 尖锐度对比 —— 判断难对训练是否改变塔的 softmax 尺度。

用法: python scripts/alpha_scan_fold.py --fold 0
  [R 基线双塔] 读 main_oof + thermal_oof (flip 提取)
  [A 难对双塔] 读 hpdual logits pkl (由 hardpair_dual_fold.py 导出, 或本轮扫时推理)
产出: 每套(基线/难对)的 平均maxprob 尖锐度 + α 扫描表(α=0.3~0.8)。
判读: 若难对 th 的 平均maxprob 相比基线显著变化 → softmax 尺度变了 → 融合α需重平衡;
      若近似 → prob_avg 仍成立。
注: fold(aug2) α 最优 ≠ test(s42) 最优(铁律), 只作"尺度是否漂移"的诊断, 不直接改锚融合。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

OOF = Path("outputs/oof")


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_member(path, fold):
    return {k: np.asarray(v, float) for k, v in pickle.load(open(path, "rb"))[fold].items()}


def scan(pm, pt, keys, labs, tag):
    print(f"\n[{tag}] 尖锐度: main 平均maxprob={pm.max(1).mean():.3f} "
          f"th={pt.max(1).mean():.3f}", flush=True)
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fu = a * pm + (1 - a) * pt
        acc = (fu.argmax(1) == labs).mean()
        print(f"  α={a} (main权重): acc={acc:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--hp_main", default="")
    ap.add_argument("--hp_th", default="")
    args = ap.parse_args()

    bm = load_member(OOF / "main_oof.pkl", args.fold)
    bt = load_member(OOF / "thermal_oof.pkl", args.fold)
    keys = [k for k in bm if k in bt]
    labs = np.array([int(k.split("/")[0]) for k in keys])
    pm = np.stack([sf(bm[k]) for k in keys])
    pt = np.stack([sf(bt[k]) for k in keys])
    scan(pm, pt, keys, labs, f"基线双塔 fold{args.fold}")

    if args.hp_main and args.hp_th:
        hm = load_member(args.hp_main, args.fold) if Path(args.hp_main).exists() else {}
        ht = load_member(args.hp_th, args.fold) if Path(args.hp_th).exists() else {}
        keys2 = [k for k in keys if k in hm and k in ht]
        labs2 = np.array([int(k.split("/")[0]) for k in keys2])
        phm = np.stack([sf(hm[k]) for k in keys2])
        pht = np.stack([sf(ht[k]) for k in keys2])
        scan(phm, pht, keys2, labs2, f"难对双塔 fold{args.fold}")


if __name__ == "__main__":
    main()
