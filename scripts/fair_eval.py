#!/usr/bin/env python3
"""P0 统一公平评估器 (2026-09-11, 见 idea.md §4).

核心原则:
  1. 单变量: 两份 logits/CSV 除被比较的变量外, 协议(seed/crop/frames/flip/quant)完全一致。
  2. 分层报告: overall Acc + per-subject + macro-F1 + worst-subject + mean/std。
  3. locked 优先: 若给了 locked split, 先报开发集(dev inner CV) 后报 locked(最终判据)。
  4. 不作 selection: 本脚本只报数字, 不帮你选超参 (超参必须在 locked 之外选)。

用法:
  python scripts/fair_eval.py --logits pairA_main.pkl pairA_th.pkl --name A \
      --logits pairB_main.pkl pairB_th.pkl --name B [--locked outputs/locked_split.json]
  python scripts/fair_eval.py --csv a.csv --logits b_main.pkl b_th.pkl --name A --name B
"""
import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def sf(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def read_csv_sub(path):
    m = {}
    with open(path) as fh:
        r = csv.reader(fh); next(r, None)
        for row in r:
            if len(row) >= 2:
                m[row[0]] = row[1]
    return m


def fuse_logits(main_d, th_d):
    """prob_avg (与提交同) → {key: pred}"""
    out = {}
    ks = set(main_d) & set(th_d)
    for k in sorted(ks):
        p = sf(np.asarray(main_d[k], float)) + sf(np.asarray(th_d[k], float))
        out[k] = str(int(p.argmax()))
    return out


def subj_of(key):
    # key 形如 '36/user17/5-1-1' 或 'SM_test_0036' 或路径
    parts = key.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("user"):
            return p
        if p.startswith("SM_test_"):
            return "test"
    return "unknown"


def report(pred, gold, name, locked=None):
    acc = np.mean([pred[k] == gold[k] for k in gold if k in pred])
    # per-subject
    per = defaultdict(lambda: [0, 0])
    for k in gold:
        if k not in pred:
            continue
        per[subj_of(k)][1] += 1
        per[subj_of(k)][0] += (pred[k] == gold[k])
    per_acc = {s: c / n for s, (c, n) in per.items() if n}
    worst = min(per_acc.values()) if per_acc else float("nan")
    # macro F1 (40类)
    K = sorted({v for v in gold.values()})
    macro = 0.0
    for c in K:
        tp = sum(1 for k in gold if pred.get(k) == c and gold[k] == c)
        fp = sum(1 for k in gold if pred.get(k) == c and gold[k] != c)
        fn = sum(1 for k in gold if gold[k] == c and pred.get(k) != c)
        denom = (2 * tp + fp + fn)
        macro += (2 * tp / denom) if denom else 0.0
    macro /= max(len(K), 1)

    print(f"[{name}] overall={acc:.4f}  worst-subject={worst:.4f}  "
          f"macroF1={macro:.4f}  per-subject mean={np.mean(list(per_acc.values())):.4f} "
          f"std={np.std(list(per_acc.values())):.4f}")
    if locked:
        ld = set(locked["locked"])
        keys = [k for k in gold if subj_of(k) in ld]
        if keys:
            la = np.mean([pred[k] == gold[k] for k in keys if k in pred])
            print(f"    locked-audit(l={len(ld)}): {la:.4f} (n={len([k for k in keys if k in pred])})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", action="append", nargs="+", default=[], help="pairs: name main.pkl th.pkl")
    ap.add_argument("--csv", action="append", nargs="+", default=[], help="pairs: name file.csv")
    ap.add_argument("--gold", required=True, help="gold pkl: {key: str(label)}")
    ap.add_argument("--locked", default="")
    args = ap.parse_args()

    gold = pickle.load(open(args.gold, "rb"))
    name2pred = {}
    for grp in args.logits:
        name, main_p, th_p = grp[0], grp[1], grp[2]
        name2pred[name] = fuse_logits(pickle.load(open(main_p, "rb")), pickle.load(open(th_p, "rb")))
    for grp in args.csv:
        name, csv_p = grp[0], grp[1]
        name2pred[name] = read_csv_sub(csv_p)
        # 让 key 与 gold 对齐(截掉路径前缀)
        name2pred[name] = {k.replace("small_model_track_test/", "").rstrip("/"): v
                           for k, v in name2pred[name].items()}

    locked = json.loads(open(args.locked).read()) if args.locked else None
    res = {}
    for name, pred in name2pred.items():
        res[name] = report(pred, gold, name, locked)
    print("\n(注: 数字仅供评估, 不参与超参选择; 超参选择必须只在 dev/inner CV)")


if __name__ == "__main__":
    main()
