#!/usr/bin/env python3
"""难对区判别力对比（回答: BiGRU/神经骨架 对难对是否无意义?）
在 HARD 5 对上，比较各成员 logits 在"2 候选类内 argmax"的准确率:
  main / thermal / 神经骨架(skel_fold) / IMU / 手工特征v3+IMU(参考 twin_v3 62-65%)
clip key: {action_id}/{subject}/{sample} → label=action_id
判读: 若 skel(神经) 难对二分类 acc ≤ main 或≈50% → BiGRU 门控对难对无补益;
      若显著>main → 门控有价值。
用法: python scripts/hardpair_discrim.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

HARD = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
OOF = Path("outputs/oof")


def argmax_acc(logits, keys, labels, pair):
    """2 候选类内 argmax 准确率。"""
    a, b = pair
    idx = [i for i, y in enumerate(labels) if y in (a, b)]
    if not idx:
        return float("nan"), 0
    sub = np.stack([logits[i] for i in idx])
    labs = np.array([labels[i] for i in idx])
    sel = sub[:, [a, b]]
    pred = np.where(sel[:, 0] > sel[:, 1], a, b)
    return (pred == labs).mean(), len(idx)


def load_oof(path, fold=None, flat=False):
    d = pickle.load(open(path, "rb"))
    if flat:
        return d
    return d.get(fold, {}) if isinstance(d, dict) and fold is not None else d


def main():
    # 收集所有 fold 的 OOF key->logits（key: action/subject/sample）
    members = {"main": {}, "thermal": {}, "skel_neur": {}, "imu": {}}
    for fold in range(3):
        for name, path in [("main", OOF / "main_oof.pkl"),
                           ("thermal", OOF / "thermal_oof.pkl")]:
            d = load_oof(path, fold)
            for k, v in d.items():
                members[name][k] = v
        for name, path in [("skel_neur", OOF / f"skel_fold{fold}.pkl"),
                           ("imu", OOF / f"imu_fold{fold}.pkl")]:
            d = load_oof(path, flat=True)
            for k, v in d.items():
                members[name][k] = v

    print(f"{'成员':<11} | " + " | ".join(
        f"({a},{b})" for a, b in HARD) + " | 难对整体")
    for name in ["main", "thermal", "skel_neur", "imu"]:
        logits = members[name]
        ks = list(logits.keys())
        labs = np.array([int(k.split("/")[0]) for k in ks])
        row = []
        hsel = []
        for (a, b) in HARD:
            sel = [i for i, k in enumerate(ks) if labs[i] in (a, b)]
            hsel += sel
            acc, n = float("nan"), 0
            if sel:
                sub = np.stack([logits[ks[i]] for i in sel])
                la = labs[sel]
                p = np.where(sub[:, [a, b]][:, 0] > sub[:, [a, b]][:, 1], a, b)
                acc, n = (p == la).mean(), len(sel)
            row.append(f"{acc:.2f}(n={n})")
        hsel = sorted(set(hsel))
        hacc, hn = float("nan"), 0
        if hsel:
            sub = np.stack([logits[ks[i]] for i in hsel])
            la = labs[hsel]
            p = []
            for pos, lab in enumerate(la):
                for (a, b) in HARD:
                    if lab in (a, b):
                        p.append(a if sub[pos][a] > sub[pos][b] else b)
                        break
            hacc, hn = (np.array(p) == la).mean(), len(hsel)
        print(f"{name:<11} | " + " | ".join(row) + f" | {hacc:.3f}(n={hn})")
    print("\n注: skel_neur=神经骨架OOF(BiGRU/MB), 手工v3+IMU参考=62-65%(难对二分类), main也弱")
    print("判读: skel_neur难对二分类若≈50%或≤main → BiGRU门控对难对无补益")


if __name__ == "__main__":
    main()
