#!/usr/bin/env python3
"""双塔难对 full 快照选点 —— 用锚同源 test 概率对比太难, 改用折内证据+test flip 监督。

因 full 无 val, 无法真早停; 折内 fold0 已标定: main/th 难对塔最佳 ~ep43-45,
80ep 必然过拟合(折内 th 60ep 已掉1.6pt)。此脚本给出"用哪些快照做锚同源 test 链"
的清单, 供 ensemble_inference 逐快照生成 CSV 后人工/脚本对比 flips 选点。

用法(在 full 训练完成后):
  1) 对 outputs/dual_hp_full/{main,th}/*_ep{40,50,60,70,80}.pth 逐快照组 dual 链
  2) 每链输出 sub_hpdual_ep{ep}.csv
  3) 与锚 sub_chain_nf32_flip.csv 比 flips 数 + 去重类数
此脚本仅打印待跑命令 + 提供 CSV 对比入口。
"""
import argparse
import csv
import sys
from pathlib import Path

OUT = Path("outputs/dual_hp_full")
ANCHOR = "outputs/sub_chain_nf32_flip.csv"


def read_csv(path):
    m = {}
    with open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                m[row[0]] = row[1]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="打印各快照组链命令")
    ap.add_argument("--ep", type=int, default=0, help="对比指定 ep 的 CSV 与锚")
    ap.add_argument("--csv", default="", help="直接指定待比 CSV")
    args = ap.parse_args()

    if args.list:
        for m in ("main", "th"):
            d = OUT / m
            snaps = sorted(d.glob("*_ep*.pth"))
            print(f"[{m}] 快照: {[s.stem for s in snaps]}")
        print("\n组链命令模板(每 ep):")
        for ep in [40, 50, 60, 70, 80]:
            main_ck = OUT / "main" / f"r2plus1d34_depthir_full_seed42_ep{ep}.pth"
            th_ck = OUT / "th" / f"r2plus1d34_thermal_full_seed42_ep{ep}.pth"
            print(f"  # ep{ep}")
            print(f"  python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \\")
            print(f"    --main {main_ck} --thermal {th_ck} "
                  f"--num_frames 16 --thermal_frames 32 "
                  f"--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \\")
            print(f"    --output outputs/sub_hpdual_ep{ep}.csv")
        return

    cand = args.csv or (f"outputs/sub_hpdual_ep{args.ep}.csv" if args.ep else "")
    if not cand:
        print("需 --ep N 或 --csv path"); return
    if not Path(cand).exists():
        # 可能是快照没到 ep: 提示
        print(f"⚠️ {cand} 不存在"); return
    a = read_csv(ANCHOR)
    b = read_csv(cand)
    diff = sum(1 for k in a if k in b and a[k] != b[k])
    print(f"{cand}: 锚 0.75124 基线 flips={diff}/405  去重类数={len(set(b.values()))}")
    # 列出 flip 明细
    from collections import Counter
    fl = [(k, a[k], b[k]) for k in a if k in b and a[k] != b[k]]
    cc = Counter((a[k], b[k]) for _, a_, b_ in [(k, a[k], b[k]) for k, _ in fl])
    print("flip 类改动top:", cc.most_common(8))


if __name__ == "__main__":
    main()
