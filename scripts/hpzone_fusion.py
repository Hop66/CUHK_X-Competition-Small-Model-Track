#!/usr/bin/env python3
"""方向2: 难对区定向融合权重 —— 推理侧零成本, 复用难对表。

发现: fold 难对区内 main 单模 > 融合(0.5) > th 单模 (fold0: 0.589/0.560/0.515)
  → 现有 prob_avg 0.5/0.5 在难对区被 th 噪声拖低; 难对区应抬高 main 权重。
做法: 输出 = (1-mask)*fuse0.5 + mask*fuse_alpha_hp   (mask=clip 落难对区)
  alpha_hp=main 权重; 非难对区保持 0.5/0.5。
用法:
  python scripts/hpzone_fusion.py --main <main_test.pkl> --th <th_test.pkl> \
      --alpha_hp 0.7 --out sub_hpzone.csv
  (与 sub_chain_nf32_flip.csv 比 flips)
判据(fold): 难对区 α≥0.6 优于 0.5(3折已扫); test 需锚同源 logits 才可信。
"""
import argparse
import csv
import pickle

import numpy as np

H11 = [(0, 1), (6, 37), (7, 37), (8, 9), (8, 10), (8, 15),
       (8, 18), (11, 14), (17, 18), (18, 20), (38, 39)]
HPID = {a for p in H11 for a in p}
ANCHOR = "outputs/sub_chain_nf32_flip.csv"


def sf(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--th", required=True)
    ap.add_argument("--alpha_hp", type=float, default=0.7,
                    help="难对区 main 权重(>0.5=main主导); 非难对区保持0.5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    main_d = pickle.load(open(args.main, "rb"))
    th_d = pickle.load(open(args.th, "rb"))
    assert set(main_d.keys()) == set(th_d.keys()), "main/th test keys 不一致"
    keys = sorted(main_d.keys())

    rows = []
    for k in keys:
        pm = sf(np.asarray(main_d[k], float))
        pt = sf(np.asarray(th_d[k], float))
        # 落难对区判定: main+th 融合0.5 的预测是否落在难对对的任一类
        fuse = 0.5 * pm + 0.5 * pt
        pred = int(fuse.argmax())
        in_hp = pred in HPID
        a = args.alpha_hp if in_hp else 0.5
        fu = a * pm + (1 - a) * pt
        rows.append((k, str(int(fu.argmax()))))

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Id", "Expected"])
        w.writerows(rows)

    # 对比锚
    try:
        a = {}
        with open(ANCHOR) as fh:
            r = csv.reader(fh); next(r, None)
            for row in r:
                if len(row) >= 2:
                    a[row[0]] = row[1]
        diff = sum(1 for k, v in rows if a.get(k) and a[k] != v)
        print(f"[{args.out}] 难对区α={args.alpha_hp} vs 锚: flips={diff}/405 "
              f"去重类={len(set(v for _, v in rows))}", flush=True)
    except FileNotFoundError:
        print(f"[{args.out}] 锚文件缺失, 跳过对比 (flips 数不可得)", flush=True)


if __name__ == "__main__":
    main()
