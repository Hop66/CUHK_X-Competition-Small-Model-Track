#!/usr/bin/env python3
"""类级分析: A) IMU 融合增益按类  B) Radar 覆盖/质量按类（用户指名要具体讨论，不看总体占比）。"""
import pickle
from pathlib import Path

import numpy as np

OOF = Path("outputs/oof")


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def lf(p, fold):
    d = pickle.load(open(p, "rb"))
    return d[fold] if isinstance(d, dict) and fold in d else d


def class_names():
    name = {}
    for ad in sorted(Path("data/Training/HAR/IMU").iterdir()):
        if ad.is_dir():
            name[int(ad.name.split("_")[0])] = ad.name
    return name


def main():
    name = class_names()

    # ---------- A. IMU 类级融合增益 ----------
    per = {}
    for fold in [0, 1, 2]:
        km = lf(OOF / "main_oof.pkl", fold); kt = lf(OOF / "thermal_oof.pkl", fold)
        kc = lf(OOF / f"imu_fold{fold}.pkl", fold); kg = lf(OOF / f"imu_gru_fold{fold}.pkl", fold)
        ks = [k for k in km if k in kt and k in kc and k in kg]
        M = sf(np.stack([km[k] for k in ks])); T = sf(np.stack([kt[k] for k in ks]))
        C = sf(np.stack([kc[k] for k in ks])); G = sf(np.stack([kg[k] for k in ks]))
        y = np.array([int(k.split("/")[0]) for k in ks])
        mt = 0.5 * (M + T); P = 0.7 * mt + 0.15 * C + 0.15 * G
        b = mt.argmax(-1) == y; p = P.argmax(-1) == y
        for j, aid in enumerate(y):
            per.setdefault(int(aid), [0, 0])
            per[int(aid)][0] += int(p[j]) - int(b[j])
            per[int(aid)][1] += int(p[j] != b[j])
    rows = sorted(((aid, v[0], v[1]) for aid, v in per.items()), key=lambda r: -r[1])
    print("=== A. IMU 融合按类(main+th vs +双IMU α0.3, 3折) ===")
    print(f"{'cls':>3} {'action':<28} {'Δcorr':>6} {'changed':>8}")
    for aid, d, ch in rows:
        flag = " *" if d >= 3 else (" !" if d <= -3 else "  ")
        print(f"{aid:>3} {name.get(aid,'?'):<28} {d:>6} {ch:>8}{flag}")
    print("净变化:", int(sum(r[1] for r in rows)), "总翻动:", int(sum(r[2] for r in rows)))

    # ---------- B. Radar 类级覆盖/质量 ----------
    print("\n=== B. Radar 类级覆盖(空csv/非空/平均行数) ===")
    base = Path("data/Training/HAR/Radar")
    radar_per = {}
    for ad in sorted(base.iterdir()):
        if not ad.is_dir():
            continue
        aid = int(ad.name.split("_")[0])
        csvs = sorted(ad.rglob("radar_output_*.csv"))
        n_empty = 0; n_nonempty = 0; rows_sum = 0
        for f in csvs:
            sz = f.stat().st_size
            if sz == 0:
                n_empty += 1
            else:
                with open(f, "rb") as fh:
                    fh.readline()
                    n_rows = sum(1 for _ in fh)
                rows_sum += n_rows
                n_nonempty += 1
        radar_per[aid] = (len(csvs), n_empty, n_nonempty,
                          (rows_sum / n_nonempty) if n_nonempty else 0)
    print(f"{'cls':>3} {'action':<28} {'csv':>5} {'empty':>6} {'nonempty':>9} {'avg_rows':>8}")
    for aid in sorted(radar_per):
        n, e, ne, ar = radar_per[aid]
        flag = " (全空!)" if ne == 0 else ""
        print(f"{aid:>3} {name.get(aid,'?'):<28} {n:>5} {e:>6} {ne:>9} {ar:>8.0f}{flag}")
    ntot = sum(v[0] for v in radar_per.values()); netot = sum(v[2] for v in radar_per.values())
    print(f"总计: csv={ntot} 非空={netot} ({netot/ntot:.1%})")


if __name__ == "__main__":
    main()
