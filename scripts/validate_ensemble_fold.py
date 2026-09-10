#!/usr/bin/env python3
"""折级集成验证(泛化 N 流) —— 用 fold OOF 重建候选集成形态, 判"成员/流多样性"是否增益(flip-only 概率平均)。

用法: python scripts/validate_ensemble_fold.py \
  --R 'main_aug2=outputs/oof/main_oof.pkl,th=outputs/oof/thermal_oof.pkl' \
  --A 'main_aug2=...,main_aug1=...,th=...' \
  [--A_name "2main+th"] ...
每个形态: "name=path,path,..."(同权平均)。报告每形态 fold acc + 相对 --基准形态 增益。
"""
import argparse
import pickle

import numpy as np


def softmax(lg):
    lg = np.asarray(lg, np.float32)
    lg = lg - lg.max()
    e = np.exp(lg)
    return e / e.sum()


def eval_fold(paths_fold, fold):
    d = [paths_fold[s][fold] for s in paths_fold]
    common = set(d[0].keys())
    for dd in d[1:]:
        common &= set(dd.keys())
    correct = total = 0
    for k in sorted(common):
        p = np.mean([softmax(dd[k]) for dd in d], axis=0)
        correct += (p.argmax() == int(k.split("/")[0]))
        total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", help="基准形态: name=path,path...")
    ap.add_argument("--A", action="append", default=[], help="候选形态(可多个)")
    ap.add_argument("--A_name", action="append", default=[])
    args = ap.parse_args()

    def load_spec(spec):
        # spec: "main=path_aug2,path_aug1,th=path_th" → {stream: OOF dict}
        streams = {}
        for seg in spec.split(","):
            name, path = seg.split("=")
            streams[name] = pickle.load(open(path, "rb"))
        return streams

    ref = load_spec(args.R)
    results = {}
    for f in range(3):
        key = f"fold{f}"

        def acc(s):
            return eval_fold(s, f)

        results.setdefault("R", []).append(acc(ref))

    cands = [(nm if nm else a, load_spec(a)) for a, nm in zip(args.A, args.A_name or [""] * len(args.A))]
    for cname, c in cands:
        for f in range(3):
            results.setdefault(cname, []).append(acc(c))

    for k, v in results.items():
        print(f"{k:22s} accs={[round(x,4) for x in v]} mean={np.mean(v):.4f}")
    r_mean = np.mean(results["R"])
    for cname in [c for c, _ in cands]:
        d = np.mean(results[cname]) - r_mean
        print(f"判定 {cname}: 相对R {d:+.4f} {'→ 值得上LB' if d > 0.005 else ('→ 噪声内' if d > -0.005 else '→ 判负')}")


if __name__ == "__main__":
    main()
