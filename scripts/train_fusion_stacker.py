#!/usr/bin/env python3
"""多模态逐类概率 Stacking（发挥各模态作用的正解，替代失败的单标量 α 融合）。

原理（研究共识）:
  - 每个模态在 subject-3 折上的 OOF 概率 = "弱专家"输入;
  - 元学习(浅层)只学 (stream,class) 该信多少 -> 坏流/误类的权重自然趋 0;
  - CV 用 leave-one-fold-out(真 OOF-of-OOF), 最终提交在全量 OOF 拟合、对测试预测。
用法:
  1) 各流 OOF:  python scripts/extract_oof_logits.py ... --out outputs/oof/<s>_oof.pkl
  2) 训练+CV :  python scripts/train_fusion_stacker.py \
       --oof main=outputs/oof/main_oof.pkl thermal=outputs/oof/thermal_oof.pkl --model mlp
  3) 加测试出CSV: --test main=xx_test.pkl thermal=... --output sub_stack.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


def _softmax(lg):
    lg = np.asarray(lg, np.float32)
    lg = lg - lg.max(); e = np.exp(lg)
    return e / e.sum()


def collate(oof_paths):
    """把多流 OOF 按 clip key 对齐成长度一致向量。

    - 只在【所有流都覆盖】的 clip 子集上做(共同键交集);
    - 折归属取首个流(fold 应一致, 不一致则告警弃用);
    - 返回 X[N, 40*S], y[N], folds[N]。
    """
    streams = []
    for sname, p in oof_paths.items():
        d = pickle.load(open(p, "rb"))  # {fold: {key: logits}}
        key_fold = {}
        for fold, kv in d.items():
            for key, lg in kv.items():
                key_fold[key] = (fold, _softmax(lg))
        streams.append((sname, key_fold))

    # 共同键（所有流都有）
    common = set(streams[0][1].keys())
    for _, kf in streams[1:]:
        common &= set(kf.keys())

    keys = sorted(common)
    X = []
    y = []
    folds = []
    warn = 0
    for k in keys:
        vec = []
        f0 = streams[0][1][k][0]
        for _, kf in streams:
            f, lg = kf[k]
            if f != f0:
                warn += 1
            vec.append(lg)
        X.append(np.concatenate(vec))
        y.append(int(k.split("/")[0]))
        folds.append(f0)
    if warn:
        print(f"[stacker] 折不一致 clip 数 = {warn}（已按首流折归属）", flush=True)
    return np.stack(X), np.asarray(y), np.asarray(folds)


def collate_test(test_paths):
    first_list = None
    mats = {}
    keys = None
    for sname, p in test_paths.items():
        d = pickle.load(open(p, "rb"))
        kv = d[0] if (isinstance(d, dict) and 0 in d) else d
        ks = sorted(kv.keys())
        if keys is None:
            keys = ks
        else:
            assert ks == keys, f"{sname} key 顺序与首流不一致"
        mats[sname] = np.stack([_softmax(kv[k]) for k in ks], 0)
    X = np.concatenate([mats[s] for s in test_paths], 1) if keys else None
    return keys, X


def make_model(kind, D):
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0)

    class MLPFit:
        def __init__(self):
            self.net = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.35),
                                     nn.Linear(128, 40))
            self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-3, weight_decay=1e-3)
        def fit(self, Xt, yt):
            Xt = torch.from_numpy(Xt.astype(np.float32)); yt = torch.from_numpy(yt).long()
            self.net.train()
            for _ in range(150):
                self.opt.zero_grad()
                loss = nn.functional.cross_entropy(self.net(Xt), yt)
                loss.backward(); self.opt.step()
        def predict(self, Xt):
            self.net.eval()
            with torch.no_grad():
                return self.net(torch.from_numpy(Xt.astype(np.float32))).argmax(1).numpy()
    return MLPFit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True, help="stream=path")
    ap.add_argument("--model", choices=["logistic", "mlp"], default="mlp")
    ap.add_argument("--test", nargs="+", default=[], help="stream=path(测试logits, 单份{key:logits})")
    ap.add_argument("--output", default="outputs/sub_stack.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_csv", default="data/Testing/test_file/test.csv")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    oofs = dict(kv.split("=", 1) for kv in args.oof)
    X, y, folds = collate(oofs)
    D = X.shape[1]
    print(f"[stacker] streams={list(oofs)} | X{X.shape}(=40x{len(oofs)}) labels={len(set(y))}", flush=True)

    accs, preds = [], np.zeros(len(y), int)
    for f in sorted(set(folds.tolist())):
        tr, va = folds != f, folds == f
        m = make_model(args.model, D); m.fit(X[tr], y[tr])
        preds[va] = m.predict(X[va]); accs.append((preds[va] == y[va]).mean())
    S = len(oofs)
    avg_pred = np.stack([X[:, c * 40:(c + 1) * 40] for c in range(S)], 0).mean(0).argmax(1)
    acc = float(np.mean(accs)); base = float((avg_pred == y).mean())
    best_stream = max(float((X[:, c*40:(c+1)*40].argmax(1) == y).mean()) for c in range(S))
    print(f"[stacker] CV acc = {acc:.4f} | 概率平均基线 = {base:.4f} | 最强单流 = {best_stream:.4f} | 提升 = {acc-base:+.4f}", flush=True)
    print(f"[stacker] 各折acc = {[round(a,4) for a in accs]}", flush=True)

    if args.test:
        tests_in = dict(kv.split("=", 1) for kv in args.test)
        tests = {s: tests_in[s] for s in oofs if s in tests_in}   # 与 --oof 同序
        assert set(tests) == set(oofs), f"测试流需包含全部 oof 流: {set(oofs)}"
        keys, Xt = collate_test(tests)
        m = make_model(args.model, D); m.fit(X, y)
        pred = m.predict(Xt)
        import pandas as pd, re
        df = pd.read_csv(args.test_csv)
        order = df["path"].astype(str).map(lambda p: re.search(r"(SM_test_\d+)", p).group(1))
        pmap = dict(zip(keys, pred.astype(int)))
        df["prediction"] = [pmap.get(k, 0) for k in order]
        out = Path(args.output)
        df[["path", "prediction"]].to_csv(out, index=False)
        print(f"[stacker] saved {out} ({len(df)} rows, 类0={int((df['prediction']==0).sum())})", flush=True)


if __name__ == "__main__":
    main()
