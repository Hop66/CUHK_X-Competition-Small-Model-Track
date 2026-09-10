#!/usr/bin/env python3
"""512d encoder 特征空间验证（无泄漏跨折）—— 回答"真信息是否在 head 之前"。

对比 40d logits（head 输出）vs 512d encoder 特征：
  1) 难对跨折可分性（LDA / NCM 二分类）
  2) 全 40 类 LR / NCM 分类 vs softmax 基线
  3) 难对 rerank：softmax 预测落难对时，用 512d NCM 在难对内重判

产物依赖: outputs/oof/main_feats512.pkl (aug2 3折 512d)
          outputs/oof/main_oof.pkl (40d, 同模型)
用法: python scripts/feat512_discrim_bench.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

HP = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24),
      (6, 37), (6, 7), (11, 14), (7, 19), (38, 39)]
HPID = {x for p in HP for x in p}
HP_SYM = {(min(a, b), max(a, b)) for a, b in HP}


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    FE = pickle.load(open("outputs/oof/main_feats512.pkl", "rb"))  # {fold: {key: 512}}
    MK = pickle.load(open("outputs/oof/main_oof.pkl", "rb"))       # {fold: {key: 40}}

    print("=== 512d vs 40d: 难对跨折可分性 (LDA 二分类, train fold≠test fold, 无泄漏) ===")
    for f in range(3):
        oth = [x for x in range(3) if x != f]
        base = []
        f5_base = []
        for a, b in HP:
            # train(其他折) 用于训 LDA: feat 512d
            Ftr, ytr = [], []
            Ltr, ytrL = [], []
            for o in oth:
                for k in FE[o]:
                    lab = int(k.split("/")[0])
                    if lab in (a, b):
                        Ftr.append(FE[o][k]); ytr.append(1 if lab == a else 0)
            if len(set(ytr)) < 2 or len(ytr) < 10:
                continue
            Xtr = np.stack(Ftr)
            lda = LDA().fit(Xtr, np.array(ytr))
            # val(当前折) 特征
            Fva, yva, Lva = [], [], []
            for k in FE[f]:
                lab = int(k.split("/")[0])
                if lab in (a, b) and k in MK[f]:
                    Fva.append(FE[f][k]); yva.append(1 if lab == a else 0); Lva.append(MK[f][k])
            if len(set(yva)) < 2 or len(yva) < 4:
                continue
            Xva = np.stack(Fva); yva = np.array(yva)
            # 512d LDA
            acc512 = (lda.predict(Xva) == yva).mean()
            # 40d softmax 判难对: 比较两类的 softmax 概率
            Lv = np.stack(Lva)
            ps = sf(Lv)
            acc40 = ((ps[:, a] > ps[:, b]) == (yva == 1)).mean()
            base.append((a, b, acc512, acc40, len(Fva)))
        if base:
            m512 = np.mean([x[2] for x in base]); m40 = np.mean([x[3] for x in base])
            print(f"  fold{f}: 512d LDA 难对acc={m512:.3f}  40d softmax 难对acc={m40:.3f}  (n_pairs={len(base)})")

    # 全 40 类: 512d NCM/LR vs 40d softmax
    print("\n=== 全40类: 512d LR / NCM vs 40d softmax 基线 ===")
    for f in range(3):
        oth = [x for x in range(3) if x != f]
        Ftr, ytr = [], []
        for o in oth:
            for k in FE[o]:
                Ftr.append(FE[o][k]); ytr.append(int(k.split("/")[0]))
        Ftr = np.stack(Ftr); ytr = np.array(ytr)
        # val
        Fv, Lv, yv = [], [], []
        for k in FE[f]:
            if k in MK[f]:
                Fv.append(FE[f][k]); Lv.append(MK[f][k]); yv.append(int(k.split("/")[0]))
        Fv = np.stack(Fv); Lv = np.stack(Lv); yv = np.array(yv)
        # 基线
        base = (sf(Lv).argmax(-1) == yv).mean()
        # 标准化特征
        mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
        Ftrn = (Ftr - mu) / sd; Fvn = (Fv - mu) / sd
        # LR
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ftrn, ytr)
            a_lr = (clf.predict(Fvn) == yv).mean()
        except Exception:
            a_lr = -1
        # NCM(余弦)
        mu_c = np.array([Ftrn[ytr == c].mean(0) if (ytr == c).sum() else np.zeros(Ftrn.shape[1]) for c in range(40)])
        mu_c = mu_c / (np.linalg.norm(mu_c, axis=-1, keepdims=True) + 1e-9)
        Fvn_n = Fvn / (np.linalg.norm(Fvn, axis=-1, keepdims=True) + 1e-9)
        a_ncm = (Fvn_n @ mu_c.T).argmax(-1) == yv
        a_ncm = a_ncm.mean()
        print(f"  fold{f}: softmax基线={base:.4f}  LR(512d)={a_lr:.4f}  NCM余弦(512d)={a_ncm:.4f}")


if __name__ == "__main__":
    main()
