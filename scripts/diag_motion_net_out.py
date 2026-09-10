#!/usr/bin/env python3
"""空腹诊断: MotionNet 在「测试侧/训练域」输入下的输出特性 + 输入不变性。

回答用户的疑问:
  Q: fold 下骨架运动明显提升(同域), 但测试侧 AB 改 motion 编码 argmax 一字不差 —— 定有问题?
  机制检验:
    1) 测试侧 MotionNet softmax 是否退化 (熵高=近均匀 / 熵低但集中在少数错误类)
    2) 测试侧 MotionNet 输出是否对输入数值不变 (base vs kp重采样×2.0: pm 逐字节?)
    3) 与训练域(同域, fold val 有用) 输出特性对比 —— 定位 "为什么同域有用、跨域崩"
    4) motion-only 在测试侧的 argmax 分布 (是否塌缩到少数类)

用法(CPU 即可, MotionNet 只有两个 1D conv):
  python scripts/diag_motion_net_out.py [--limit 405]
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.motion_net import MotionNet
from src.skeleton_motion import extract_motion_features, load_skeleton, resample_to_T

J_TEST = Path.home() / "Multimodal/data/Testing/data/small_model_track_test"
MC = Path.home() / "Multimodal/outputs/motion_cache.pkl"
CKPT = Path.home() / "Multimodal/outputs/main_dual_full/main_SM_full_seed42.pth"  # full-SM fp32
T = 16


def load_motion(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m = MotionNet()
    m.load_state_dict({k[len("motion."):]: v for k, v in sd.items() if k.startswith("motion.")})
    m.eval()
    a = float(torch.sigmoid(sd["a"]).item()) if "a" in sd else float("nan")
    return m, a


def run(m, feats):
    """feats: [N, T, 29] -> softmax [N,40]"""
    with torch.no_grad():
        p = torch.softmax(m(torch.from_numpy(np.asarray(feats, np.float32))), -1).numpy()
    return p


def mstats(name, p):
    H = -(p * np.log(p + 1e-12)).sum(1)          # 熵 nats (均匀40类=ln40≈3.689)
    conf = p.max(1)
    pred = p.argmax(1)
    uni = np.ones(40) / 40
    kld = (p * (np.log(p + 1e-12) - np.log(uni))).sum(1)  # 与均匀的 KL
    print(f"-- {name}: clips={len(p)}")
    print(f"   熵 mean={H.mean():.3f} (均匀=3.689 {'' if H.mean()<3.5 else '<-- 接近均匀!'}) "
          f"| top1置信 mean={conf.mean():.3f}")
    print(f"   去重预测类数={len(np.unique(pred))} | 预测类频次top5={np.bincount(pred, minlength=40).argsort()[-5:][::-1].tolist()}")
    return H, conf, pred


def gather_test(rsf=0.0, ss=1.0):
    feats, nf = [], []
    for d in sorted(J_TEST.iterdir()):
        if not d.is_dir() or not d.name.startswith("SM_test_"):
            continue
        kp, _ = load_skeleton(d / "Skeleton" / "predictions")
        if kp.shape[0] == 0:
            continue
        M = max(2, int(round(kp.shape[0] * rsf))) if rsf > 0 else None
        feats.append(extract_motion_features(kp, T=T, speed_scale=ss, resample=M))
        nf.append(kp.shape[0])
    return np.stack(feats), np.asarray(nf)


def gather_train(limit):
    with open(MC, "rb") as f:
        cache = pickle.load(f)
    out, nf = [], []
    for k in list(cache.keys())[:limit]:
        arr = np.asarray(cache[k], np.float32)   # [N,29] 训练域特征(提取时 speed_scale=1, 无重采样)
        out.append(resample_to_T(arr, T))
        nf.append(arr.shape[0])
    return np.stack(out), np.asarray(nf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=405)
    args = ap.parse_args()
    m, alpha = load_motion(CKPT)
    print(f"==== MotionNet(full-SM fp32) | learned α = {alpha:.3f} ====", flush=True)

    # 训练域(同域, 训练时 MotionNet 见过的)
    tr, tr_nf = gather_train(args.limit)
    p_tr = run(m, tr)
    mstats("训练域 motion_cache (同域, 应有信息量)", p_tr)

    # 测试侧两种编码: base(ss0.5, 即推理基线) vs kp升采样×2.0 (真·重采样)
    te_base, te_nf = gather_test(rsf=0.0, ss=0.5)
    te_rs, _ = gather_test(rsf=2.0, ss=1.0)
    p_base = run(m, te_base)
    p_rs = run(m, te_rs)
    H1, c1, pr1 = mstats("测试侧 base编码 (ss0.5)", p_base)
    H2, c2, pr2 = mstats("测试侧 kp重采样×2.0", p_rs)

    # 输入不变性: base vs rs20 的 MotionNet 输出
    same = (pr1 == pr2).mean()
    l1 = np.abs(p_base - p_rs).mean()
    print(f"\n==== 输入不变性: MotionNet输出 base vs kp×2.0 ====")
    print(f"   softmax argmax 一致率 = {same:.4f}  (1.0 = 输出完全与输入无关)")
    print(f"   逐类概率 |Δp| mean     = {l1:.5f}")
    print(f"   熵差 base-rs = {H1.mean()-H2.mean():+.4f} | top1置信差 = {c1.mean()-c2.mean():+.4f}")

    # motion-only 在测试侧的 argmax 分布 (塌缩程度)
    bc = np.bincount(pr1, minlength=40)
    topk = bc.argsort()[-8:][::-1]
    print(f"\n  [测试侧 motion-only 预测分布] 去重类数={len(np.unique(pr1))} | top8类={[(int(i),int(bc[i])) for i in topk]})")

    print("\n[判读]")
    if same > 0.999:
        print("  ❗ MotionNet 对测试侧 motion 输入数值完全不敏感(输出近固定) → 这正是 AB '一字不差' 的机制")
    if H1.mean() > 3.5:
        print("  ⚠️ 测试侧 motion softmax 近均匀 → 至少不是高置信错误")
    elif len(np.unique(pr1)) < 15:
        print(f"  ⚠️ 测试侧 motion-only 坍塌到 {len(np.unique(pr1))} 类 → 高置信错误偏置 → 融合时拖累 static")
    print(f"  训练域熵={H1.mean() if False else ''}对比 → 见上方数字")


if __name__ == "__main__":
    main()
