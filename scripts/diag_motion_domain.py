#!/usr/bin/env python3
"""训练 vs 测试骨架运动特征「同口径」域对比 —— 裁决 motion OOD 是否真凶

动机：fold val 复现 0.6652（训练侧提取逻辑一致），但 LB 崩（fold 三折 0.622 /
      full 0.637 vs main 0.706）。怀疑测试骨架（Color 视频,20-49帧）与训练骨架
      （~10fps,~47帧）帧率/采样密度不同 → 速度特征(/帧)跨域 → MotionNet OOD。

输出（全部对「原 N 帧特征」与「同 resample 到 16 帧」各做一遍）：
  1) 帧数分布（帧率/秒数代理）：train N mean/std vs test N mean/std
  2) 速度类 dims（6-24 关节/全局速度、24 朝向角速）per-frame 幅值均值：
     train vs test —— 若 test 显著 > train → 帧率/scale 跨域坐实
  3) 全部 29 维按均值差排序的 Top 差异 dims
判读：
  - 速度 dims 的测试幅值 ≈ 训练 → 域一致 → 骨架上无 bug，问题在别处（融合/权重）
  - 速度 dims 测试幅值 ≫(或≪) 训练 → scale/帧率 OOD → 修法：测试运动速度项按
    帧率因子归一 / 或把测试骨架重采样到训练 fps 再提特征
用法: python scripts/diag_motion_domain.py [--limit 200]
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.skeleton_motion import extract_motion_features, load_skeleton, resample_to_T

J_TEST = Path.home() / "Multimodal/data/Testing/data/small_model_track_test"
MC = Path.home() / "Multimodal/outputs/motion_cache.pkl"

# 速度/统计量相关 dims（index 见 skeleton_motion 头注释）
VEL_DIMS = list(range(6, 25))          # 6-20 关节速度, 21 幅度, 22 手, 23 朝向角, 24 角度速度
ABS_VEL = list(range(6, 23))           # 6-22 纯速度/幅度（去掉朝向角这类有符号角度）
T = 16


def norm16(arr):
    """[N,29] → [16,29]（与训练/推理同样的比例重采样）"""
    return resample_to_T(np.asarray(arr, np.float32), T)


def gather_test(speed_scale=1.0, resample_factor=0.0):
    v16, nframes = [], []
    for d in sorted(J_TEST.iterdir()):
        if not d.is_dir() or not d.name.startswith("SM_test_"):
            continue
        pd = d / "Skeleton" / "predictions"
        kp, _ = load_skeleton(pd)
        if kp.shape[0] == 0:
            continue
        M = None
        if resample_factor > 0:
            M = max(2, int(round(kp.shape[0] * resample_factor)))
        f = extract_motion_features(kp, speed_scale=speed_scale, resample=M)  # 原 N 帧
        nframes.append(kp.shape[0])
        v16.append(norm16(f))
    return np.stack(v16), np.asarray(nframes)


def gather_train(limit):
    with open(MC, "rb") as f:
        cache = pickle.load(f)
    keys = list(cache.keys())[:limit]
    v16, nframes = [], []
    for k in keys:
        arr = np.asarray(cache[k], np.float32)   # [N,29]
        nframes.append(arr.shape[0])
        v16.append(norm16(arr))
    return np.stack(v16), np.asarray(nframes)


def report(name, v16, nf):
    # 每帧速度幅值（对全部帧统计，跨 clip 汇总）
    spd = np.abs(v16[:, :, ABS_VEL])             # [C,16,len]
    vel_frame = spd.mean(axis=(1, 2))            # [C] 每 clip 每帧平均速度幅值
    print(f"-- {name}: clips={len(v16)} | N帧 mean={nf.mean():.1f} std={nf.std():.1f} "
          f"min={nf.min()} max={nf.max()}")
    print(f"   速度dims(6-22) 每帧平均幅值 = {vel_frame.mean():.4f} (std={vel_frame.std():.4f})")
    allf = np.abs(v16).mean(axis=(1, 2))         # [C] 全29维每帧平均
    print(f"   全29维 每帧平均幅值          = {allf.mean():.4f}")
    return vel_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--speed_scale", type=float, default=1.0,
                    help="对测试侧速度项缩放（验证帧率归一：传 0.5 后速度幅值比应≈1）")
    ap.add_argument("--resample_scan", action="store_true",
                    help="扫描 kp 级真·重采样因子 {0.5,0.75,1.0,1.5,2.0,2.5,3.0} 看速度比变化")
    args = ap.parse_args()

    tv, tn = gather_test(args.speed_scale)
    rv, rn = gather_train(args.limit)
    print("==== resample 到 T=16 的同口径下：训练 vs 测试 ====", flush=True)
    vt = report("训练 motion_cache", rv, rn)
    vp = report("测试骨架(重提取)", tv, tn)

    ratio = vp.mean() / max(vt.mean(), 1e-9)
    print(f"\n速度dims 测试/训练 幅值比 = {ratio:.3f}  (<0.85 或 >1.15 → 跨域坐实)")

    if args.resample_scan:
        print("\n==== kp 级真·重采样因子扫描（只改密度，不动速度项缩放）====")
        for f in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
            tvf, _ = gather_test(speed_scale=1.0, resample_factor=f)
            rf = np.abs(tvf[:, :, ABS_VEL]).mean() / max(vt.mean(), 1e-9)
            mark = "✓" if 0.85 <= rf <= 1.15 else ""
            print(f"  resample kp×{f:.2f} (M≈N×{f:.2f}) → 速度比 = {rf:.3f}  {mark}", flush=True)

    # 全部 dims 均值差 Top
    m_tr = np.abs(rv).mean(axis=(0, 1))          # [29]
    m_te = np.abs(tv).mean(axis=(0, 1))
    diff = np.argsort(-np.abs(m_te - m_tr))
    print("\n==== dims 均值差异 Top8（测试−训练，绝对值）====")
    for dd in diff[:8]:
        print(f"  dim{dd:2d}: train={m_tr[dd]:.4f} test={m_te[dd]:.4f} "
              f"(Δ={m_te[dd]-m_tr[dd]:+.4f})")

    print("\n[判读]")
    if 0.85 <= ratio <= 1.15:
        print("  速度幅值比≈1 → 训练/测试运动域一致 → 骨架域不是元凶 → 查融合/权重")
    else:
        print(f"  速度幅值比={ratio:.2f} → 测试运动特征 scale 与训练不符（帧率/pose 域）→ "
              f"真凶：motion OOD → 修法：测试速度项除以帧率因子 / 骨架按训练fps重采样")


if __name__ == "__main__":
    main()
