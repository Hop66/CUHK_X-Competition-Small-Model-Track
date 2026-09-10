#!/usr/bin/env python3
"""诊断：SM dual 提交 0.55223 回退根因 —— 对比训练 motion_cache 与测试骨架提取的「运动特征域」。

假设 1：测试骨架读取路径/格式对不上 → motion 全零 → 稀释 static。
假设 2：测试骨架是绝对坐标（root x/y 有位移）→ extract_motion_features 走
        has_abs 分支，与训练（root 中心化 → fallback 分支）域不一致 → MotionNet 崩。
假设 3：轴序/单位/拓扑不一致。

输出：
  1) 测试 Skeleton 目录结构预览（首 clip）
  2) 测试骨架 kp 统计：帧数、root x/y std、坐标范围
  3) 测试运动特征活性：energy 分布、零 clip 数
  4) 训练 motion_cache 对照：覆盖数、root 中心化标志、feat 统计
  5) 结论建议

用法: python scripts/diag_test_motion.py
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.skeleton_motion import extract_motion_features, load_skeleton

J_ROOT = 0

TEST = Path.home() / "Multimodal/data/Testing/data/small_model_track_test"
TRAIN_CACHE = Path.home() / "Multimodal/outputs/motion_cache.pkl"


def quick_stats(kp):
    if kp.shape[0] == 0:
        return None
    root = kp[:, J_ROOT, :]
    return dict(
        n_frames=int(kp.shape[0]),
        kp_min=float(kp.min()), kp_max=float(kp.max()),
        root_xy_std=float(np.std(root[:, :2])),
        root_z_min=float(root[:, 2].min()), root_z_max=float(root[:, 2].max()),
    )


def main():
    print("==== [1] 测试骨架目录结构预览 ====")
    d0s = [d for d in sorted(TEST.iterdir())
           if d.is_dir() and d.name.startswith("SM_test_")]
    print(f"SM_test_* 目录数 = {len(d0s)}")
    if d0s:
        d = d0s[0]
        print(f"[{d.name}] 顶层 = {[p.name for p in sorted(d.iterdir())][:10]}")
        skel = d / "Skeleton"
        if skel.is_dir():
            print(f"  Skeleton/ = {[p.name for p in sorted(skel.iterdir())][:10]}")
            pred = skel / "predictions"
            if pred.is_dir():
                js = sorted(pred.glob("*.json"))
                print(f"  Skeleton/predictions/ = {len(js)} json, 首: {js[:2]}")

    print("\n==== [2] 测试骨架 kp 统计（前 8 clip）====")
    stat_tests = []
    n_empty = 0
    for d in d0s[:8]:
        pd = d / "Skeleton" / "predictions"
        kp, conf = load_skeleton(pd)
        st = quick_stats(kp)
        if st is None:
            n_empty += 1
            print(f"  {d.name}: 空（0 帧）")
        else:
            stat_tests.append(st)
            print(f"  {d.name}: {st}")
    print(f"  [前8] 空={n_empty}")

    print("\n==== [3] 测试运动特征活性（405 全量）====")
    fees = []
    zero_clips = 0
    for d in d0s:
        pd = d / "Skeleton" / "predictions"
        kp, _ = load_skeleton(pd)
        f = extract_motion_features(kp, T=16)
        e = float(np.abs(f).sum())
        fees.append(e)
        if e < 1e-4:
            zero_clips += 1
    fees = np.asarray(fees)
    print(f"  energy mean={fees.mean():.4f} median={np.median(fees):.4f} "
          f"std={fees.std():.4f} max={fees.max():.4f}")
    print(f"  ** 零运动 clip = {zero_clips}/{len(d0s)} ({zero_clips/max(len(d0s),1):.1%}) **")

    print("\n==== [4] 训练 motion_cache 对照 ====")
    if TRAIN_CACHE.exists():
        with open(TRAIN_CACHE, "rb") as fh:
            cache = pickle.load(fh)
        vals = list(cache.values())
        feas = np.array([np.abs(np.asarray(v, dtype=np.float32)).sum() for v in vals])
        print(f"  覆盖 clips = {len(cache)}  样本形状 = {vals[0].shape if vals else 'N/A'}")
        print(f"  energy mean={feas.mean():.4f} median={np.median(feas):.4f} "
              f"std={feas.std():.4f} max={feas.max():.4f}")
    else:
        print("  ⚠️ 缺训练缓存", TRAIN_CACHE)
    # 训练 root 中心化确认（决定特征走 has_abs 还是 fallback 分支）
    import glob as _g
    fs = _g.glob(str(Path.home() / "Multimodal/data/Training/data/Skeleton/**/predictions/*.json"),
                 recursive=True)
    if fs:
        import json as _j
        data = _j.loads(Path(fs[0]).read_text(encoding="utf-8"))
        fr = data[0] if isinstance(data, list) else data
        kp_t = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
        print(f"  训练骨架取样: {Path(fs[0]).parent.parent.parent.name}/"
              f"{Path(fs[0]).parent.parent.name}")
        print(f"  训练 root x/y std = {float(np.std(kp_t[:, 0])):.5f} / "
              f"{float(np.std(kp_t[:, 1])):.5f}  (≈0=root 中心化 → fallback 分支)")

    print("\n==== [5] 判读 ====")
    if zero_clips == len(d0s):
        print("  ❌ 全部零运动 → 测试骨架路径/解析失败（MOTION 全零，稀释 static）→ 查 [1] 结构")
    elif zero_clips > 0:
        print(f"  ⚠️ {zero_clips} 个零运动 clip（会被门控为纯 static）；其余有活性但需比域")
    # root 中心化对比
    rxy_tests = [st["root_xy_std"] for st in stat_tests]
    if rxy_tests:
        mx = max(rxy_tests)
        print(f"  测试 root x/y std max = {mx:.5f}")
        print(f"  root_xy_std > 1e-3 即走 has_abs 分支（训练 root 中心化走 fallback 分支）"
              f" → 域不一致是根因" if mx > 1e-3 else
              f"  root_xy_std ≈ 0 → 测试也是 root 中心化，走 fallback 分支（与训练一致）✓")


if __name__ == "__main__":
    main()
