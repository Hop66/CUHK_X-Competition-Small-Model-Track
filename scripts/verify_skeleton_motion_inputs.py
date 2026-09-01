#!/usr/bin/env python3
"""验证骨架运动通路的前提（服务器一次、CPU 分钟级，非重复内容检查）。

历史已全量验证: 结构[17,3]米制 / conf恒1.0(1.5M全扫) / 缺失·抖动·肩宽 / H3.6M-17拓扑 / 轴序。
本脚本只补 4 个「运动特征设计前提」：
  ① root(骨盆) x/y 是否中心化（std<1e-3 占比）→ 决定 extract_motion_features 走 绝对 / fallback 分支
  ② 骨架 clip 对 thermal clip 的同 key 覆盖率（运动流数据源完整性；key=f"{action_id}/{subject}/{sample}" 同构）
  ③ 每 clip 骨架帧数分布（<8 帧的比例 = 重采样困难的 clip）
  ④ 坐标范围 sanity（米制量级，z 高度应 0~2.3m）

用法:
  全量:    cd ~/Multimodal && python scripts/verify_skeleton_motion_inputs.py
  单目录:  python scripts/verify_skeleton_motion_inputs.py --pred_dir data/Training/data/Skeleton   # 本地自测
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--pred_dir", default="", help="给定单个 predictions 目录则只统计它（本地自测）")
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 个 clip（调试）")
    args = ap.parse_args()

    if args.pred_dir:
        from src.skeleton_motion import load_skeleton
        kp, _ = load_skeleton(Path(args.pred_dir))
        root = kp[:, 0]
        print(f"[pred_dir] frames={kp.shape[0]} root_xy_std={root[:, :2].std():.5f} "
              f"z_range=({kp[:, :, 2].min():.3f},{kp[:, :, 2].max():.3f})")
        return

    from src.dataset import build_thermal_index
    from src.skeleton_dataset import build_skeleton_index
    from src.skeleton_motion import load_skeleton

    root = Path(args.train_root).expanduser()
    sk_clips = build_skeleton_index(root)
    th_clips = build_thermal_index(root)
    th_keys = {f"{c.action_id}/{c.subject}/{c.sample}" for c in th_clips}
    print(f"[skeleton] {len(sk_clips)} / thermal {len(th_clips)}", flush=True)

    n = len(sk_clips) if not args.limit else min(args.limit, len(sk_clips))
    root_std = []
    nframes = []
    missing = 0
    covered = 0
    zmin, zmax = [], []
    for i in range(n):
        c = sk_clips[i]
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        if key in th_keys:
            covered += 1
        kp, _ = load_skeleton(c.pred_dir)
        if kp.shape[0] == 0:
            missing += 1
            continue
        nframes.append(kp.shape[0])
        root_std.append(float(kp[:, 0, :2].std()))
        zmin.append(float(kp[:, :, 2].min()))
        zmax.append(float(kp[:, :, 2].max()))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    rs = np.asarray(root_std)
    nf = np.asarray(nframes)
    print("\n==== ① root 中心化判定 ====")
    print(f"root_xy_std: mean={rs.mean():.5f} median={np.median(rs):.5f} "
          f"p90={np.percentile(rs, 90):.5f} | 中心化比例(std<1e-3)={(rs < 1e-3).mean() * 100:.1f}%")
    cent = (rs < 1e-3).mean() > 0.9
    verdict = "数据根中心化：运动特征走 FALLBACK 分支" if cent else "有绝对空间位移：走绝对分支"
    print(f"  → {verdict}")
    print("\n==== ② thermal 同 key 覆盖 ====")
    print(f"骨架 clip {n} 中命中 thermal key: {covered} ({covered / max(n, 1) * 100:.1f}%) | 骨架缺失 clip: {missing}")
    print("\n==== ③ 帧数分布 ====")
    print(f"nframes: min={nf.min()} median={int(np.median(nf))} max={nf.max()} | <8帧比例={(nf < 8).mean() * 100:.1f}%")
    print("\n==== ④ 坐标范围 sanity（米制，z=高度） ====")
    print(f"z: [{np.min(zmin):.3f}, {np.max(zmax):.3f}]  (期望 ~[0, 2.3])")
    print(f"x_all: min={np.min(zmin):.3f} max={np.max(zmax):.3f} (占位)")


if __name__ == "__main__":
    main()
