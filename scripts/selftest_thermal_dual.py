#!/usr/bin/env python3
"""Thermal 双流本地自测（synthetic）：dataset 形状 + 双流前向。用法: python scripts/selftest_thermal_dual.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.dataset import ThermalClipIndex
from src.thermal_dual_dataset import ThermalDualDataset
from scripts.train_thermal_dual import ThermalDual  # noqa: N812


def _frames_dir(n=10):
    """确保一个含 n 张 8bit 帧的目录（本地样例不足则自造）。"""
    d = Path("data/Training/data/Thermal")
    if d.is_dir() and len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png"))) >= 8:
        return d
    import cv2
    d = Path("outputs/selftest_dual_frames")
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((64, 64, 3), 40 + i * 20 % 180, np.uint8)
        cv2.imwrite(str(d / f"frame_{i:06d}.jpg"), img)
    return d


def main():
    frames = _frames_dir()
    clips = [ThermalClipIndex(a, "s1", f"1-{a}-1", frames) for a in range(3)]
    motion_cache = {f"{a}/s1/1-{a}-1": np.random.default_rng(a).random((60, 29)).astype(np.float16)
                    for a in range(3)}
    motion_cache["0/s1/1-0-1"] = np.zeros((0, 29), np.float16)  # 缺帧 clip → valid=0 路径

    ds = ThermalDualDataset(clips, num_frames=8, size=112, is_train=True,
                            crop_cache={}, motion_cache=motion_cache,
                            sample_mode="segment", mean_std=((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)))
    print(f"[dataset] len={len(ds)} coverage={ds.coverage():.3f}", flush=True)
    for i in range(len(ds)):
        x, m, v, y = ds[i][0], ds[i][1], ds[i][2], ds[i][3]
        print(f"  item{i}: x={tuple(x.shape)} motion={tuple(m.shape)} valid={int(v)} label={int(y)}")
        assert tuple(x.shape) == (8, 3, 112, 112)
        assert tuple(m.shape) == (8, 29)
        assert v.item() in (0.0, 1.0)

    # 模型前向（resnet18 S+M / framenet S+M / M-only）
    x_in = next(iter(ds))[0].unsqueeze(0)
    t_in = next(iter(ds))[1].unsqueeze(0)
    for arch, us, um in [("resnet18", True, True), ("framenet", True, True), ("framenet", False, True)]:
        mdl = ThermalDual(arch, us, um, "learn")
        out, comp = mdl(x_in, t_in)
        print(f"[fwd] arch={arch} S={us} M={um} out={tuple(out.shape)} comp={sorted(comp.keys())}")
        assert tuple(out.shape) == (1, 40)
    print("[OK] thermal dual selftest passed")


if __name__ == "__main__":
    main()
