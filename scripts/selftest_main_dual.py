#!/usr/bin/env python3
"""Main 双流本地自测（synthetic）：DepthIRDualDataset 形状 + MainDual 前向。
用法: python scripts/selftest_main_dual.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset import ClipIndex
from src.main_dual_dataset import DepthIRDualDataset
from scripts.train_main_dual import MainDual


def main():
    depth = Path("data/Training/data/Depth_Color")
    ir = Path("data/Training/data/IR")
    assert depth.is_dir() and ir.is_dir(), "本地样例缺 Depth_Color/IR"
    clips = [ClipIndex(0, "s1", "1-1-1", depth, ir)]
    key = "0/s1/1-1-1"
    motion = {key: np.random.default_rng(0).random((60, 29)).astype(np.float16)}

    ds = DepthIRDualDataset(clips, num_frames=8, size=64, is_train=True,
                            crop_cache={}, motion_cache=motion, sample_mode="uniform")
    print(f"[dataset] len={len(ds)} coverage={ds.coverage():.3f}", flush=True)
    x, m, v, y, sub = ds[0]
    print(f"  x={tuple(x.shape)} motion={tuple(m.shape)} valid={int(v)} label={int(y)}")
    assert tuple(x.shape) == (8, 4, 64, 64)
    assert tuple(m.shape) == (8, 29)

    mdl = MainDual(weights_path=None, use_motion=True)
    out, comp = mdl(x.unsqueeze(0), m.unsqueeze(0))
    print(f"[fwd SM] out={tuple(out.shape)} comp={sorted(comp.keys())}")
    assert tuple(out.shape) == (1, 40)

    mdl2 = MainDual(weights_path=None, use_motion=False)
    out2, comp2 = mdl2(x.unsqueeze(0), m.unsqueeze(0))
    print(f"[fwd S ] out={tuple(out2.shape)} comp={sorted(comp2.keys())}")
    assert tuple(out2.shape) == (1, 40)
    print("[OK] main dual selftest passed")


if __name__ == "__main__":
    main()
