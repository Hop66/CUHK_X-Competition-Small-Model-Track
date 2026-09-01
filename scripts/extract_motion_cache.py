#!/usr/bin/env python3
"""伺服器：骨架→运动特征全量缓存（thermal 双流运动通路的数据源）。

产出: outputs/motion_cache.pkl = {clip_key(f"{action_id}/{subject}/{sample}"): arr[N,29] float16}
  - 按 build_skeleton_index 同路径 discovery（与 thermal/depth/IR 同 key，天然对齐）
  - T 重采样在训练侧按需做（比例对齐，非帧号硬对齐）
  - 骨架缺失 clip 不写入（训练/推理侧按 key 缺失补零 + valid 掩码）

用法（服务器，一次）:
  sbatch scripts/extract_motion_cache.sbatch   （或直接 python scripts/extract_motion_cache.py）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.skeleton_motion import build_motion_cache  # noqa: E402

if __name__ == "__main__":
    root = Path.home() / "Multimodal" / "data" / "Training" / "HAR"
    out = str(Path.home() / "Multimodal" / "outputs" / "motion_cache.pkl")
    print(f"[extract_motion] root={root}", flush=True)
    cache = build_motion_cache(root, out)
    print(f"DONE clips={len(cache)} -> {out}", flush=True)
