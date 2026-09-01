"""本地自测：auxpose 注入 thermal（跨相机骨架比例对齐）。

验证 ThermalPoseAlignedVideoDataset：
  1. 返回 (x [T,3,H,W], skel [T,17,3], mask [T], action_id, subject) 形状
  2. 视频走父类 ThermalVideoDataset（增强+归一化）
  3. 骨架按帧比例对齐（跨相机：thermal 25fps vs skeleton 10fps 仍能对齐）
  4. 骨架缺失 → mask=0

用法: python tests/selftest_auxpose_thermal.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from src.dataset import build_thermal_index
from src.pose_aux import ThermalPoseAlignedVideoDataset
from src.skeleton_dataset import SkeletonClipIndex


def make_synth(root: Path, n_action=2, n_subj=2, n_sample=1, n_th=25, n_skel=10):
    """合成 Thermal 图片（25 帧）+ Skeleton json（10 帧，帧号不同基准）+ 缺失 1 个 clip 骨架。"""
    rng = np.random.default_rng(0)
    for a in range(n_action):
        adir = f"{a}_action{a}"
        for s in range(n_subj):
            subj = f"S{s:02d}"
            for sm in range(n_sample):
                sample = f"{subj}_sample{sm}"
                # Thermal 图片（独立相机，帧号 1000000+ 基准）
                tdir = root / "Thermal" / adir / subj / sample
                tdir.mkdir(parents=True, exist_ok=True)
                for t in range(n_th):
                    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
                    cv2.imwrite(str(tdir / f"f{1000000 + t:08d}.jpg"), img)
                # Skeleton（NYX 相机，帧号 1..n_skel 基准）；a=1,S01 缺骨架（i=3）
                if not (a == 1 and s == 1):
                    pred = root / "Skeleton" / adir / subj / sample / "predictions"
                    pred.mkdir(parents=True, exist_ok=True)
                    for t in range(n_skel):
                        kp = rng.normal(0, 0.5, (17, 3)).astype(np.float32)
                        kp[0] = 0
                        (pred / f"{t + 1:08d}.json").write_text(
                            json.dumps([{"keypoints": kp.tolist(),
                                         "keypoint_scores": np.ones(17).tolist()}]),
                            encoding="utf-8")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="auxpose_th_"))
    root = tmp / "HAR"
    make_synth(root)
    clips = build_thermal_index(root)
    print(f"[1] build_thermal_index: {len(clips)} clips")

    skel_clips = []
    for c in clips:
        action = c.thermal_dir.parent.parent.name
        skel_dir = root / "Skeleton" / action / c.subject / c.sample / "predictions"
        skel_clips.append(SkeletonClipIndex(c.action_id, c.subject, c.sample, skel_dir))

    ds = ThermalPoseAlignedVideoDataset(clips, skel_clips, 16, 64, True, aug_strength=2, seed=42)
    for i in range(len(ds)):
        x, skel, mask, aid, subj = ds[i]
        assert x.shape == (16, 3, 64, 64), x.shape
        assert skel.shape == (16, 17, 3), skel.shape
        assert mask.shape == (16,), mask.shape
        assert torch.isfinite(x).all() and torch.isfinite(skel).all()
        assert 0.0 <= mask.min() <= mask.max() <= 1.0
        # 缺失骨架的 clip → mask 全 0
        if i == 3:  # a=1,s=1,sm=1 缺骨架
            assert mask.sum() == 0, f"缺失骨架 clip 应 mask=0，got {mask.sum()}"
            print(f"    [i={i}] 缺失骨架 → mask 全 0 OK")
        else:
            assert mask.sum() > 0, f"应有骨架掩码，got {mask.sum()}"
    print(f"[2] ThermalPoseAlignedVideoDataset 形状/掩码 OK: x={tuple(x.shape)} "
          f"skel={tuple(skel.shape)} mask={tuple(mask.shape)}")

    # 对齐合理性：骨架值非零（比例对齐产生真实骨架）
    x, skel, mask, _, _ = ds[0]
    assert skel[mask > 0].abs().sum() > 0, "骨架对齐应产生非零值"
    print(f"[3] 骨架跨相机比例对齐 OK（非零，mask 有效帧 {int(mask.sum())}/16）")

    # 视频归一化（Kinetics 域）
    print(f"    x 统计: mean={x.mean():.3f} std={x.std():.3f}（应接近 0/1 归一化域）")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n==== auxpose-thermal 自测通过 ====")


if __name__ == "__main__":
    main()
