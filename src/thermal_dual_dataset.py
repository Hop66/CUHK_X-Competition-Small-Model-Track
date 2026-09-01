"""Thermal 双流数据集：静态热像流（S）+ 骨架运动流（M）同 clip 对齐。

- S 流复用 ThermalVideoDataset（大框 crop / segment 采样 / gray_norm / 增强全继承）
- M 流从离线骨架运动缓存 outputs/motion_cache.pkl 取 [N,29] f16，按 T 比例重采样 → [T,29]
  （骨架 10fps vs 热像 24fps，用比例对齐非帧号；缺失 clip → 补零 + valid=0）
- 返回 (x[T,3,H,W], motion[T,29] f32, valid[T], label, subject)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import ThermalVideoDataset
from src.skeleton_motion import resample_to_T


class ThermalDualDataset(Dataset):
    def __init__(self, clips, num_frames: int = 8, size: int = 112, is_train: bool = True,
                 crop_cache: Optional[Dict] = None,
                 motion_cache: Optional[Dict[str, np.ndarray]] = None,
                 sample_mode: str = "segment",
                 mean_std=None, aug_strength: int = 2, seed: int = 0,
                 sample_offset: float = -1.0):
        self.clips = clips
        self.num_frames = num_frames
        self.motion_cache = motion_cache or {}
        # 静态热像流：完全复用 ThermalVideoDataset（crop/aug/segment/gray 由它负责）
        self._tv = ThermalVideoDataset(
            clips, num_frames, size, is_train, crop_cache or {},
            seed=seed, aug_strength=aug_strength, sample_offset=sample_offset,
            sample_mode=sample_mode, mean_std=mean_std)
        self._cov = None

    def coverage(self) -> float:
        """骨架运动流对 thermal clips 的命中覆盖（调用后才能打印准确值）。"""
        if self._cov is None:
            hit = 0
            for c in self.clips:
                key = f"{c.action_id}/{c.subject}/{c.sample}"
                a = self.motion_cache.get(key)
                if a is not None and getattr(a, "shape", (0,))[0] > 0:
                    hit += 1
            self._cov = hit / max(len(self.clips), 1)
        return self._cov

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i: int):
        x, label, subject = self._tv[i]          # [T,3,H,W], int, str
        c = self.clips[i]
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        arr = self.motion_cache.get(key)
        T = self.num_frames
        if arr is not None and arr.shape[0] > 0:
            motion = resample_to_T(np.asarray(arr, dtype=np.float32), T)   # [T,29]
            valid = 1
        else:
            motion = np.zeros((T, 29), np.float32)
            valid = 0
        return (x, torch.from_numpy(motion), torch.tensor(float(valid)),
                torch.tensor(int(label)), subject)
