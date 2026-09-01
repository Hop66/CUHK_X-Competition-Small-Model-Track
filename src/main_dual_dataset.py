"""Main 双流数据集：静态 DepthIR 流（R2+1D 固定框）+ 骨架运动流（M）同 clip 对齐。

- S 流复用 DepthIRVideoDataset（crop_cache 固定框 / sample_mode / 增强全继承）
- M 流从离线骨架运动缓存 outputs/motion_cache.pkl 取 [N,29] f16 → 比例重采样 [T,29]
  （骨架与 IR/depth 同相机场景同 key；缺失 → 补零 + valid=0）
- 返回 (x[T,4,H,W], motion[T,29] f32, valid, label, subject)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import DepthIRVideoDataset
from src.skeleton_motion import resample_to_T


class DepthIRDualDataset(Dataset):
    def __init__(self, clips, num_frames: int = 16, size: int = 128, is_train: bool = True,
                 crop_cache: Optional[Dict] = None,
                 motion_cache: Optional[Dict[str, np.ndarray]] = None,
                 sample_mode: str = "uniform",
                 aug_strength: int = 2, seed: int = 0, sample_offset: float = -1.0):
        self.clips = clips
        self.num_frames = num_frames
        self.motion_cache = motion_cache or {}
        self._dv = DepthIRVideoDataset(
            clips, num_frames, size, is_train, crop_cache or {},
            seed=seed, aug_strength=aug_strength, sample_offset=sample_offset,
            sample_mode=sample_mode)
        self._cov = None

    def coverage(self) -> float:
        if self._cov is None:
            hit = 0
            for c in self.clips:
                a = self.motion_cache.get(f"{c.action_id}/{c.subject}/{c.sample}")
                if a is not None and getattr(a, "shape", (0,))[0] > 0:
                    hit += 1
            self._cov = hit / max(len(self.clips), 1)
        return self._cov

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i: int):
        x, label, subject = self._dv[i]           # [T,4,H,W], int, str
        key = f"{self.clips[i].action_id}/{self.clips[i].subject}/{self.clips[i].sample}"
        arr = self.motion_cache.get(key)
        T = self.num_frames
        if arr is not None and arr.shape[0] > 0:
            motion = resample_to_T(np.asarray(arr, dtype=np.float32), T)
            valid = 1
        else:
            motion = np.zeros((T, 29), np.float32)
            valid = 0
        return (x, torch.from_numpy(motion), torch.tensor(float(valid)),
                torch.tensor(int(label)), subject)
