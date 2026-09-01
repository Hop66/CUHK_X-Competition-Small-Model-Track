"""骨架质量处理（用户强调"骨架有用但处理不对"的尝试之一）。

NYX 骨架是 MMPose 在 RGB 上提取，可能有噪声/抖动。此模块提供时间平滑
（滑动平均去关节抖动）——这是"处理骨架本身"的首次尝试。

SmoothMotionBertSkeletonDataset：继承 MotionBertSkeletonDataset，加载后平滑。
默认不影响现有行为（smooth_window=1 时完全等同原数据集）。

用法：train_skeleton_motionbert.py --smooth 5（新增参数，默认 1=关，可复位）
"""

from __future__ import annotations

import numpy as np

from src.skeleton_dataset import MotionBertSkeletonDataset


def smooth_sequence(kp: np.ndarray, window: int = 5) -> np.ndarray:
    """时间维滑动平均去噪。kp [N,17,3]（原始 3D 坐标）→ 平滑后同形状。

    window 为奇数（默认 5，10fps 下 0.5 秒窗口）；边界用可用邻域均值。
    """
    if window <= 1 or kp.shape[0] <= window:
        return kp
    half = window // 2
    n = kp.shape[0]
    out = np.zeros_like(kp)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        out[t] = kp[lo:hi].mean(axis=0)
    return out


class SmoothMotionBertSkeletonDataset(MotionBertSkeletonDataset):
    """继承 MotionBertSkeletonDataset，加载 3D 骨架后做时间平滑。

    smooth_window=1（默认）时与父类完全一致（可复位）。
    """

    def __init__(self, clips, num_frames: int = 16, is_train: bool = True, seed: int = 0,
                 input3d: bool = True, smooth_window: int = 1):
        super().__init__(clips, num_frames, is_train, seed, input3d=input3d)
        self.smooth_window = smooth_window

    def _load_frames(self, pred_dir):
        kp, conf = super()._load_frames(pred_dir)
        if self.smooth_window > 1 and kp.shape[0] > self.smooth_window:
            kp = smooth_sequence(kp, self.smooth_window)
        return kp, conf
