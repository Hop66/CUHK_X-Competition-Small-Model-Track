"""CUHK-X —— Skeleton 针对性清洗（纯 numpy，不依赖 scipy）。

针对单目 MMPose 3D 骨架的固有噪声：
  1. 时间抖动（尤其 z 深度）→ smooth_time（滑动中值 + 轻高斯）
  2. 检测错误帧（遮挡/侧身 → 关节漂移、速度尖峰）→ fix_speed_outliers
  3. 全零帧（检测丢失）→ fix_zero_frames（线性插值）
  4. 肩宽漂移帧（整帧 scale 异常）→ fix_shoulder_outliers

用法（离线或在线）:
    from src.skeleton_clean import clean_skeleton
    kp = clean_skeleton(kp)   # kp: [T,17,3] float32
清洗应在中心化/肩宽归一化**之前**应用（原始米制坐标上做物理一致性修复）。
"""

from __future__ import annotations

import numpy as np


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """沿时间轴滑动中值。x: [T,V,C] → 同形。"""
    T = x.shape[0]
    if T < 3 or w < 3:
        return x
    w = min(w, T if T % 2 == 1 else T - 1)
    pad = w // 2
    out = np.empty_like(x)
    for t in range(T):
        lo, hi = max(0, t - pad), min(T, t + pad + 1)
        out[t] = np.median(x[lo:hi], axis=0)
    return out


def _gauss_smooth(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """沿时间轴一维高斯平滑（轻量，消除高频抖动）。x: [T,V,C]。"""
    T = x.shape[0]
    if T < 3:
        return x
    r = int(3 * sigma)
    t = np.arange(-r, r + 1)
    w = np.exp(-(t ** 2) / (2 * sigma ** 2))
    w /= w.sum()
    # 用 padding 保持长度
    xp = np.pad(x, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for i, wt in enumerate(w):
        out += wt * xp[i:i + T]
    return out


def smooth_time(x: np.ndarray, med_w: int = 5, sigma: float = 1.0) -> np.ndarray:
    """滑动中值（去尖峰）+ 轻高斯（去抖动）。"""
    return _gauss_smooth(_rolling_median(x, med_w), sigma)


def fix_zero_frames(x: np.ndarray) -> np.ndarray:
    """全零帧（检测丢失）→ 前后非零帧线性插值。"""
    T, V, C = x.shape
    zero = np.abs(x).sum(axis=(1, 2)) == 0
    if not zero.any() or zero.all():
        return x
    idx = np.arange(T)
    good = idx[~zero]
    out = x.copy()
    for c in range(C):
        for v in range(V):
            col = x[:, v, c]
            if np.abs(col).sum() == 0:      # 该关节全程零（非待插值），跳过
                continue
            out[:, v, c] = np.interp(idx, good, col[good])
    return out


def fix_speed_outliers(x: np.ndarray, thresh_q: float = 0.995,
                       max_replace: float = 0.05) -> np.ndarray:
    """速度（帧间位移）超阈值的帧 → 用局部中值替换该帧。

    检测错误帧表现为瞬时速度尖峰。thresh_q: 全局速度分位（默认 0.995 只动最尖的 0.5% 帧）。
    """
    T = x.shape[0]
    if T < 3:
        return x
    vel = np.linalg.norm(np.diff(x, axis=0), axis=-1)          # [T-1, V]
    thr = np.percentile(vel, thresh_q * 100)
    if thr <= 0:
        return x
    bad = (vel > thr) & (vel > np.maximum(thr, 0.08))          # 硬下限防误伤
    bad_frames = np.union1d(np.where(bad.any(axis=1))[0] + 1,
                            np.where(bad.any(axis=1))[0])      # 尖峰前后帧
    out = x.copy()
    n_bad = len(bad_frames)
    if n_bad > T * max_replace:
        return x                                                # 太烂，放弃（坏 clip 由诊断标记）
    for t in bad_frames:
        lo, hi = max(0, t - 2), min(T, t + 3)
        out[t] = np.median(x[lo:hi], axis=0)
    return out


def fix_shoulder_outliers(x: np.ndarray, thresh: float = 0.10) -> np.ndarray:
    """肩宽（关节 11-14）与时间中值偏差 > thresh 的帧 → 局部中值替换（整帧 scale 异常）。"""
    T = x.shape[0]
    if T < 3:
        return x
    sw = np.linalg.norm(x[:, 11] - x[:, 14], axis=-1)
    med = np.median(sw)
    if med <= 0:
        return x
    bad = np.abs(sw - med) > thresh * med
    if not bad.any():
        return x
    out = x.copy()
    for t in np.where(bad)[0]:
        lo, hi = max(0, t - 2), min(T, t + 3)
        out[t] = np.median(x[lo:hi], axis=0)
    return out


def clean_skeleton(x: np.ndarray, smooth: bool = True, speed: bool = True,
                   zero: bool = True, shoulder: bool = True,
                   med_w: int = 5, sigma: float = 1.0,
                   thresh_q: float = 0.995) -> np.ndarray:
    """组合清洗：零帧插值 → 速度异常 → 肩宽漂移 → 平滑。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] < 3:
        return x
    if zero:
        x = fix_zero_frames(x)
    if speed:
        x = fix_speed_outliers(x, thresh_q=thresh_q)
    if shoulder:
        x = fix_shoulder_outliers(x)
    if smooth:
        x = smooth_time(x, med_w=med_w, sigma=sigma)
    return x


if __name__ == "__main__":
    # 自测：造一个带噪声序列，验证清洗后速度尖峰被压低
    rng = np.random.default_rng(0)
    T = 40
    base = np.zeros((T, 17, 3), np.float32)
    base[:, :, 2] = np.linspace(0.3, 1.0, T)[:, None]          # 慢速上升（z）
    base[:, 15, 0] = np.sin(np.linspace(0, 2 * np.pi, T)) * 0.1  # 手臂摆动
    noisy = base.copy()
    noisy[10] += np.array([0.8, -0.5, 0.4])                     # 尖峰帧（检测错误）
    noisy[12] = 0                                               # 全零帧
    clean = clean_skeleton(noisy)
    v_old = np.linalg.norm(np.diff(noisy, axis=0), axis=-1)
    v_new = np.linalg.norm(np.diff(clean, axis=0), axis=-1)
    print(f"max vel 原始={v_old.max():.3f} → 清洗后={v_new.max():.3f}")
    print(f"全零帧 原始={(np.abs(noisy).sum((1,2))==0).sum()} → 清洗后={(np.abs(clean).sum((1,2))==0).sum()}")
    assert v_new.max() < v_old.max() and (np.abs(clean).sum((1, 2)) != 0).all()
    print("skeleton_clean 自测 OK")
