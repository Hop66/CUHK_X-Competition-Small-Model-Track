"""CUHK-X —— IMU 数据加载（P2：异源惯性摸底）

IMU 格式（实测）：
- 结构 IMU/<Action>/<Subject>/<sample>/{down(LL+RL).csv, up(LA+RA+C).csv}
- down = 左腿+右腿 2 设备；up = 左臂+右臂+躯干 3 设备（共 5 设备）
- 21 列：时间/设备名称/加速度XYZ(g)/角速度XYZ(°/s)/角度XYZ/磁场XYZ/四元数4/温度/版本/电量
- 多设备交错写入、时间戳乱序 → 必须按设备分组 + 组内排序 + 统一时间轴对齐
- 特征：加速度 + 角速度（6 维/设备 × 5 设备 = 30 维/帧），~100Hz

输出：[T, 30]（T 固定采样帧，30 = 5 设备 × 6 通道）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 5 设备固定顺序（WitMotion 命名：LL=左腿 RL=右腿 LA=左臂 RA=右臂 C=躯干）
DEVICE_ORDER = ["WTLL", "WTRL", "WTLA", "WTRA", "WTC"]
NUM_DEVICES = len(DEVICE_ORDER)
CH_PER_DEV = 6  # acc3 + gyro3
FEAT_DIM = NUM_DEVICES * CH_PER_DEV  # 30

COL_ACC = ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]
COL_GYRO = ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]


class IMUClipIndex:
    def __init__(self, action_id: int, subject: str, sample: str, imu_dir: Path):
        self.action_id = action_id
        self.subject = subject
        self.sample = sample
        self.imu_dir = imu_dir


def build_imu_index(train_root: Path, main_clips: Optional[list] = None) -> List[IMUClipIndex]:
    """从 main clips（Depth discovery）构造 IMU 路径，保证与 main 同一 subject split。"""
    root = Path(train_root)
    clips = []
    for c in main_clips:
        imu_dir = root / "IMU" / c.depth_dir.parent.parent.name / c.subject / c.sample
        clips.append(IMUClipIndex(c.action_id, c.subject, c.sample, imu_dir))
    return clips


def _device_of(name: str) -> Optional[str]:
    """从设备名 'WTRL(E5:9E:...)' 提取主体名（去掉 MAC）。"""
    m = re.match(r"([A-Z]+)", str(name))
    return m.group(1) if m else None


def _parse_time(s) -> float:
    """时间 '2025-06-10 10:43:49.390' → 秒（相对 clip 起点由调用方归一化）。"""
    return float(pd.Timestamp(s).timestamp())


# test down 文件是英文列名（DeviceName/AccX.../AsX...），train 是中文 → 统一映射
_COL_MAP = {
    "时间": "时间", "time": "时间",
    "设备名称": "设备名称", "DeviceName": "设备名称",
    "加速度X(g)": "加速度X(g)", "AccX(g)": "加速度X(g)", "AccX (g)": "加速度X(g)",
    "加速度Y(g)": "加速度Y(g)", "AccY(g)": "加速度Y(g)", "AccY (g)": "加速度Y(g)",
    "加速度Z(g)": "加速度Z(g)", "AccZ(g)": "加速度Z(g)", "AccZ (g)": "加速度Z(g)",
    "角速度X(°/s)": "角速度X(°/s)", "AsX(°/s)": "角速度X(°/s)", "AsX (°/s)": "角速度X(°/s)",
    "角速度Y(°/s)": "角速度Y(°/s)", "AsY(°/s)": "角速度Y(°/s)", "AsY (°/s)": "角速度Y(°/s)",
    "角速度Z(°/s)": "角速度Z(°/s)", "AsZ(°/s)": "角速度Z(°/s)", "AsZ (°/s)": "角速度Z(°/s)",
}


def load_imu_sequence(imu_dir: Path) -> np.ndarray:
    """读 down/up 两个 csv，按设备分组 + 时间排序，返回 {设备名: (t_sec, feat[N,6])}。"""
    dev_data = {}  # 设备名 -> (times[], feats[N,6])
    for fname in ("down(LL+RL).csv", "up(LA+RA+C).csv"):
        f = imu_dir / fname
        if not f.is_file():
            continue
        df = pd.read_csv(f, encoding="utf-8-sig")
        if df.empty:
            continue
        # 列名统一（中文/英文兼容；test down 是英文 → 否则整个文件被丢 → 域"伪造"差异）
        df = df.rename(columns={c: _COL_MAP.get(str(c).strip(), str(c).strip()) for c in df.columns})
        if "设备名称" not in df.columns or "时间" not in df.columns \
                or not all(c in df.columns for c in COL_ACC + COL_GYRO):
            continue
        for dev, grp in df.groupby("设备名称"):
            dev_name = _device_of(dev)
            if dev_name not in DEVICE_ORDER:
                continue
            t = grp["时间"].apply(_parse_time).to_numpy()
            acc = grp[COL_ACC].to_numpy(np.float32)
            gyro = grp[COL_GYRO].to_numpy(np.float32)
            feat = np.concatenate([acc, gyro], axis=-1)  # [N, 6]
            # 组内按时间排序
            order = np.argsort(t)
            t, feat = t[order], feat[order]
            if dev_name in dev_data:
                prev_t, prev_f = dev_data[dev_name]
                dev_data[dev_name] = (np.concatenate([prev_t, t]), np.concatenate([prev_f, feat]))
            else:
                dev_data[dev_name] = (t, feat)
    return dev_data


def time_align(dev_data, T: int = 128) -> np.ndarray:
    """把 5 设备对齐到统一时间轴 [0,1]，每帧取各设备最近值 → [T, 30]。"""
    out = np.zeros((T, FEAT_DIM), np.float32)
    if not dev_data:
        return out
    # 全局时间范围
    all_t = np.concatenate([d[0] for d in dev_data.values()])
    if all_t.size < 2:
        return out
    t_min, t_max = all_t.min(), all_t.max()
    if t_max - t_min < 1e-6:
        return out
    grid = np.linspace(0.0, 1.0, T)
    for di, dev in enumerate(DEVICE_ORDER):
        if dev not in dev_data:
            continue
        t, feat = dev_data[dev]
        tn = (t - t_min) / (t_max - t_min)  # 归一化到 [0,1]
        # 最近邻：对每个 grid 点找最近的 tn 索引
        idx = np.clip(np.searchsorted(tn, grid, side="left"), 0, len(tn) - 1)
        idx = np.minimum(idx, len(tn) - 1)
        # 更精确：选左右最近
        idx = np.where((idx > 0) & (np.abs(tn[idx] - grid) > np.abs(tn[idx - 1] - grid)), idx - 1, idx)
        out[:, di * CH_PER_DEV:(di + 1) * CH_PER_DEV] = feat[idx]
    return out


class IMUDataset(Dataset):
    """IMU -> [T, 30]（5 设备 × acc3+gyro3，时间对齐）。"""

    def __init__(self, clips: List[IMUClipIndex], T: int = 128, is_train: bool = True,
                 seed: int = 0, jitter: bool = True):
        self.clips = clips
        self.T = T
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        dev_data = load_imu_sequence(clip.imu_dir)
        feat = time_align(dev_data, self.T)  # [T, 30]
        # 训练增强：时间缩放/噪声（轻量，避免过拟合）
        if self.is_train and self.jitter and dev_data:
            # 时间抖动：随机选时间窗口子段（长度 0.8-1.0 T），再 resize 回 T
            if self.rng.random() < 0.5:
                sub_len = int(self.T * self.rng.uniform(0.7, 1.0))
                start = self.rng.integers(0, max(self.T - sub_len, 1))
                feat = feat[start:start + sub_len]
                feat = np.array([np.interp(np.linspace(0, len(feat) - 1, self.T), np.arange(len(feat)), feat[:, j])
                                 for j in range(FEAT_DIM)]).T
            # 高斯噪声
            if self.rng.random() < 0.5:
                feat = feat + self.rng.normal(0, 0.02, feat.shape).astype(np.float32)
        # 统一转回 float32（np.interp 会升级成 float64）+ 保证连续内存
        x = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32))
        return x, clip.action_id, clip.subject
