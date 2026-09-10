"""CUHK-X —— Radar 点云数据加载(异源摸底第3模态, 与 IMU 同构)。

Radar 格式实测:
- data/Training/HAR/Radar/<Action>/<Subject>/<sample>/radar_output_*.csv  (同 IMU 路径层级)
- 每行 = 一个检测目标点: timestamp, frame, DetObj#, x, y, z, v, snr, noise  (点云, 非固定通道时序)
- ~47% 文件为 0 行(空), 非空 79~800+ 行, avg 34 帧/clip
特征: 逐帧聚合 → [T, 13] = [n, v_mean, v_std, v_max, snr_mean, snr_max,
       x_mean, x_std, y_mean, y_std, z_mean, z_std, noise_mean]
空/缺 → 全 0 (zero-pad, 与 IMU 缺失同处理)
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEAT_DIM = 13
T_DEFAULT = 64


class RadarClipIndex:
    def __init__(self, action_id: int, subject: str, sample: str, radar_csv: Path):
        self.action_id = action_id
        self.subject = subject
        self.sample = sample
        self.radar_csv = radar_csv


def build_radar_index(train_root: Path, main_clips: list) -> List[RadarClipIndex]:
    """从 main clips(Depth discovery)构造 Radar csv 路径, 保证与 main 同一 subject split。
    main_clips 的 c.depth_dir.parent.parent.name = <Action> 目录名(与 IMU 一致)。"""
    root = Path(train_root)
    clips = []
    for c in main_clips:
        action_dir = c.depth_dir.parent.parent.name
        d = root / "Radar" / action_dir / c.subject / c.sample
        csvs = sorted(glob.glob(str(d / "radar_output_*.csv")))
        clips.append(RadarClipIndex(c.action_id, c.subject, c.sample,
                                    Path(csvs[0]) if csvs else None))
    return clips


_FEAT_COLS = {"v", "snr", "x", "y", "z", "noise"}


def _aggregate_clip(csv_path: Path, T: int) -> np.ndarray:
    """读 radar csv → [T, FEAT_DIM]。空/异常 → 全 0。"""
    out = np.zeros((T, FEAT_DIM), np.float32)
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "frame" not in df.columns:
            return out
        grp = df.groupby("frame")
        rows = []
        for _, g in grp:
            n = len(g)
            rec = [n]
            for c in ["v", "snr", "x", "y", "z"]:
                a = g[c].to_numpy(np.float32)
                rec += [a.mean(), a.std(), a.max()]
            rec += [g["noise"].mean()]
            rows.append(rec)
        if not rows:
            return out
        arr = np.array(rows, np.float32)          # [T_in, 13]
        T_in = arr.shape[0]
        if T_in != T:
            t_src = np.linspace(0, 1, T_in)
            t_dst = np.linspace(0, 1, T)
            arr_r = np.zeros((T, FEAT_DIM), np.float32)
            for c in range(FEAT_DIM):
                if T_in == 1:
                    arr_r[:, c] = arr[0, c]
                else:
                    arr_r[:, c] = np.interp(t_dst, t_src, arr[:, c])
            arr = arr_r
        # 归一化: 标准尺度(量纲差不多的量级) —— speed/snr/x 归一化到 ~[-1,1]
        SCALE = np.array([20, 0.5, 0.5, 0.5, 50, 50, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20],
                         np.float32)
        arr = arr / SCALE[None, :]
        out[...] = arr
    except Exception:
        pass
    return out


class RadarDataset(Dataset):
    def __init__(self, clips: List[RadarClipIndex], T: int = T_DEFAULT,
                 is_train: bool = False, seed: int = 0):
        self.clips = clips
        self.T = T
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        feat = _aggregate_clip(clip.radar_csv, self.T) if clip.radar_csv else np.zeros(
            (self.T, FEAT_DIM), np.float32)
        if self.is_train:
            # 轻量增强: 高斯噪声 + 时间缩放(仿 IMU jitter)
            if self.rng.random() < 0.5:
                feat = feat + self.rng.normal(0, 0.02, feat.shape).astype(np.float32)
            if self.rng.random() < 0.3:
                sub = self.rng.uniform(0.8, 1.0)
                n2 = max(int(self.T * sub), 2)
                s0 = self.rng.integers(0, max(self.T - n2, 1))
                seg = feat[s0:s0 + n2]
                feat = np.array([
                    np.interp(np.linspace(0, len(seg) - 1, self.T), np.arange(len(seg)), seg[:, c])
                    for c in range(FEAT_DIM)], np.float32).T
        return (torch.from_numpy(np.ascontiguousarray(feat, np.float32)),
                clip.action_id, clip.subject)
