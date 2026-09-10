#!/usr/bin/env python3
"""CUHK-X —— IMU 姿态角积分（用户方向：IMU 积分工程方法，对抗 1.5× 幅值域差）。

原理：
  WitMotion IMU 输出欧拉角有 ±180° 环绕跳变（177°→-178°，meanΔ~82°假跳变），
  直接做特征会被污染。且 acc/gyro 幅值在 train/test 有 ~1.9× 域差（train大）。

  方案：角速度(gyro, °/s) 时间积分 → 连续相对姿态角位移（累积旋转量）。
  - 对时间步 dt 积分：∠_n = ∠_{n-1} + gyro·dt
  - 结果 = 无环绕的连续旋转轨迹，域差从幅值缩放 → 旋转角速度-时间包络
  - 下游用「增量/变化率/累积包络度」等比例不变特征，天然抵抗恒幅缩放

接口：
  imu_pose_integ(imu_dir, T) -> [T, 30]  (与 IMUDataset 相同 5dev×6ch 布局,
                                    但 6ch = [gyro原始 3 + 积分姿态角增量 3])
  pose_regress_stats(imu_dir) -> 诊断统计（域鲁棒性/跳变检查）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.imu_dataset import COL_ACC, COL_GYRO, DEVICE_ORDER, _device_of, _parse_time

# 完整列映射（补欧拉角英文名，供可选使用）
_COL_POSE_MAP = {
    "时间": "时间", "time": "时间",
    "设备名称": "设备名称", "DeviceName": "设备名称",
    "角速度X(°/s)": "gyroX", "角速度Y(°/s)": "gyroY", "角速度Z(°/s)": "gyroZ",
    "AsX(°/s)": "gyroX", "AsY(°/s)": "gyroY", "AsZ(°/s)": "gyroZ",
    "AsX (°/s)": "gyroX", "AsY (°/s)": "gyroY", "AsZ (°/s)": "gyroZ",
    "角度X(°)": "angX", "角度Y(°)": "angY", "角度Z(°)": "angZ",
    "AngleX(°)": "angX", "AngleY(°)": "angY", "AngleZ(°)": "angZ",
}


def _load_sorted_per_device(imu_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """读 down/up，返回 {设备名: (t_sec[N], feat[N,6])} acc3+gyro3，组内按时间排序。"""
    dev_data = {}
    for fname in ("down(LL+RL).csv", "up(LA+RA+C).csv"):
        f = imu_dir / fname
        if not f.is_file():
            continue
        df = pd.read_csv(f, encoding="utf-8-sig")
        if df.empty:
            continue
        df = df.rename(columns={c: _COL_POSE_MAP.get(str(c).strip(), str(c).strip()) for c in df.columns})
        if "设备名称" not in df.columns or "时间" not in df.columns \
                or not all(c in df.columns for c in ("gyroX", "gyroY", "gyroZ")):
            continue
        for dev, grp in df.groupby("设备名称"):
            dev_name = _device_of(dev)
            if dev_name not in DEVICE_ORDER:
                continue
            t = grp["时间"].apply(_parse_time).to_numpy(np.float64)
            gyro = grp[["gyroX", "gyroY", "gyroZ"]].to_numpy(np.float32)
            order = np.argsort(t)
            t, gyro = t[order], gyro[order]
            if dev_name in dev_data:
                pt, pg = dev_data[dev_name]
                dev_data[dev_name] = (np.concatenate([pt, t]), np.concatenate([pg, gyro]))
            else:
                dev_data[dev_name] = (t, gyro)
    return dev_data


def _wrap_to_continuous(ang: np.ndarray) -> np.ndarray:
    """把 [-180,180] 环绕欧拉角展开为连续角度（补 ±360 跳变）。ang: [N,3]。"""
    ang = np.asarray(ang, np.float64) % 360.0
    ang[ang > 180.0] -= 360.0
    out = np.array(ang, copy=True)
    for i in range(3):
        delta = np.diff(out[:, i])
        # 环绕点：单步跳变 > 180 → 平移 ±360 补偿
        jumps = np.where(delta > 180, delta - 360, np.where(delta < -180, delta + 360, delta))
        out[1:, i] = out[0, i] + np.cumsum(jumps)
    return out


def _integrate_gyro(t: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """gyro 时间积分 → 连续相对姿态角。返回 (dt_used, relative_angle[N,3]) 单位°。"""
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-4)  # 防 0
    # 用每个采样点的 gyro × 间隔，累积
    rel = np.zeros_like(gyro, np.float64)
    rel[1:] = np.cumsum(gyro[1:] * dt[:, None], axis=0)
    return dt, rel


def imu_pose_integ(imu_dir: Path, T: int = 128) -> np.ndarray:
    """产出 [T, 30]：5 设备 × [gyro(3), 姿态角增量(3)]。

    姿态角增量来自 gyro 积分（连续，无环绕）。比原始欧拉角鲁棒。
    """
    dev_data = _load_sorted_per_device(imu_dir)
    out = np.zeros((T, len(DEVICE_ORDER) * 6), np.float32)
    if not dev_data:
        return out
    # 全局时间轴对齐（同一 clip 内）
    all_t = np.concatenate([d[0] for d in dev_data.values()])
    if all_t.size < 2:
        return out
    t_min, t_max = all_t.min(), all_t.max()
    if t_max - t_min < 1e-6:
        return out
    grid = np.linspace(0, 1.0, T)
    for di, dev in enumerate(DEVICE_ORDER):
        if dev not in dev_data:
            continue
        t, gyro = dev_data[dev]
        _, rel = _integrate_gyro(t, gyro)
        # rel 累积到末尾最大 → 每个时间点重基准化: 用「截至该点的累积旋转」的差分特征
        # 直接在原始时间轴上重采样 rel 和 gyro
        tn = (t - t_min) / (t_max - t_min)
        feat_gyro = np.stack([
            np.interp(grid, tn, gyro[:, j]) for j in range(3)
        ], -1)  # [T,3]
        feat_rel = np.stack([
            np.interp(grid, tn, rel[:, j]) for j in range(3)
        ], -1)  # [T,3] 累积姿态
        out[:, di * 6:(di + 1) * 6] = np.concatenate([feat_gyro, feat_rel.astype(np.float32)], -1)
    return out


def pose_regress_stats(imu_dir: Path) -> Dict:
    """诊断：gyro/积分姿态的域统计。返回 dict 或 None。"""
    dev_data = _load_sorted_per_device(imu_dir)
    if not dev_data:
        return None
    gyro_s = []
    rel_s = []
    for dev in DEVICE_ORDER:
        if dev in dev_data:
            t, gyro = dev_data[dev]
            if len(gyro) < 2:
                continue
            _, rel = _integrate_gyro(t, gyro)
            gyro_s.append(gyro.std(0).mean())
            rel_s.append(np.abs(np.diff(rel, axis=0)).mean(0).mean())  # 平均角速度(积分后变化率)
    if not gyro_s:
        return None
    return dict(gyro_std=np.mean(gyro_s), rel_rate=np.mean(rel_s),
                ndev=len(gyro_s), nframes=sum(len(dev_data[d][0]) for d in DEVICE_ORDER if d in dev_data))


if __name__ == "__main__":
    import argparse
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="domain", choices=["domain", "dump"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()

    if args.mode == "domain":
        # 域鲁棒性对比：gyro 幅值 vs 积分姿态变化率（train/test 比，近 1=鲁棒）
        ts = []
        tr = []
        tdir = Path("data/Testing/data/small_model_track_test")
        for sd in sorted(list(tdir.iterdir())):
            s = pose_regress_stats(sd / "IMU")
            if s:
                ts.append(s)
        for p in glob.glob("data/Training/HAR/IMU/*/user*/[0-9]*"):
            s = pose_regress_stats(Path(p))
            if s:
                tr.append(s)
        tg = np.array([s["gyro_std"] for s in ts]); trg = np.array([s["gyro_std"] for s in tr])
        trl = np.array([s["rel_rate"] for s in ts]); trr = np.array([s["rel_rate"] for s in tr])
        print(f"n_train={len(tr)} n_test={len(ts)}")
        print(f"gyro幅值   mean train={trg.mean():.3f} test={tg.mean():.3f} 比(train/test)={trg.mean()/tg.mean():.3f}")
        print(f"积分姿态率 mean train={trr.mean():.3f} test={trl.mean():.3f} 比(train/test)={trr.mean()/trl.mean():.3f}")
        nd = [s["ndev"] for s in ts]
        print(f"test 设备数: {np.unique(nd, return_counts=True)}")
    elif args.mode == "dump":
        clip = Path(args.clip)
        s = pose_regress_stats(clip)
        print(f"stats {clip}: {s}")
        x = imu_pose_integ(clip, T=128)
        print(f"feature shape {x.shape} 非零帧 {np.count_nonzero(x.any(-1))}/128")
