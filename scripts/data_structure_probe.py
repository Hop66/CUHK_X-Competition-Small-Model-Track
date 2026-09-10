#!/usr/bin/env python3
"""数据层硬自检: test vs train 骨架/IMU 是否为"文件结构差异→假域差"。
逐项对比: 骨架 json keys/shape/帧数/坐标系; IMU 列名/设备/时长/时间格式。
输出结构差异表; 无差异=真域差(只作训练帮助), 有差异=修读取。
"""
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

rng = random.Random(0)


def skel_struct(gl, tag, n=40):
    ks = set(); shapes = Counter(); frames = []; zrange = []; conf_has = 0; cn = 0
    for d in rng.sample(sorted(glob.glob(gl))[:400], min(n, len(glob.glob(gl)))):
        fs = sorted(Path(d).glob("*.json"))
        frames.append(len(fs))
        if not fs:
            continue
        for f in fs[:3]:
            try:
                o = json.loads(f.read_text("utf-8"))
                fr = o if isinstance(o, dict) else o[0]
                ks.update(fr.keys())
                kp = np.asarray(fr["keypoints"], np.float32)
                shapes[kp.shape] += 1
                zrange.append((kp[..., 0].min(), kp[..., 0].max()))
                if "keypoint_scores" in fr:
                    conf_has += 1
                cn += 1
            except Exception:
                pass
    print(f"[{tag}] json keys={sorted(ks)[:8]} shape分布={dict(list(shapes.items())[:3])} "
          f"帧数mean={np.mean(frames):.0f} x范围=({np.mean([r[0] for r in zrange]):.2f},{np.mean([r[1] for r in zrange]):.2f}) "
          f"有scores比例={conf_has/max(cn,1):.2f}")


def imu_struct(gl_down, tag, n=50):
    cols = Counter(); devs = Counter(); rows = []; tfmt = Counter(); per_file_cols = 0
    files = rng.sample(sorted(glob.glob(gl_down))[:600], min(n, 600))
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", nrows=5)
            cols.update(list(df.columns))
            rows.append(len(pd.read_csv(f, encoding="utf-8-sig", usecols=[0])))
            if "设备名称" in df.columns:
                devs.update(df["设备名称"].str.extract(r"([A-Z]+)")[0].dropna().tolist())
            if "时间" in df.columns:
                t = str(df["时间"].iloc[0])
                tfmt[type(df["时间"].iloc[0]).__name__] += 1
                if not t.startswith("2025"):
                    tfmt[t[:10]] += 1
        except Exception:
            pass
    print(f"[{tag}] 列数={len(cols)} 设备={dict(list(devs.most_common(6)))} "
          f"行数mean={np.mean(rows):.0f} 时间类型={dict(list(tfmt.most_common(2)))}")


if __name__ == "__main__":
    skel_struct("data/Training/HAR/Skeleton/*/*/*/predictions", "train-skel")
    skel_struct("data/Testing/data/small_model_track_test/*/Skeleton/predictions", "test-skel")
    imu_struct("data/Training/HAR/IMU/*/*/*/down*.csv", "train-imu")
    imu_struct("data/Testing/data/small_model_track_test/*/IMU/down*.csv", "test-imu")
