#!/usr/bin/env python3
"""域根因检验(train vs test): IMU(真实管道)/Depth图像/bbox人物占比/骨架 逐模态硬统计。"""
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.imu_dataset import load_imu_sequence, time_align


def probe_imu(gl, tag):
    ns, g_m, a_m, T = [], [], [], []
    files = random.sample([str(f) for f in sorted(glob.glob(gl))][:200], min(140, 200))
    for f in files:
        try:
            dev = load_imu_sequence(Path(f))
            x = time_align(dev, T=128)
            if x is None or not np.any(x):
                continue
            gyro = x[:, 3::6]
            acc = x[:, 0::6]
            g_m.append(np.abs(gyro).mean()); a_m.append(np.abs(acc).mean())
            ns.append(1)
        except Exception:
            continue
    if g_m:
        print(f"[{tag}] IMU(align128) gyro|均值={np.mean(g_m):.2f}  acc|均值={np.mean(a_m):.3f}  n={len(g_m)}")


def probe_img(gl, tag):
    cand = sorted(glob.glob(gl))
    vals = []
    files = random.sample(cand, min(50, len(cand))) if cand else []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        vals.append(im.mean() / 255.0)
    if vals:
        print(f"[{tag}] Depth 灰度均值={np.mean(vals):.3f}  n={len(vals)}")


def probe_bbox(tag, bj):
    try:
        js = json.load(open(bj))
        areas, ars = [], []
        for v in js.values():
            x1, y1, x2, y2 = v[:4]
            w, h = x2 - x1, y2 - y1
            areas.append(w * h / (640 * 480))
            ars.append(w / max(h, 1))
        print(f"[{tag}] bbox 人物面积占比={np.mean(areas):.4f} 宽高比={np.mean(ars):.3f} n={len(areas)}")
    except Exception as e:
        print(f"[{tag}] bbox err {e}")


def probe_skel(gl, tag):
    sp = []
    for d in random.sample(sorted(glob.glob(gl))[:400], 60):
        fs = sorted(Path(d).glob("*.json"))
        if len(fs) < 3:
            continue
        pts = []
        for f in fs[:12]:
            try:
                o = json.loads(f.read_text())
                fr = o if isinstance(o, dict) else o[0]
                kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
                pts.append(kp[0, :2])
            except Exception:
                pass
        if len(pts) > 2:
            p = np.array(pts)
            sp.append(np.abs(np.diff(p, axis=0)).mean())
    if sp:
        print(f"[{tag}] 骨�骨盆速度={np.mean(sp):.4f} n={len(sp)}")


if __name__ == "__main__":
    random.seed(0)
    probe_imu("data/Training/HAR/IMU/*/*/*", "train")
    probe_imu("data/Testing/data/small_model_track_test/*/IMU", "test")
    probe_img("data/Training/HAR/Depth_Color/*/*/*/*.png", "train")
    probe_img("data/Testing/data/small_model_track_test/*/Depth_Color/*.png", "test")
    probe_bbox("train", "bbox_train.json")
    probe_bbox("test", "bbox_test.json")
    probe_skel("data/Training/HAR/Skeleton/*/*/*/predictions", "train")
    probe_skel("data/Testing/data/small_model_track_test/*/Skeleton/predictions", "test")
