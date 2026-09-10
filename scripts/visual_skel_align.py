#!/usr/bin/env python3
"""可视化骨架对齐: 同一 clip 的
 A) Depth 原始帧  B) 当前 heatmap(归一化骨架空间, 非对齐)  C) 3D米→正视图投影叠加到帧上
以确认 M1 的 heatmap 到底和图像对不对齐。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.skeleton_dataset import frame_num_of

SKEL = Path("data/Training/HAR/Skeleton/27_Take_a_selfie/user1/5-1-1/predictions")
DEPT = Path("data/Training/HAR/Depth_Color/27_Take_a_selfie/user1/5-1-1")


def load_mid():
    fs = sorted(SKEL.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    fi = len(fs) // 2
    o = json.loads(fs[fi].read_text("utf-8"))
    fr = o if isinstance(o, dict) else o[0]
    kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
    # 对应中间帧 Depth 图
    dgr = sorted(list(DEPT.glob("*.png")))
    if abs(len(fs) - len(dgr)) > 5:
        dgr = sorted(list(DEPT.glob("*.png")))
    idx = min(len(dgr) - 1, int(fi * len(dgr) / max(len(fs), 1)))
    img = cv2.imread(str(dgr[idx]))
    return kp, img, dgr[idx].name


def skeleton_lines():
    return [(0, 8), (8, 9), (8, 11), (8, 14), (11, 12), (12, 13), (14, 15), (15, 16),
            (9, 10), (10, 7), (9, 6), (2, 3), (3, 4), (5, 6), (5, 2)]


def main():
    kp, img, fname = load_mid()
    H, W = img.shape[:2]
    # C) 3D 米 → 正视图投影: X→水平, -Y→垂直(上), 忽略 z; 躯干(~颈-盆 Y差) 定标
    # pelvis(0)-neck(8)
    dy_neck = kp[0, 1] - kp[8, 1]
    K = 110.0 / max(abs(dy_neck), 1e-3)     # 使颈-盆投影长 ~110px
    cx, cy = W / 2, H - 40
    p2 = np.stack([cx - kp[:, 1] * K,      # 用 Y 当左右? 3D: x,y,z 语义是 水平/深/垂直? 之前 ds 注释: [水平,深度,垂直]=[0,1,2] → 水平=x, 垂直=z, 深度=y
                   cy - kp[:, 2] * K], -1).astype(np.int32)
    # 水平用 x, 垂直用 z(按 [水平,深度,垂直] 语义)
    p2_ = np.stack([cx + kp[:, 0] * K, cy - kp[:, 2] * K], -1).astype(np.int32)
    canvas = img.copy()
    for a, b in skeleton_lines():
        cv2.line(canvas, tuple(p2_[a]), tuple(p2_[b]), (0, 0, 255), 2)
    for j in range(17):
        cv2.circle(canvas, tuple(p2_[j]), 3, (255, 255, 0), -1)
    # B) 当前 heatmap(独立 128×128 骨架空间)
    from src.dataset import _skeleton_heatmap
    hm = _skeleton_heatmap(SKEL, T=16, S=128)
    hmv = np.uint8(hm[8, 0] * 255)
    hmv = cv2.resize(hmv, (W // 2, H))
    hmc = cv2.cvtColor(hmv, cv2.COLOR_GRAY2BGR)
    out = np.vstack([np.hstack([img, hmc]),
                     np.zeros((H // 4, W + W // 2, 3))]).astype(np.uint8)
    cv2.putText(out, f"L:Depth帧+3D正投影(红)  R_top:当前heatmap(独立空间)  {fname}",
                (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/skel_align_check.png", out)
    print("saved outputs/skel_align_check.png", "img", img.shape)


if __name__ == "__main__":
    main()
