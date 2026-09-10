#!/usr/bin/env python3
"""v2 修正可视化: 骨架 3D 语义 = [x水平, y深度, z垂直(z∈[0,~1.4]~身高);
  用正确轴 [x,z] + bbox 窗 + 帧时间对齐 画投影; 对比修复后的 heatmap 是否"立起来贴人"。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

SKEL = Path("data/Training/HAR/Skeleton/27_Take_a_selfie/user1/5-1-1/predictions")
DEPT = Path("data/Training/HAR/Depth_Color/27_Take_a_selfie/user1/5-1-1")


def ts_of(name):
    m = re.search(r"(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.\d+)", str(name))
    return m.group(1) if m else ""


def main():
    bbox = json.load(open("bbox_train.json")).get("0/user16/1-1-1")
    if bbox is None:
        bbox = json.load(open("bbox_train.json")).get("27/user16/1-1-1")
    sk = sorted(SKEL.glob("*.json"), key=lambda f: ts_of(f.name))
    dp = sorted(DEPT.glob("*.png"), key=lambda f: ts_of(f.name))
    # 取骨架中段帧; 用时间戳匹配最近 Depth 帧
    fi = len(sk) // 2
    tgt = ts_of(sk[fi].name)
    di = min(range(len(dp)), key=lambda i: abs(compare(ts_of(dp[i].name), tgt))) if dp else 0
    img = cv2.imread(str(dp[di]))
    H, W = img.shape[:2]
    o = json.loads(sk[fi].read_text("utf-8"))
    fr = o if isinstance(o, dict) else o[0]
    kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)  # [x,y,z]

    # bbox pixel
    x1, y1, x2, y2 = [v * (W if i % 2 == 0 else H) for i, v in enumerate(bbox)]
    bx, by, bw, bh = (x1 + x2) / 2, y2, (x2 - x1), (y2 - y1)

    zj = kp[:, 2]; xj = kp[:, 0]
    zm = zj.min(); zM = zj.max()
    # 垂直: z→bbox 高(脚在底, 头在顶); 水平: x→bbox 中 60% 宽(以骨盆/均值定中)
    vh = 0.85 * bh / max(zM - zm, 1e-3)
    xm = float(np.mean(xj))
    vx = 0.6 * bw / max((xj.max() - xj.min()), 0.15)
    p2 = np.stack([bx + (xj - xm) * vx, by - (zj - zm) * vh], -1)
    p2 = p2.astype(np.int32)

    canvas = img.copy()
    lines = [(0, 8), (8, 9), (8, 11), (8, 14), (11, 12), (12, 13), (14, 15), (15, 16),
             (9, 10), (10, 7), (9, 6), (2, 3), (3, 4), (5, 6), (5, 2)]
    for a, b in lines:
        cv2.line(canvas, tuple(p2[a]), tuple(p2[b]), (255, 255, 255), 4)
        cv2.line(canvas, tuple(p2[a]), tuple(p2[b]), (0, 0, 255), 2)
    for j in range(17):
        cv2.circle(canvas, tuple(p2[j]), 4, (255, 255, 0), -1)
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    # 修复轴后的 heatmap([x,z] 画 128)
    from src.dataset import _skeleton_heatmap
    hm = _skeleton_heatmap(SKEL, T=16, S=128, box=bbox)
    hmv = np.uint8(np.clip(hm[8, 0], 0, 1) * 255)
    hmv = cv2.resize(cv2.cvtColor(hmv, cv2.COLOR_GRAY2BGR), (W // 2, H // 2))
    out = np.vstack([cv2.hconcat([cv2.resize(canvas, (W // 2, H * 2 // 3)),
                                  cv2.resize(hmv, (W // 4, H * 2 // 3))]),
                     np.zeros((60, W * 3 // 4, 3))]).astype(np.uint8)
    cv2.putText(out, "L: bbox框(黄)+3D[x,z]投影(红线白描)  R: 修复轴 heatmap", (5, out.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/skel_align2.png", out)
    print("saved outputs/skel_align2.png | skel ts=", tgt, "depth ts=", ts_of(dp[di].name))


def compare(a, b):
    from datetime import datetime
    try:
        da = datetime.strptime(a, "%Y-%m-%d_%H-%M-%S.%f")
        db = datetime.strptime(b, "%Y-%m-%d_%H-%M-%S.%f")
        return (da - db).total_seconds()
    except Exception:
        return 1e9


if __name__ == "__main__":
    main()
