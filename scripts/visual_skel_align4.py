#!/usr/bin/env python3
"""v4: 用有 bbox 的 clip(27/user1/5-1-2) 重做:
 上=Depth帧+真实bbox(黄)+镜像后红骨架; 下=3正交视图 heatmap(已镜像)。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

CLS, SUB, SMP = 27, "user1", "5-1-2"
SKEL = Path(f"data/Training/HAR/Skeleton/{CLS}_{'Take_a_selfie'}_{SUB}"[:0] or f"data/Training/HAR/Skeleton/27_Take_a_selfie/{SUB}/{SMP}/predictions")
DEPT = Path(f"data/Training/HAR/Depth_Color/27_Take_a_selfie/{SUB}/{SMP}")


def ts_of(name):
    m = re.search(r"(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.\d+)", str(name))
    return m.group(1) if m else ""


def cmp(a, b):
    from datetime import datetime
    try:
        return (datetime.strptime(a, "%Y-%m-%d_%H-%M-%S.%f") -
                datetime.strptime(b, "%Y-%m-%d_%H-%M-%S.%f")).total_seconds()
    except Exception:
        return 1e9


def main():
    bbox = json.load(open("bbox_train.json"))[f"{CLS}/{SUB}/{SMP}"]
    sk = sorted(SKEL.glob("*.json"), key=lambda f: ts_of(f.name))
    dp = sorted(DEPT.glob("*.png"), key=lambda f: ts_of(f.name))
    fi = len(sk) // 2
    tgt = ts_of(sk[fi].name)
    di = min(range(len(dp)), key=lambda i: abs(cmp(ts_of(dp[i].name), tgt)))
    img = cv2.imread(str(dp[di]))
    H, W = img.shape[:2]
    o = json.loads(sk[fi].read_text("utf-8"))
    kp = np.asarray((o if isinstance(o, dict) else o[0])["keypoints"], np.float32).reshape(17, 3)

    x1, y1, x2, y2 = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
    bx, by = (x1 + x2) / 2, y2
    bw, bh = x2 - x1, y2 - y1
    zj = kp[:, 2]; xj = kp[:, 0]
    vh = 0.85 * bh / max(zj.max() - zj.min(), 1e-3)
    vx = 0.6 * bw / max(xj.max() - xj.min(), 0.15)
    pix = np.stack([bx - (xj - xj.mean()) * vx,      # 镜像: 世界x与图像x反向
                    by - (zj - zj.min()) * vh], -1).astype(np.int32)
    canvas = img.copy()
    lines = [(0, 8), (8, 9), (8, 11), (8, 14), (11, 12), (12, 13), (14, 15), (15, 16),
             (9, 10), (10, 7), (9, 6), (2, 3), (3, 4), (5, 6), (5, 2), (0, 2), (0, 5)]
    for a, b in lines:
        cv2.line(canvas, tuple(pix[a]), tuple(pix[b]), (255, 255, 255), 4)
        cv2.line(canvas, tuple(pix[a]), tuple(pix[b]), (0, 0, 255), 2)
    for j in range(17):
        cv2.circle(canvas, tuple(pix[j]), 4, (0, 255, 0), -1)
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    cv2.putText(canvas, f"bbox={CLS}/{SUB}/{SMP}", (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    from src.dataset import _skeleton_heatmap
    hm = _skeleton_heatmap(SKEL, T=16, S=128, box=bbox)
    names = ["正面[z,x](镜像)", "侧面[z,y]", "俯视[x,y](镜像)"]
    w = W // 3
    panel = np.zeros((H, W, 3), np.uint8)
    for v in range(3):
        imv = np.uint8(np.clip(hm[8, v], 0, 1) * 255)
        imv = cv2.resize(cv2.cvtColor(imv, cv2.COLOR_GRAY2BGR), (w, H))
        cv2.putText(imv, names[v], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        panel[:, v * w:(v + 1) * w] = imv
    out = np.vstack([cv2.resize(canvas, (W, H)), panel]).astype(np.uint8)
    cv2.putText(out, "上=Depth+bbox(黄)+骨架(红/绿点已镜像)  下=3view heatmap", (8, H * 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite("outputs/skel_align4.png", out)
    print("saved outputs/skel_align4.png", img.shape)


if __name__ == "__main__":
    main()
