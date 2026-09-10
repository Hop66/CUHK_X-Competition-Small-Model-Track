#!/usr/bin/env python3
"""M1 v3 可视: 3 正交视图 heatmap stack(正[z,x]/侧[z,y]/俯[x,y]) + Depth帧投影。确认 3D 保留。"""
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
    sk = sorted(SKEL.glob("*.json"), key=lambda f: ts_of(f.name))
    dp = sorted(DEPT.glob("*.png"), key=lambda f: ts_of(f.name))
    fi = len(sk) // 2
    tgt = ts_of(sk[fi].name)
    di = min(range(len(dp)), key=lambda i: abs((ts_of(dp[i].name) > tgt) - (ts_of(dp[i].name) < tgt)) if False else 1e9) or 0
    di = min(range(len(dp)), key=lambda i: abs(weighted(ts_of(dp[i].name), tgt)))
    img = cv2.imread(str(dp[di]))
    H, W = img.shape[:2]
    o = json.loads(sk[fi].read_text("utf-8"))
    kp = np.asarray((o if isinstance(o, dict) else o[0])["keypoints"], np.float32).reshape(17, 3)

    from src.dataset import _skeleton_heatmap
    hm = _skeleton_heatmap(SKEL, T=16, S=128, box=bbox)   # [T,3,S,S]
    names = ["正面[z,x]", "侧面[z,y]", "俯视[x,y]"]
    w = W // 3
    panel = np.zeros((H, W, 3), np.uint8)
    for v in range(3):
        imv = np.uint8(np.clip(hm[8, v], 0, 1) * 255)
        imv = cv2.resize(cv2.cvtColor(imv, cv2.COLOR_GRAY2BGR), (w, H))
        cv2.putText(imv, names[v], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        panel[:, v * w:(v + 1) * w] = imv
    out = np.vstack([cv2.resize(img, (W, H)), panel]).astype(np.uint8)
    cv2.putText(out, "上=Depth帧(bbox黄框+骨架红)  下=3正交视图 heatmap(3D保留)", (8, H * 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite("outputs/skel_align3.png", out)
    print("saved outputs/skel_align3.png")


def weighted(a, b):
    from datetime import datetime
    try:
        da = datetime.strptime(a, "%Y-%m-%d_%H-%M-%S.%f")
        db = datetime.strptime(b, "%Y-%m-%d_%H-%M-%S.%f")
        return (da - db).total_seconds()
    except Exception:
        return 1e9


if __name__ == "__main__":
    main()
