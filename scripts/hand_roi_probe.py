#!/usr/bin/env python3
"""skeleton-guided 手部 ROI 可行性探针: 用骨架腕点(4右/7左)裁手部窗, 看孪生对手部窗是否有像素信息。
对 (18写,17键)/(7吃,6喝)/(22翻,21读): 裁 80px 手部窗 → 统计窗内梯度能量 + 存图供人工核对。
"""
import json
import glob
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.skeleton_dataset import frame_num_of

OUT = Path("outputs/hand_roi")
OUT.mkdir(parents=True, exist_ok=True)


def skeleton_for(clip_key, sample_dir):
    sk_root = Path("data/Training/HAR/Skeleton") / Path(sample_dir).parents[1].name \
        / Path(sample_dir).name.split("_")[0]  # rough
    return None


def hand_window(depth_dir, skel_pred_dir):
    fs = sorted(Path(skel_pred_dir).glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    if len(fs) < 2:
        return None, None
    # 中间帧骨架
    d = json.loads(fs[len(fs) // 2].read_text("utf-8"))
    fr = d if isinstance(d, dict) else d[0]
    kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)  # [x,y,z]
    imgs = sorted(depth_dir.glob("*.png"))
    if not imgs:
        return None, None
    im = cv2.imread(str(imgs[len(imgs) // 2]), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    H, W = im.shape
    # 手: 腕点(右4,左7)转像素(注意骨架坐标系与图像尺度: 取两腕中点)
    ws = [kp[4, :2], kp[7, :2]]
    cx = int(np.mean([w[0] for w in ws]) * W)
    cy = int(np.mean([w[1] for w in ws]) * H)
    h = 90
    x1, y1 = max(cx - h, 0), max(cy - h, 0)
    crop = im[y1:y1 + 2 * h, x1:x1 + 2 * h]
    if crop.size == 0:
        return None, None
    energ = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1))
    return crop, energ.mean()


def main():
    import collections
    stats = collections.defaultdict(list)
    pairs = [(18, 17), (7, 6), (22, 21), (13, 12)]
    for a, b in pairs:
        for cls in (a, b):
            sk_dirs = glob.glob(f"data/Training/HAR/Skeleton/{cls}_*/user*/1-*/predictions")
            n = 0
            for sd in sorted(sk_dirs)[:120]:
                if n >= 6:
                    break
                # 对上深度目录
                parts = Path(sd).parts
                # Skeleton/<cls>/user/sample/predictions → Depth_Color/<cls>/user/sample
                rel = "/".join(parts[-3:])[:-len("/predictions")]
                depth_dir = Path("data/Training/HAR/Depth_Color") / rel
                if not depth_dir.exists():
                    continue
                crop, en = hand_window(depth_dir, Path(sd))
                if crop is None:
                    continue
                stats[(cls, a, b)].append(en)
                if cls == a and len(crop) > 0:
                    cv2.imwrite(str(OUT / f"{a}_{b}_cls{a}_{n}.png"), crop)
                    n += 1
    print("手部窗梯度能量(细节量) 均值:")
    for key in stats:
        cls, a, b = key
        vs = stats[key]
        print(f"  孪生对({a},{b}) cls{cls}: energy={np.mean(vs):.0f} (n={len(vs)})")
    print(f"图存 {OUT}/")


if __name__ == "__main__":
    main()
