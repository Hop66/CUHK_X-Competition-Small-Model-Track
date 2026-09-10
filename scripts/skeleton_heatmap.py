#!/usr/bin/env python3
"""M1 构建块: 骨架 3D → 2D heatmap 生成(骨盆中心化+bbox 尺度归一化空间, 17关节高斯点)。
供并入 main 输入(PoseC3D 思路早期融合), [T,1,S,S]。
验证: 对 1 个训练 clip 生成并输出形状/非零率(冒烟)。
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.skeleton_dataset import frame_num_of


def skeleton_heatmap(pred_dir, T=16, S=128, sigma=3.0, target_frames=16):
    """读 clip 骨架 json → 逐帧(插值采样到 T)画 17 关节高斯热图 → [T,1,S,S]。"""
    fs = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    if len(fs) < 2:
        return np.zeros((T, 1, S, S), np.float32)
    idx = np.linspace(0, len(fs) - 1, target_frames).round().astype(int)
    hm = np.zeros((T, 1, S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    var = sigma * sigma
    for t, fi in enumerate(idx[:T]):
        try:
            o = json.loads(fs[fi].read_text("utf-8"))
            fr = o if isinstance(o, dict) else o[0]
            kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
        except Exception:
            continue
        # 骨盆中心化 + 用肩宽归一化到 ~32px range(bbox 尺度环境的相对布局图)
        k = kp - kp[0]                      # pelvis 0
        scale = 28.0 / (np.linalg.norm(kp[0] - kp[8]) + 1e-6)   # 躯干长 → 28px
        p = k[:, :2] * scale + S / 2
        # 剔除出界点(depth 大/噪声)
        for j in range(17):
            px, py = p[j]
            if not (0 <= px < S and 0 <= py < S):
                continue
            hm[t, 0] += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * var))
    hm = np.clip(hm, 0, 1)
    return hm


def main():
    d = "data/Training/HAR/Skeleton/27_Take_a_selfie/user1/5-1-1/predictions"
    hm = skeleton_heatmap(Path(d))
    print("heatmap shape:", hm.shape, "非零率:", float((hm > 0).mean()))
    import cv2
    vis = np.uint8(hm[8, 0] * 255)
    cv2.imwrite("outputs/skel_heatmap_smoke.png", vis)
    print("saved outputs/skel_heatmap_smoke.png (第8帧)")
    # 全数据覆盖率(多少 clip 有骨架≥2帧)
    n_ok = n_tot = 0
    for pd_ in glob.glob("data/Training/HAR/Skeleton/*/*/*/predictions"):
        n_tot += 1
        fs = sorted(Path(pd_).glob("*.json"))
        if len(fs) >= 2:
            n_ok += 1
    print(f"训练骨架覆盖率(≥2帧): {n_ok}/{n_tot} = {n_ok/max(n_tot,1):.2%}")


if __name__ == "__main__":
    main()
