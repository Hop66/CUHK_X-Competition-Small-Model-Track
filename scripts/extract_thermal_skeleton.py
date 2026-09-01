#!/usr/bin/env python3
"""CUHK-X —— Step 2：全量热像 2D 骨架提取（KeypointRCNN 零样本迁移）

对每个 train 热像 sample，逐帧跑 KeypointRCNN → 存紧凑 npz：
  <out>/<Action>/<Subject>/<sample>.npz 含:
    kp   [N,17,2]   H3.6M-17 顺序！2D 坐标 root-relative(减骨盆) + 肩宽归一化 → [-1,1]
    conf [N,17]     关键点置信度（KeypointRCNN 的 keypoint_scores，COCO→H36M 重排后）

关键处理（顶会结论，2026-08-23）：
  1. **COCO-17 → H3.6M-17 拓扑映射**：COCO 与 H36M 关节定义/顺序不兼容，必须重排
     12 个直接关节 + 派生 4 个（骨盆=L髋R髋中点、颈=双肩中点、头=鼻颈中点、脊柱=骨盆颈中点）
  2. **root-relative**：减骨盆（关节0），消除绝对位置（跨被试/跨房间泄漏）——与 NYX 3D 同协议
  3. **肩宽归一化**：除以肩宽（H36M 11-14 距离），与 NYX 3D 同协议
  4. **置信度统计**：打印 conf 分布（判断是否退化——官方 Skeleton conf 曾恒 1.0 全退化）

门禁已过（feasibility_thermal_pose.py，人检测 1.00 / 骨架完整 0.95）。
此脚本跑全量 2931 clip（约 1 小时/GPU，帧采样 step 控制速度）。

用法（服务器 GPU，sbatch）:
  python scripts/extract_thermal_skeleton.py --root ~/Multimodal/data/Training/HAR
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from src.dataset import build_thermal_index

# ---------------- COCO-17 → H3.6M-17 映射 ----------------
# H3.6M-17 顺序（MotionBERT 输入）:
#   0骨盆 1R髋 2R膝 3R踝 4L髋 5L膝 6L踝 7脊柱 8颈 9鼻 10头 11L肩 12L肘 13L腕 14R肩 15R肘 16R腕
# COCO-17 顺序（KeypointRCNN 输出）:
#   0鼻 1L眼 2R眼 3L耳 4R耳 5L肩 6R肩 7L肘 8R肘 9L腕 10R腕 11L髋 12R髋 13L膝 14R膝 15L踝 16R踝
# 直接映射: H36M 1,2,3,4,5,6,9,11,12,13,14,15,16 <- COCO 12,14,16,11,13,15,0,5,7,9,6,8,10
# 派生:     0骨盆=(L髋11+R髋12)/2, 7脊柱=(骨盆+颈)/2, 8颈=(L肩5+R肩6)/2, 10头=(鼻0+颈)/2
DIRECT_COCO_2_H36M = [
    (1, 12), (2, 14), (3, 16),   # R髋/膝/踝 <- COCO R髋12/R膝14/R踝16
    (4, 11), (5, 13), (6, 15),   # L髋/膝/踝 <- COCO L髋11/L膝13/L踝15
    (9, 0),                      # 鼻 <- COCO 鼻0
    (11, 5), (12, 7), (13, 9),   # L肩/肘/腕 <- COCO L肩5/L肘7/L腕9
    (14, 6), (15, 8), (16, 10),  # R肩/肘/腕 <- COCO R肩6/R肘8/R腕10
]


def coco_to_h36m(kp: np.ndarray, conf: np.ndarray) -> tuple:
    """COCO-17 [17,2]+conf[17] -> H3.6M-17 [17,2]+conf[17]（派生关节 conf 取来源平均）。

    返回 (kp_h36m [17,2], conf_h36m [17])，缺失关节（坐标 0,0 conf 0）保留给训练时 mask。
    """
    k = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    for h, cidx in DIRECT_COCO_2_H36M:
        k[h] = kp[cidx]
        c[h] = conf[cidx]
    # 派生关节
    nose = kp[0]
    lhip, rhip = kp[11], kp[12]
    lsho, rsho = kp[5], kp[6]
    neck = (lsho + rsho) / 2.0
    c_neck = (conf[5] + conf[6]) / 2.0
    hip_c = (lhip + rhip) / 2.0
    c_hip = (conf[11] + conf[12]) / 2.0
    k[8] = neck            # 颈
    c[8] = c_neck
    k[0] = hip_c           # 骨盆
    c[0] = c_hip
    k[7] = (hip_c + neck) / 2.0   # 脊柱
    c[7] = (c_hip + c_neck) / 2.0
    k[10] = (nose + neck) / 2.0   # 头
    c[10] = (conf[0] + c_neck) / 2.0
    return k, c


def normalize_h36m(kp: np.ndarray, conf: np.ndarray) -> tuple:
    """root-relative（减骨盆）+ 肩宽归一化（H36M 11-14），与 NYX 3D 同协议。

    返回 (kp_norm [17,2], conf_norm [17])。肩宽为 0/缺失时返回零（训练时 mask）。
    """
    pelvis = kp[0]
    kp = kp - pelvis
    shoulder = np.linalg.norm(kp[11] - kp[14])  # L肩-R肩 2D 距离
    if shoulder < 1e-6 or not np.isfinite(shoulder):
        return kp, conf * 0.0  # 肩宽不可靠 → 置信度置 0 让训练 mask
    return kp / shoulder, conf


def load_frames(thermal_dir: Path):
    return sorted(thermal_dir.glob("*.jpg")) + sorted(thermal_dir.glob("*.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--out", default="outputs/thermal_skeleton")
    ap.add_argument("--step", type=int, default=1, help="帧采样步长（1=全帧，2=隔帧，加速）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个 clip（调试用）")
    args = ap.parse_args()

    import torchvision
    from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                              keypointrcnn_resnet50_fpn)
    print(f"[skel] torchvision {torchvision.__version__}，加载 KeypointRCNN...", flush=True)
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval().to(args.device)
    tv_major, tv_minor = (int(x) for x in torchvision.__version__.split(".")[:2])
    self_norm = (tv_major, tv_minor) >= (0, 15)  # >=0.15 期望 [0,1] 输入

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    clips = build_thermal_index(root)
    if args.limit > 0:
        clips = clips[:args.limit]
    print(f"[skel] 共 {len(clips)} clip，step={args.step}", flush=True)

    done = skip = 0
    all_conf = []
    for i, c in enumerate(clips):
        rel = f"{c.action_id}/{c.subject}/{c.sample}"
        dst = out / f"{rel}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file():
            skip += 1
            continue
        files = load_frames(c.thermal_dir)
        if not files:
            skip += 1
            continue
        files = files[:: args.step]
        kps, confs = [], []
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255.0).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            if self_norm:
                arr = img.astype(np.float32) / 255.0
            else:
                arr = img.astype(np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred = model(tensor)[0]
            scores = pred["scores"].cpu().numpy()
            if len(scores) == 0 or scores[0] < 0.5:
                continue  # 无检测 → 跳过该帧（双视角模型按帧数对齐，会 mask 缺失）
            kpt = pred["keypoints"].cpu().numpy()[0]  # [17,3] = x,y,score
            ksc = pred["keypoints_scores"].cpu().numpy()[0]  # [17]
            # COCO 顺序 -> H3.6M 顺序（重排 12 + 派生 4）
            k_h, c_h = coco_to_h36m(kpt[:, :2].astype(np.float32), ksc.astype(np.float32))
            # root-relative + 肩宽归一化（与 NYX 3D 同协议）
            k_n, c_n = normalize_h36m(k_h, c_h)
            kps.append(k_n)
            confs.append(c_n)
            all_conf.append(c_n)
        if not kps:
            skip += 1
            continue
        kp_arr = np.stack(kps, 0).astype(np.float32)     # [N,17,2] H36M 顺序 归一化
        conf_arr = np.stack(confs, 0).astype(np.float32)  # [N,17]
        np.savez_compressed(dst, kp=kp_arr, conf=conf_arr)
        done += 1
        if (i + 1) % 200 == 0 or i == len(clips) - 1:
            print(f"[skel] {i+1}/{len(clips)} (done={done} skip={skip})", flush=True)

    print(f"\n==== 热像骨架提取完成：done={done} skip={skip} → {out} ====", flush=True)
    # 置信度退化检测（顶会结论：退化则丢弃 conf；官方 Skeleton conf 曾恒 1.0）
    if all_conf:
        ac = np.concatenate(all_conf, 0)
        print(f"\n==== 置信度统计（判断是否退化）====", flush=True)
        print(f"  conf shape={ac.shape} mean={ac.mean():.4f} std={ac.std():.4f} "
              f"min={ac.min():.4f} max={ac.max():.4f}", flush=True)
        print(f"  conf>0.5 占比 = {(ac > 0.5).mean():.4f}", flush=True)
        print(f"  conf==1.0 占比 = {(ac == 1.0).mean():.4f}", flush=True)
        print(f"  判读: std<1e-6 → 退化（丢 conf 走 [x,y]）；否则 conf 有信息量（[x,y,conf] + 加权 loss）",
              flush=True)


if __name__ == "__main__":
    main()
