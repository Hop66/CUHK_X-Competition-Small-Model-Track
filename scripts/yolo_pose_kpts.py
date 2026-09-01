#!/usr/bin/env python3
"""CUHK-X —— YOLO11n-pose 关键点部位特征提取（精细部位检测）

现状：MMPose 骨架 3D 但置信度恒 1.0（无法过滤坏帧）。
本脚本：YOLO11n-pose（~5.4MB）在 IR 上检测 17 COCO 关键点（带真实置信度！），
提取部位级运动特征：
  置信度(质量) / 头·手·脚·髋速度 / 手-头距离(喝水/拍手) / 手-髋距离 / 肩宽(尺度) /
  关键点 bbox 宽高比(站/躺) / 中心垂直位置
两种用法：
  A) 特征 → GBDT 辅助分类（与 main 决策级融合）
  B) 关键点序列 → 新的 2D 骨架来源（带置信度，对照 MMPose 3D 0.54）

COCO-17 关键点：0鼻 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘
               9左腕 10右腕 11左髋 12右髋 13左膝 14右膝 15左踝 16右踝

用法:
    python scripts/yolo_pose_kpts.py --train_root ~/Multimodal/data/Training/HAR \
        --out outputs/feats/kpts_feats_train.csv [--stride 2]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.dataset import build_train_index

FEAT_NAMES = ["conf_mean", "conf_std", "n_valid",
              "v_head", "v_hand", "v_foot", "v_hip",
              "hand_head_dist", "hand_hip_dist", "shoulder_w",
              "kbbox_ratio", "cy_mean", "cy_range"]


def _load_yolo_pose():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from ultralytics import YOLO
    local = Path("yolo11n-pose.pt")
    if local.is_file():
        return YOLO(str(local))
    from huggingface_hub import hf_hub_download
    print("[kpts] 从 HF 镜像下载 yolo11n-pose.pt ...", flush=True)
    p = hf_hub_download(repo_id="Ultralytics/YOLO11", filename="yolo11n-pose.pt",
                        local_dir=".", local_dir_use_symlinks=False)
    return YOLO(str(p))


def clip_kpts(model, ir_dir: Path, device: str, stride: int):
    """逐帧 YOLO-pose 检测，返回关键点序列 [T,17,3]（x,y,conf，图像归一化）。"""
    files = sorted(ir_dir.glob("*.png")) + sorted(ir_dir.glob("*.jpg"))
    files = files[::stride]
    if not files:
        return np.zeros((0, 17, 3), np.float32)
    imgs = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.dtype == np.uint16:
            img = (img / 65535.0 * 255.0).astype(np.uint8)
        if img.ndim == 2:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        else:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not imgs:
        return np.zeros((0, 17, 3), np.float32)
    res = model.predict(imgs, conf=0.25, verbose=False, device=device, batch=16)
    seq = []
    for r in res:
        H, W = r.orig_shape
        if r.keypoints is None or len(r.keypoints) == 0 or len(r.keypoints.data) == 0:
            seq.append(np.zeros((17, 3), np.float32))
            continue
        k = np.asarray(r.keypoints.xy[0].cpu().numpy() if hasattr(r.keypoints.xy, "cpu")
                       else r.keypoints.xy[0].numpy()).reshape(-1)
        c = np.asarray(r.keypoints.conf[0].cpu().numpy() if hasattr(r.keypoints.conf, "cpu")
                       else r.keypoints.conf[0].numpy()).reshape(-1)
        # 防御：不同 ultralytics 版本 keypoints 形状可能 [17,2]/[1,17,2]/[17]，统一截断 reshape
        k = k[:34].reshape(17, 2)
        c = c[:17]
        k[:, 0] /= W
        k[:, 1] /= H
        seq.append(np.column_stack([k[:, 0], k[:, 1], c]).astype(np.float32))
    return np.stack(seq, 0) if seq else np.zeros((0, 17, 3), np.float32)


def extract_feats(kpts):
    """关键点序列 [T,17,3] → 部位运动特征。有效帧 <3 返回 None。"""
    # 帧级过滤：按帧平均置信度 >0.25（不能逐元素布尔索引，会把 [T,17,3] 展平成 2D）
    kpts = kpts[kpts[:, :, 2].mean(axis=1) > 0.25]
    T = kpts.shape[0]
    if T < 3:
        return None
    xy = kpts[:, :, :2]                         # [T,17,2] 归一化
    conf = kpts[:, :, 2]
    vel = np.linalg.norm(np.diff(xy, axis=0), axis=-1)   # [T-1,17]
    def v(j):
        return float(vel[:, j].mean())
    hand_head = np.linalg.norm((xy[:, 9] + xy[:, 10]) / 2 - xy[:, 0], axis=-1).mean()
    hand_hip = np.linalg.norm((xy[:, 9] + xy[:, 10]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=-1).mean()
    shoulder_w = float(np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1).mean())
    x0, x1 = xy[:, :, 0].min(), xy[:, :, 0].max()
    y0, y1 = xy[:, :, 1].min(), xy[:, :, 1].max()
    kratio = float((x1 - x0) / ((y1 - y0) + 1e-8))
    cy = xy[:, :, 1].mean(axis=1)
    return [float(conf.mean()), float(conf.std()), T,
            v(0), v(9) + v(10) / 2, v(15) + v(16) / 2, v(11) + v(12) / 2,
            float(hand_head), float(hand_hip), shoulder_w,
            kratio, float(cy.mean()), float(cy.max() - cy.min())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--out", default="outputs/feats/kpts_feats_train.csv")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    model = _load_yolo_pose()
    clips = build_train_index(Path(args.train_root).expanduser())
    if args.limit:
        clips = clips[: args.limit]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    n_feat = 0
    with open(out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["sample", "label"] + FEAT_NAMES)
        for i, c in enumerate(clips):
            kpts = clip_kpts(model, c.ir_dir, device, args.stride)
            feats = extract_feats(kpts)
            if feats is None:
                continue
            wtr.writerow([f"{c.action_id}/{c.subject}/{c.sample}", c.action_id] +
                         [f"{v:.6f}" for v in feats])
            n_feat += 1
            if (i + 1) % 100 == 0:
                print(f"[kpts] {i+1}/{len(clips)} 已提取 {n_feat}", flush=True)
    print(f"YOLO-pose 关键点特征：{len(clips)} clip → {n_feat} 有效 → {out}")


if __name__ == "__main__":
    main()
