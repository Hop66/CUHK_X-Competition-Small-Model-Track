#!/usr/bin/env python3
"""CUHK-X —— YOLO 逐帧 bbox 运动特征提取（"人怎么动"线索，零训练成本）

现状：src/detect.py 只存每 clip 一个合并窗口 → 逐帧运动信息全丢。
本脚本：对每 clip 逐帧 YOLO 检测 person bbox → 提取宏观运动特征：
  覆盖率 / 中心轨迹(std·位移·轨迹长) / 面积(mean·std·首末比) / 宽高比 / 垂直位置
区分度（40 类日常动作）：Walk=位移大、Watch_TV=静止、Sit=高度变化、Lie=宽高比。
用途：单独训 GBDT（规则允许非 DL）或与 main 决策级融合（Stacking）。

用法:
    python scripts/yolo_motion_feats.py --train_root ~/Multimodal/data/Training/HAR \
        --out outputs/feats/bbox_feats_train.csv [--stride 2]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.dataset import build_train_index
from src.detect import load_yolo

FEAT_NAMES = ["coverage", "cx_std", "cy_std", "cx_mean", "cy_mean",
              "disp_norm", "traj_len", "w_mean", "h_mean",
              "area_mean", "area_std", "area_ratio", "ratio_mean", "ratio_std",
              "cy_range"]


def clip_boxes(model, ir_dir: Path, device: str, stride: int):
    """逐帧 YOLO 检测 person，返回归一化 bbox 序列（None=该帧没检测到人）。"""
    files = sorted(ir_dir.glob("*.png")) + sorted(ir_dir.glob("*.jpg"))
    files = files[::stride]
    if not files:
        return []
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
        return []
    res = model.predict(imgs, classes=[0], conf=0.25, verbose=False,
                        device=device, batch=16)
    boxes = []
    for r in res:
        H, W = r.orig_shape
        if len(r.boxes):
            b = r.boxes.xyxy[r.boxes.conf.argmax()].tolist()
            boxes.append([b[0] / W, b[1] / H, b[2] / W, b[3] / H])
        else:
            boxes.append(None)
    return boxes


def extract_feats(boxes):
    """bbox 序列 → 运动特征向量（list）。有效帧 <3 返回 None。"""
    valid = [b for b in boxes if b is not None]
    if len(valid) < 3:
        return None
    arr = np.asarray(valid, np.float64)          # [N,4] xyxy 归一化
    N = len(arr)
    cx = (arr[:, 0] + arr[:, 2]) / 2.0
    cy = (arr[:, 1] + arr[:, 3]) / 2.0
    w = arr[:, 2] - arr[:, 0]
    h = arr[:, 3] - arr[:, 1]
    area = w * h
    ratio = w / (h + 1e-8)
    dc = np.diff(np.stack([cx, cy], -1), axis=0)
    step = np.linalg.norm(dc, axis=-1)
    return [len(valid) / max(len(boxes), 1), np.std(cx), np.std(cy), float(cx.mean()),
            float(cy.mean()),
            float(np.sqrt((cx[-1] - cx[0]) ** 2 + (cy[-1] - cy[0]) ** 2)),
            float(step.sum()),
            float(w.mean()), float(h.mean()),
            float(area.mean()), float(area.std()), float(area[-1] / (area[0] + 1e-8)),
            float(ratio.mean()), float(ratio.std()), float(cy.max() - cy.min())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--out", default="outputs/feats/bbox_feats_train.csv")
    ap.add_argument("--stride", type=int, default=2, help="每 N 帧检测一帧")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    model = load_yolo()
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
            boxes = clip_boxes(model, c.ir_dir, device, args.stride)
            feats = extract_feats(boxes)
            if feats is None:
                continue
            wtr.writerow([f"{c.action_id}/{c.subject}/{c.sample}", c.action_id] +
                         [f"{v:.6f}" for v in feats])
            n_feat += 1
            if (i + 1) % 100 == 0:
                print(f"[feats] {i+1}/{len(clips)} 已提取 {n_feat}", flush=True)
    print(f"bbox 运动特征：{len(clips)} clip → {n_feat} 有效 → {out}")


if __name__ == "__main__":
    main()
