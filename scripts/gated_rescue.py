#!/usr/bin/env python3
"""CUHK-X —— 置信门控救援（骨架/thermal 作为"只救不伤"的安全网）

核心思想：main 高置信直接信 main；main 低置信（max-softmax < τ）时才融合
aux 模态（skeleton/thermal）的软概率 —— 强项不被弱模态污染，弱模态只在
main 不确定时补位。

输出：
  - main 单独 acc（基线）
  - 扫 τ ∈ [0,1] 的最优门控 acc + 最优 τ + 被救援样本数
  - 每个 aux 单独救援 vs main 单独

用法（服务器，CPU/GPU 均可）:
  python scripts/gated_rescue.py --fold 0 \
    --main_ckpt outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth \
    --thermal_ckpt outputs/thermal_v2/r2plus1d34_thermal_fold0.pth \
    --skeleton_ckpt outputs/skeleton_mb/motionbert_fold0.pth
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex,
                         ThermalVideoDataset, build_train_index)
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.model import build_model
from src.skeleton_dataset import MotionBertSkeletonDataset
from src.split import split_by_subject


def load_sd(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "main" in sd:  # 对比学习保存格式
        sd = sd["main"]
    return sd


def load_mb(ckpt, device):
    bb = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                   num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6))
    m = ActionNet(backbone=bb, dim_rep=512, num_classes=40, dropout_ratio=0.5,
                  version="class", hidden_dim=512, num_joints=17).to(device)
    m.load_state_dict(load_sd(ckpt))
    return m


def logits(model, loader, device, is_skel=False):
    """返回 [N,40] logits（硬预测太粗，门控需要置信度）。"""
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            if is_skel:
                x = x.unsqueeze(1)
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def gated_rescue(main_logits, aux_logits_list, labels, steps=201):
    """只救不伤：main 低置信才融合 aux。返回最优 (acc, tau, n_rescued)。"""
    main_prob = softmax(main_logits)
    main_conf = main_prob.max(-1)
    main_pred = main_logits.argmax(-1)
    base = (main_pred == labels).mean()
    aux_prob = (np.mean([softmax(L) for L in aux_logits_list], axis=0)
                if aux_logits_list else None)
    best = (base, 1.0, 0)
    curve = []
    for tau in np.linspace(0.0, 1.0, steps):
        mask = main_conf < tau
        pred = main_pred.copy()
        if aux_prob is not None and mask.sum() > 0:
            fused = softmax(main_logits[mask]) + aux_prob[mask]
            pred[mask] = fused.argmax(-1)
        acc = (pred == labels).mean()
        curve.append((tau, acc, int(mask.sum())))
        if acc > best[0] + 1e-9:
            best = (acc, tau, int(mask.sum()))
    return base, best, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--main_ckpt", default="outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth")
    ap.add_argument("--thermal_ckpt", default="outputs/thermal_v2/r2plus1d34_thermal_fold0.pth")
    ap.add_argument("--skeleton_ckpt", default="outputs/skeleton_mb/motionbert_fold0.pth")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    folds = split_by_subject(main_clips, 3)
    _, val_idx = folds[args.fold]
    va_main = [main_clips[i] for i in val_idx]
    labels = np.array([c.action_id for c in va_main], dtype=np.int64)
    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
    print(f"val {len(va_main)} clips, main acc baseline 见下", flush=True)

    # main
    ds = DepthIRVideoDataset(va_main, 16, 128, False, main_crop, use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    main_model = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=16).to(device)
    main_model.load_state_dict(load_sd(args.main_ckpt))
    main_logits = logits(main_model, loader, device)
    base = (main_logits.argmax(-1) == labels).mean()
    print(f"main 单独 acc = {base:.4f}", flush=True)

    # thermal（可选，--thermal_ckpt 存在则用）
    aux_logits = []
    th_path = Path(args.thermal_ckpt).expanduser()
    if th_path.is_file():
        th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                     root / "Thermal" / c.depth_dir.parent.parent.name
                                     / c.subject / c.sample) for c in va_main]
        ds_t = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
        loader_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
        th_model = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=16).to(device)
        th_model.load_state_dict(load_sd(args.thermal_ckpt))
        th_logits = logits(th_model, loader_t, device)
        aux_logits.append(th_logits)
        print(f"thermal 单独 acc = {(th_logits.argmax(-1) == labels).mean():.4f}", flush=True)

    # skeleton（可选，--skeleton_ckpt 存在则用）
    sk_path = Path(args.skeleton_ckpt).expanduser()
    if sk_path.is_file():
        skel_clips = [type("C", (), {"action_id": c.action_id, "subject": c.subject, "sample": c.sample,
                                     "pred_dir": root / "Skeleton" / c.depth_dir.parent.parent.name
                                     / c.subject / c.sample / "predictions"})() for c in va_main]
        ds_s = MotionBertSkeletonDataset(skel_clips, 16, False)
        loader_s = DataLoader(ds_s, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
        skel_model = load_mb(args.skeleton_ckpt, device)
        skel_logits = logits(skel_model, loader_s, device, is_skel=True)
        aux_logits.append(skel_logits)
        print(f"skeleton 单独 acc = {(skel_logits.argmax(-1) == labels).mean():.4f}", flush=True)

    # 门控：分别用 main+thermal / main+skeleton / main+全部
    names = {0: "main+thermal", 1: "main+skeleton"}
    for idx in range(len(aux_logits)):
        b, best, _ = gated_rescue(main_logits, [aux_logits[idx]], labels)
        n = names.get(idx, f"aux{idx}")
        print(f"\n==== 门控救援：{n} ====", flush=True)
        print(f"  main 单独 = {b:.4f} | 最优门控 = {best[0]:.4f}（τ={best[1]:.2f}, 救援 {best[2]} 样本）", flush=True)
        print(f"  净增益 = {best[0]-b:+.4f}", flush=True)
    if len(aux_logits) == 2:
        b, best, _ = gated_rescue(main_logits, aux_logits, labels)
        print(f"\n==== 门控救援：main+thermal+skeleton 全部 ====", flush=True)
        print(f"  main 单独 = {b:.4f} | 最优门控 = {best[0]:.4f}（τ={best[1]:.2f}, 救援 {best[2]} 样本）", flush=True)
        print(f"  净增益 = {best[0]-b:+.4f}", flush=True)


if __name__ == "__main__":
    main()
