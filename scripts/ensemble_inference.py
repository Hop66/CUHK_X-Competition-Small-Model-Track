#!/usr/bin/env python3
"""
CUHK-X —— 三模态集成推理 → submission.csv（含 int8 量化加载）

集成: 主线 Depth+IR(R2+1D) + Thermal(R2+1D) + 骨架(MotionBERT)
- 各模态多折 logit 平均
- 三模态 logit 加权平均（--w_main/--w_thermal/--w_skeleton）
- 可选水平翻转 TTA
- --quantize: 加载 int8 量化包（反量化推理）

用法:
    python scripts/ensemble_inference.py \
        --main outputs/pack/main_int8.pth --quantize \
        --thermal outputs/pack/thermal_int8.pth \
        --skeleton outputs/pack/skeleton_int8.pth \
        --output submission.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalVideoDataset,
                         build_test_index)
from src.model import build_model
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.quantize import load_quantized
from src.skeleton_dataset import MotionBertSkeletonDataset, SkeletonClipIndex


def load_state(ckpt, quantize, device):
    """加载 fp32 或 int8 量化权重，返回 fp32 state_dict。"""
    ckpt = Path(ckpt).expanduser()
    if quantize:
        return load_quantized(ckpt, device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:  # MotionBERT ActionNet 格式
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def safe_prior_decode(probabilities, lam=0.30, conf_gate=0.55, max_flip_frac=0.06):
    """v3 bounded-prior（fork 自 yolo-for-cuhk-x.ipynb UPDATE 3）。
    朝均匀先验做一次 logit 调整，只翻低置信度 clip，硬上限 max_flip_frac。
    适用：模型 class-biased（我们 40 类长尾 29.5 倍）。20 seed 测量最差 −1.7%/典型 +2.3%。
    """
    rows, classes = probabilities.shape
    target = rows / classes
    soft_counts = probabilities.sum(axis=0)
    adjustment = lam * (np.log(target) - np.log(np.maximum(soft_counts, 1e-9)))
    adjusted = np.log(np.maximum(probabilities, 1e-12)) + adjustment[None, :]
    base = probabilities.argmax(axis=1)
    candidate = adjusted.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    flip = (candidate != base) & (confidence < conf_gate)
    gain = adjusted[np.arange(rows), candidate] - adjusted[np.arange(rows), base]
    budget = int(max_flip_frac * rows)
    flip_rows = np.where(flip)[0]
    if len(flip_rows) > budget:
        keep = flip_rows[np.argsort(-gain[flip_rows])[:budget]]
        flip = np.zeros(rows, dtype=bool)
        flip[keep] = True
    decoded = base.copy()
    decoded[flip] = candidate[flip]
    return decoded.astype(int), int(flip.sum())


def noisy_or_fuse(collected):
    """BHaRNet 决策级 Noisy-OR 融合（实验性）。
    p_i,k = per-row softmax 概率；p_nor,k = 1 - Π_i(1 - w_i·p_i,k)。
    解释：某个类只要任一可靠模态高置信就获得高证据 → 弱模态低置信几乎不拖累。
    注意：同向（非互补）模型上 OR 会放大置信、可能增大分歧错误，必须对拍 prob_avg。
    """
    N = collected[0][1].shape[0]
    fused = np.ones((N, 40), dtype=np.float32)
    for w, l in collected:
        p = l - l.max(axis=1, keepdims=True)
        p = np.exp(p)
        p = p / p.sum(axis=1, keepdims=True)
        fused *= (1.0 - w * p)
    fused = 1.0 - fused
    return fused


# ---------------- 主线（Depth+IR） ----------------
def infer_main(args, device):
    in_channels = 8 if args.frame_diff else 4
    model = build_model(args.main_backbone, num_classes=40, in_channels=in_channels,
                        n_segment=args.num_frames).to(device)
    model.eval()
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    crop_cache = load_crop(args.main_crop, raw)
    clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
    ds = DepthIRVideoDataset(clips, args.num_frames, args.size, False, crop_cache,
                             use_frame_diff=args.frame_diff)
    offsets = [-1.0] if args.time_tta <= 1 else \
        [i / (args.time_tta - 1) for i in range(args.time_tta)]
    logits = np.zeros((len(ds), 40), dtype=np.float32)
    n_forward = 0
    for ckpt in args.main:
        model.load_state_dict(load_state(ckpt, args.quantize, device))
        for off in offsets:
            ds = DepthIRVideoDataset(clips, args.num_frames, args.size, False, crop_cache,
                                     use_frame_diff=args.frame_diff, sample_offset=off)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            with torch.no_grad():
                s = 0
                for x, _, _ in loader:
                    x = x.to(device)
                    out = model(x)
                    if args.flip_tta:
                        out = out + model(torch.flip(x, dims=(-1,)))
                    logits[s:s + len(out)] += out.float().cpu().numpy()
                    s += len(out)
            n_forward += 1
        print(f"[main] loaded {ckpt} (time_tta={len(offsets)})", flush=True)
    return logits / n_forward


# ---------------- Thermal ----------------
def infer_thermal(args, device):
    # Thermal 定版是 3ch（无帧差，帧差对 Thermal 有害），固定 3ch
    in_channels = 3
    model = build_model(args.thermal_backbone, num_classes=40, in_channels=in_channels,
                        n_segment=args.num_frames).to(device)
    model.eval()
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    crop_cache = load_crop(args.thermal_crop, raw)
    clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                            "thermal_dir": ddir.parent / "Thermal"})()
             for (cid, ddir, _) in raw]
    ds = ThermalVideoDataset(clips, args.num_frames, args.size, False, crop_cache,
                             use_frame_diff=False)
    offsets = [-1.0] if args.time_tta <= 1 else \
        [i / (args.time_tta - 1) for i in range(args.time_tta)]
    logits = np.zeros((len(ds), 40), dtype=np.float32)
    n_forward = 0
    for ckpt in args.thermal:
        model.load_state_dict(load_state(ckpt, args.quantize, device))
        for off in offsets:
            ds = ThermalVideoDataset(clips, args.num_frames, args.size, False, crop_cache,
                                     use_frame_diff=False, sample_offset=off)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            with torch.no_grad():
                s = 0
                for x, _, _ in loader:
                    x = x.to(device)
                    out = model(x)
                    if args.flip_tta:
                        out = out + model(torch.flip(x, dims=(-1,)))
                    logits[s:s + len(out)] += out.float().cpu().numpy()
                    s += len(out)
            n_forward += 1
        print(f"[thermal] loaded {ckpt} (time_tta={len(offsets)})", flush=True)
    return logits / n_forward


# ---------------- 骨架（MotionBERT） ----------------
def build_test_skeleton_clips(test_root):
    clips = []
    for d in sorted(test_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("SM_test_"):
            continue
        skel = d / "Skeleton"
        pred = skel / "predictions" if (skel / "predictions").is_dir() else skel
        clips.append(SkeletonClipIndex(-1, d.name, d.name, pred))
    return clips


def load_motionbert_backbone(ckpt, quantize, device):
    """从 fp32 或 int8 包加载 ActionNet 完整权重（backbone + head）。"""
    from functools import partial
    import torch.nn as nn
    backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                         num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = ActionNet(backbone=backbone, dim_rep=512, num_classes=40,
                      dropout_ratio=0.5, version="class", hidden_dim=512, num_joints=17).to(device)
    model.load_state_dict(load_state(ckpt, quantize, device))
    return model


def infer_skeleton(args, device):
    test_root = Path(args.test_root).expanduser()
    clips = build_test_skeleton_clips(test_root)
    ds = MotionBertSkeletonDataset(clips, args.num_frames, False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    logits = np.zeros((len(ds), 40), dtype=np.float32)
    for ckpt in args.skeleton:
        model = load_motionbert_backbone(ckpt, args.quantize, device)
        model.eval()
        with torch.no_grad():
            s = 0
            for x, _, _ in loader:
                x = x.to(device).unsqueeze(1)  # [N,1,T,17,3]
                out = model(x)
                logits[s:s + len(out)] += out.float().cpu().numpy()
                s += len(out)
        print(f"[skeleton] loaded {ckpt}", flush=True)
    return logits / len(args.skeleton)


def load_crop(cache_path, raw):
    if not cache_path:
        return {}
    cache = json.loads(Path(cache_path).expanduser().read_text(encoding="utf-8"))
    return {f"-1/{k}/{k}": v for k, v in cache.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", default=[], help="主线 checkpoint（可多折）")
    ap.add_argument("--thermal", nargs="+", default=[], help="Thermal checkpoint")
    ap.add_argument("--skeleton", nargs="+", default=[], help="骨架 MotionBERT checkpoint")
    ap.add_argument("--main_backbone", type=str, default="r2plus1d34",
                    help="主线 backbone（r2plus1d34 为定版）")
    ap.add_argument("--thermal_backbone", type=str, default="r2plus1d34",
                    help="Thermal backbone（r2plus1d34 为定版）")
    ap.add_argument("--frame_diff", action="store_true",
                    help="主线用帧差（4→8ch）；Thermal 固定 3ch 无帧差（帧差对其有害）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--quantize", action="store_true", help="checkpoint 是 int8 量化包")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--time_tta", type=int, default=1,
                    help="时间 TTA：>1 时按 N 个时间偏移窗口分别推理后平均（1=关闭）")
    ap.add_argument("--w_main", type=float, default=1.0)
    ap.add_argument("--w_thermal", type=float, default=1.0)
    ap.add_argument("--w_skeleton", type=float, default=1.0)
    ap.add_argument("--val_acc_main", type=float, default=0.0, help="主线 val acc，用于 --auto_weight")
    ap.add_argument("--val_acc_thermal", type=float, default=0.0)
    ap.add_argument("--val_acc_skeleton", type=float, default=0.0)
    ap.add_argument("--auto_weight", action="store_true",
                    help="用 val acc 的 softmax 自动设权重（覆盖 --w_xxx）")
    ap.add_argument("--prob_avg", action="store_true",
                    help="softmax 后加权平均概率（推荐：不同模型 logit 尺度不同，概率平均更稳）")
    ap.add_argument("--noisy_or", action="store_true",
                    help="BHaRNet 决策级 Noisy-OR 融合（与 --prob_avg 互斥）："
                         "p_k = 1 - Π_i(1 - w_i·p_i,k)，'至少一模态置信'证据池化，"
                         "弱模态低置信贡献小→缓解拖累。实验性，需对比 prob_avg+val")
    ap.add_argument("--main_crop", type=str, default="")
    ap.add_argument("--thermal_crop", type=str, default="")
    ap.add_argument("--test_root", type=str,
                    default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--test_csv", type=str, default="~/Multimodal/data/Testing/test_file/test.csv")
    ap.add_argument("--output", type=str, default="submission.csv")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    # v3 bounded-prior 实验（零训练，side file；主 submission 仍是 argmax）
    ap.add_argument("--prior_experiment", action="store_true",
                    help="输出 *_prior.csv：朝均匀先验调 logit，只翻低置信度 clip（类偏置修正）")
    ap.add_argument("--prior_lambda", type=float, default=0.30)
    ap.add_argument("--prior_conf_gate", type=float, default=0.55)
    ap.add_argument("--prior_max_flip_frac", type=float, default=0.06)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动权重：softmax(val_acc)，差异越大权重越拉开
    if args.auto_weight:
        acc_map = {"main": args.val_acc_main, "thermal": args.val_acc_thermal,
                   "skeleton": args.val_acc_skeleton}
        present = {m: a for m, a in acc_map.items() if a > 0}
        if not present:
            print("错误: --auto_weight 需要至少一个 --val_acc_xxx > 0")
            sys.exit(1)
        z = sum(np.exp(a) for a in present.values())
        weights = {m: np.exp(a) / z for m, a in present.items()}
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
        print(f"auto weights (softmax of val acc): {w_str}")
        args.w_main = weights.get("main", 0.0)
        args.w_thermal = weights.get("thermal", 0.0)
        args.w_skeleton = weights.get("skeleton", 0.0)

    # 各模态分别推理，收集 (权重, logits)
    collected = []
    if args.main:
        collected.append((args.w_main, infer_main(args, device)))
    if args.thermal:
        collected.append((args.w_thermal, infer_thermal(args, device)))
    if args.skeleton:
        collected.append((args.w_skeleton, infer_skeleton(args, device)))
    if not collected:
        print("错误: 至少提供一个模态的 checkpoint")
        sys.exit(1)

    wsum = sum(w for w, _ in collected)
    if args.noisy_or:
        # BHaRNet Noisy-OR：per-class 证据池化，非概率分布的 argmax 仍有效
        fused = noisy_or_fuse(collected)
    elif args.prob_avg:
        # 概率平均：softmax 后加权，不同模型 logit 尺度不敏感
        fused = np.zeros((405, 40), dtype=np.float32)
        for w, l in collected:
            p = l - l.max(axis=1, keepdims=True)
            p = np.exp(p)
            p = p / p.sum(axis=1, keepdims=True)
            fused += w * p
    else:
        # logit 平均
        fused = np.zeros((405, 40), dtype=np.float32)
        for w, l in collected:
            fused += w * l
        fused = fused / wsum
    preds = fused.argmax(axis=1).astype(int)

    test_root = Path(args.test_root).expanduser()
    clip_ids = [d.name for d in sorted(test_root.iterdir())
                if d.is_dir() and d.name.startswith("SM_test_")]
    test_df = pd.read_csv(Path(args.test_csv).expanduser())
    assert len(test_df) == len(clip_ids), f"{len(test_df)} vs {len(clip_ids)}"
    pred_map = dict(zip(clip_ids, preds))
    # 从 test.csv 的 path 提取 clip_id（SM_test_XXXX），不是取最后一段文件名
    def _clip_of_path(p):
        m = re.search(r"(SM_test_\d+)", str(p))
        return m.group(1) if m else str(p)
    order = test_df["path"].astype(str).map(_clip_of_path)
    missing = sum(1 for k in order if k not in pred_map)
    if missing:
        print(f"⚠️ {missing} 个 path 未匹配到 clip_id（检查 test.csv 格式）", flush=True)
    test_df["prediction"] = [pred_map.get(k, 0) for k in order]
    n_zero = int((test_df["prediction"] == 0).sum())
    print(f"预测类分布: 类0={n_zero}/{len(test_df)}, 去重类数={test_df['prediction'].nunique()}", flush=True)

    out = Path(args.output).expanduser()
    test_df[["path", "prediction"]].to_csv(out, index=False)
    print(f"submission saved: {out} ({len(test_df)} rows)")
    assert test_df["prediction"].between(0, 39).all() and len(test_df) == 405

    if args.prior_experiment:
        # 归一化 fused 成概率（prob_avg 时 fused 是加权概率和，需再归一化）
        p = fused - fused.max(axis=1, keepdims=True)
        p = np.exp(p)
        p = p / p.sum(axis=1, keepdims=True)
        soft_counts = p.sum(axis=0)
        imbalance = float(soft_counts.max() / max(soft_counts.min(), 1e-9))
        prior_preds, n_flip = safe_prior_decode(
            p, args.prior_lambda, args.prior_conf_gate, args.prior_max_flip_frac)
        print(f"[prior] soft-count imbalance={imbalance:.2f}（<2 则 unlikely 有帮助，skip）"
              f" 翻转 {n_flip}/405 ({n_flip/405:.1%}, cap {args.prior_max_flip_frac:.0%})",
              flush=True)
        prior_pred_map = dict(zip(clip_ids, prior_preds))
        df2 = test_df.copy()
        df2["prediction"] = [prior_pred_map.get(k, 0) for k in order]
        prior_out = out.with_name(out.stem + "_prior.csv")
        df2[["path", "prediction"]].to_csv(prior_out, index=False)
        print(f"[prior] side file saved: {prior_out}（主 submission 仍是 argmax，"
              f"只在有 spare 提交时测）", flush=True)


if __name__ == "__main__":
    main()
