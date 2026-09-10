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
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalVideoDataset,
                         build_test_index)
from src.model import build_model
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.quantize import load_quantized
from src.skeleton_dataset import MotionBertSkeletonDataset, SkeletonClipIndex
from src.skeleton_motion import extract_motion_features, load_skeleton


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
    # roll_jitter 模式：时序抖动 ±1（v9 fork），与 offset 互斥
    roll_shifts = []
    if args.time_tta_mode == "roll_jitter" and args.time_tta > 1:
        roll_shifts = list(range(-(args.time_tta // 2), args.time_tta // 2 + 1))
        if len(roll_shifts) > args.time_tta:
            roll_shifts = roll_shifts[:args.time_tta]

    logits = np.zeros((len(ds), 40), dtype=np.float32)
    n_forward = 0
    for ckpt in args.main:
        model.load_state_dict(load_state(ckpt, args.quantize, device))
        # 多组 segment 采样（num_infer_samples）
        for sample_idx in range(args.num_infer_samples):
            ds_base = DepthIRVideoDataset(clips, args.num_frames, args.size, False, crop_cache,
                                          use_frame_diff=args.frame_diff)
            if args.time_tta_mode == "offset":
                # 旧偏移窗口模式
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
            elif args.time_tta_mode == "roll_jitter":
                # roll_jitter 模式：对输入做 torch.roll 时序抖动
                ds = ds_base
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
                with torch.no_grad():
                    s = 0
                    for x, _, _ in loader:
                        x = x.to(device)
                        out_sum = None
                        for shift in roll_shifts if roll_shifts else [0]:
                            x_roll = torch.roll(x, shifts=shift, dims=2)  # dim=2 是时间轴 [B,C,T,H,W]
                            o = model(x_roll)
                            if args.flip_tta:
                                o = o + model(torch.flip(x_roll, dims=(-1,)))
                            if out_sum is None:
                                out_sum = o
                            else:
                                out_sum = out_sum + o
                        logits[s:s + len(out_sum)] += out_sum.float().cpu().numpy()
                        s += len(out_sum)
                n_forward += len(roll_shifts) if roll_shifts else 1
            else:
                # 默认：不做时间 TTA
                ds = ds_base
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
        print(f"[main] loaded {ckpt} (time_tta={n_forward}, mode={args.time_tta_mode}, samples={args.num_infer_samples})", flush=True)
    return logits / max(n_forward, 1)


# ---------------- 主线 dual（DepthIR 静态 + 骨架运动 MotionNet） ----------------
def build_main_dual(device):
    from src.backbone_compare import build_backbone
    from src.motion_net import MotionNet

    class _DualStaticMotion(nn.Module):
        """与 scripts/train_main_dual.py::MainDual 同构（键 static.*/motion.*/a）。"""
        def __init__(self):
            super().__init__()
            self.static = build_backbone("r2plus1d34", num_classes=40, in_channels=4,
                                         weights_path=None, traj_dim=0)
            self.motion = MotionNet()
            self.a = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, t):                 # x[B,T,4,H,W] t[B,T,29]
            s = self.static(x)
            m = self.motion(t)
            alpha = torch.sigmoid(self.a)
            p = alpha * torch.softmax(s, -1) + (1 - alpha) * torch.softmax(m, -1)
            return torch.log(p.clamp_min(1e-8))

    return _DualStaticMotion().to(device)


def infer_main_dual(args, device):
    """SM dual 主模态：static R2+1D ⊕ MotionNet（测试骨架提取运动特征）α 融合。
    返回 [N,40] 的 log-概率（logit 尺度）——与 infer_main/thermal 同尺度，
    prob_avg 时 softmax(log p) = p（p 已归一化）→ 精确恢复融合概率、不被压平。
    """
    model = build_main_dual(device).eval()
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    crop_cache = load_crop(args.main_crop, raw)
    clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]

    # 每 clip 骨架运动特征 [T,29]（测试 Skeleton/predictions 提取；缺失→零，同训练 fallback）
    ss = 0.479 if args.test_speed_scale <= 0 else float(args.test_speed_scale)   # 1/2.09 帧率归一
    motions = []
    for (cid, ddir, _) in raw:
        pred_dir = ddir.parent / "Skeleton" / "predictions"
        kp, _ = load_skeleton(pred_dir)
        # --test_resample f(>0)：kx 提取前先 kp 级重采样到 round(N*f) 帧（真·重采样路线）
        M = None
        if args.test_resample > 0:
            M = max(2, int(round(kp.shape[0] * args.test_resample)))
        motions.append(extract_motion_features(kp, T=args.num_frames, speed_scale=ss,
                                               resample=M))
    if args.test_resample > 0:
        print(f"[main_dual] 真·重采样 kp×{args.test_resample:g} 后再提特征（每 clip 帧数减按比例）",
              flush=True)
    motions = np.stack(motions).astype(np.float32)                      # [N,T,29]
    energ = np.abs(motions).sum(axis=(1, 2))                            # [N] 每 clip 运动能量
    static_only = energ < 1e-4                                          # 零/近零运动 → 门控为纯 static（防稀释）
    n_sz = int(static_only.sum())
    print(f"[main_dual] test motion {len(motions)} clips shape {motions.shape} | "
          f"energy mean={energ.mean():.3f} std={energ.std():.3f} "
          f"零运动门控 {n_sz}/{len(motions)} ({n_sz/max(len(motions), 1):.1%})", flush=True)

    roll_shifts = []
    if args.time_tta_mode == "roll_jitter" and args.time_tta > 1:
        roll_shifts = list(range(-(args.time_tta // 2), args.time_tta // 2 + 1))
        roll_shifts = roll_shifts[:args.time_tta] if len(roll_shifts) > args.time_tta else roll_shifts

    logit = np.zeros((len(clips), 40), dtype=np.float32)
    for ckpt in args.main_dual:
        model.load_state_dict(load_state(ckpt, args.quantize, device))
        with torch.no_grad():
            a = float(torch.sigmoid(model.a).item())
        if args.dual_alpha >= 0.0:
            a = float(args.dual_alpha)
        tip = '  (learned α)' if args.dual_alpha < 0.0 else f'  (OVERRIDE α={a:.3f})'
        print(f"[main_dual] ckpt {Path(ckpt).name} | alpha={a:.3f}{tip}\n"
              f"           alpha 含义: 0=纯Motion 0.5=均分 1=纯static", flush=True)
        for _samp in range(args.num_infer_samples):
            ds = DepthIRVideoDataset(clips, args.num_frames, args.size, False, crop_cache,
                                     use_frame_diff=False)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            with torch.no_grad():
                s = 0
                for x, _, _ in loader:
                    x = x.to(device)
                    mb = torch.from_numpy(motions[s:s + len(x)]).to(device)  # [B,T,29]
                    g = torch.from_numpy(static_only[s:s + len(x)]).to(device)  # [B] 门控掩码
                    a_v = a
                    pm = torch.softmax(model.motion(mb), -1)   # 运动流（与视频 flip/roll 无关）
                    # 总 forward 数：samples点×flip2（多 TTA log p 之和归一到均值）
                    nf = len(roll_shifts or [0]) * (2 if args.flip_tta else 1)
                    out_sum = None
                    for sh in (roll_shifts or [0]):
                        xr = torch.roll(x, shifts=sh, dims=2) if sh else x
                        ps = torch.softmax(model.static(xr), -1)
                        p = a_v * ps + (1 - a_v) * pm
                        if bool(g.any()):
                            p = torch.where(g[:, None], ps, p)   # 零运动 clip → 不稀释，纯 static
                        o = torch.log(p.clamp_min(1e-12))
                        if args.flip_tta:
                            psf = torch.softmax(model.static(torch.flip(xr, dims=(-1,))), -1)
                            pf = a_v * psf + (1 - a_v) * pm
                            if bool(g.any()):
                                pf = torch.where(g[:, None], psf, pf)
                            o = o + torch.log(pf.clamp_min(1e-12))
                        out_sum = o if out_sum is None else out_sum + o
                    # 均值 log p（logit 尺度）——与 thermal 同尺度，prob_avg 内 softmax(logp)=p
                    logit[s:s + len(x)] += (out_sum / nf).float().cpu().numpy()
                    s += len(x)
        print(f"[main_dual] loaded {ckpt} (samples={args.num_infer_samples}, "
              f"mode={args.time_tta_mode}, flip={args.flip_tta})", flush=True)
    return logit / max(len(args.main_dual), 1)


# ---------------- Thermal ----------------
def infer_thermal(args, device):
    # Thermal 定版是 3ch（无帧差，帧差对 Thermal 有害），固定 3ch
    in_channels = 3
    nf_th = (args.thermal_frames or args.num_frames)
    model = build_model(args.thermal_backbone, num_classes=40, in_channels=in_channels,
                        n_segment=nf_th).to(device)
    model.eval()
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    crop_cache = load_crop(args.thermal_crop, raw)
    clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                            "thermal_dir": ddir.parent / "Thermal"})()
             for (cid, ddir, _) in raw]
    ds = ThermalVideoDataset(clips, nf_th, args.size, False, crop_cache,
                             use_frame_diff=False)
    offsets = [-1.0] if args.time_tta <= 1 else \
        [i / (args.time_tta - 1) for i in range(args.time_tta)]
    roll_shifts = []
    if args.time_tta_mode == "roll_jitter" and args.time_tta > 1:
        roll_shifts = list(range(-(args.time_tta // 2), args.time_tta // 2 + 1))
        if len(roll_shifts) > args.time_tta:
            roll_shifts = roll_shifts[:args.time_tta]

    logits = np.zeros((len(ds), 40), dtype=np.float32)
    n_forward = 0
    for ckpt in args.thermal:
        sd = load_state(ckpt, args.quantize, device)
        # 兼容 dual 训练产物（键 static.*）——thermal 目标是裸 R2+1D，剥前缀
        if any(k.startswith("static.") for k in sd):
            sd = {k[len("static."):]: v for k, v in sd.items() if k.startswith("static.")}
        model.load_state_dict(sd)
        for sample_idx in range(args.num_infer_samples):
            ds_base = ThermalVideoDataset(clips, nf_th, args.size, False, crop_cache,
                                          use_frame_diff=False)
            if args.time_tta_mode == "offset":
                for off in offsets:
                    ds = ThermalVideoDataset(clips, nf_th, args.size, False, crop_cache,
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
            elif args.time_tta_mode == "roll_jitter":
                ds = ds_base
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
                with torch.no_grad():
                    s = 0
                    for x, _, _ in loader:
                        x = x.to(device)
                        out_sum = None
                        for shift in roll_shifts if roll_shifts else [0]:
                            x_roll = torch.roll(x, shifts=shift, dims=2)
                            o = model(x_roll)
                            if args.flip_tta:
                                o = o + model(torch.flip(x_roll, dims=(-1,)))
                            if out_sum is None:
                                out_sum = o
                            else:
                                out_sum = out_sum + o
                        logits[s:s + len(out_sum)] += out_sum.float().cpu().numpy()
                        s += len(out_sum)
                n_forward += len(roll_shifts) if roll_shifts else 1
            else:
                ds = ds_base
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
        print(f"[thermal] loaded {ckpt} (time_tta={n_forward}, mode={args.time_tta_mode}, samples={args.num_infer_samples})", flush=True)
    return logits / max(n_forward, 1)


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
    nf_skel = (args.skeleton_frames or args.num_frames)  # 骨架训练 nf24, 独立帧数
    ds = MotionBertSkeletonDataset(clips, nf_skel, False, input3d=True, clean=args.skel_clean, norm=args.skel_norm)
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


# ---------------- 骨架特征门控融合（58742/58743 主链） ----------------
def infer_gate_fused(args, device):
    """gate_skel_fold{i}.pth（train_gate_fuse.py 产物）→ test logits [N,40].
    结构: main(R2+1D 4ch) + SkeletonBranch(BiGRU) → gate=σ(Linear(512+128,512))
          f=(1-g)*fv + g*proj_s(fs) → head(512→40)。与训练 froze 一致。
    """
    from train_gate_fuse import SkeletonBranch, skel_seq
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    crop_cache = load_crop(args.main_crop, raw)
    clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]

    # 每 clip 骨架 [T,17,6]（与训练 skel_seq 同实现；缺失 → 零张量回退）
    nf = args.num_frames
    skels = []
    for (cid, ddir, _) in raw:
        pred_dir = ddir.parent / "Skeleton" / "predictions"
        s = skel_seq(str(pred_dir), T=nf)
        skels.append(np.zeros((nf, 17, 6), np.float32) if s is None else s)

    main_model = build_model(args.main_backbone, num_classes=40, in_channels=4,
                             n_segment=nf).to(device)
    skel_branch = SkeletonBranch(hidden=128, T=nf).to(device)
    gate = nn.Sequential(nn.Linear(512 + 128, 512), nn.Sigmoid()).to(device)
    proj_s = nn.Sequential(nn.Linear(128, 512), nn.ReLU()).to(device)
    head = nn.Linear(512, 40).to(device)

    feats = {}

    def _hook(mod, inp, o):
        feats["fv"] = o

    main_model.encoder.register_forward_hook(_hook)

    @torch.no_grad()
    def fused(x, s):
        feats.clear()
        _ = main_model(x)
        fv = feats["fv"]
        fs = skel_branch(s)
        g = gate(torch.cat([fv, fs], -1))
        f = (1 - g) * fv + g * proj_s(fs)
        return head(f)

    logits = np.zeros((len(clips), 40), dtype=np.float32)
    n_forward = 0
    for ckpt in args.gate_fused:
        ck = torch.load(Path(ckpt).expanduser(), map_location=device)
        main_model.load_state_dict(ck["main"])
        skel_branch.load_state_dict(ck["skel"])
        gate.load_state_dict(ck["gate"])
        proj_s.load_state_dict(ck["proj_s"])
        head.load_state_dict(ck["head"])
        main_model.eval(); skel_branch.eval(); gate.eval(); proj_s.eval()
        for _samp in range(args.num_infer_samples):
            ds = DepthIRVideoDataset(clips, nf, args.size, False, crop_cache,
                                     use_frame_diff=args.frame_diff)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            s_idx = 0
            for x, _, _ in loader:
                x = x.to(device)
                s = torch.from_numpy(np.stack(skels[s_idx:s_idx + len(x)])).to(device)
                out_sum = None
                xr = x
                # 骨架 TTA：时间反转对 BiGRU 语义不一致，只做水平翻转（与 main 链一致）
                o = fused(xr, s)
                if args.flip_tta:
                    o = o + fused(torch.flip(xr, dims=(-1,)), s)
                out_sum = o
                logits[s_idx:s_idx + len(x)] += out_sum.float().cpu().numpy()
                s_idx += len(x)
            n_forward += (2 if args.flip_tta else 1)
        print(f"[gate_fused] loaded {Path(ckpt).name} (flip={args.flip_tta})", flush=True)
    return logits / max(n_forward, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", default=[], help="主线 checkpoint（可多折）")
    ap.add_argument("--gate_fused", nargs="+", default=[],
                    help="骨架特征门控融合产物 gate_skel_fold{i}.pth（58742/58743）")
    ap.add_argument("--w_gate", type=float, default=1.0)
    ap.add_argument("--main_dual", nargs="+", default=[],
                    help="主线 dual(SM) checkpoint：static R2+1D ⊕ MotionNet 融合（motion 用测试骨架提取）")
    ap.add_argument("--dual_alpha", type=float, default=-1.0,
                    help="SM dual 推理强制 α（默认-1=训练 learned α；1.0=纯static 0.5=均分 0.0=纯motion 诊断）")
    ap.add_argument("--test_speed_scale", type=float, default=1.0,
                    help="测试骨架运动速度项缩放（训练fps/测试fps，实测≈0.5；0 时自动用 0.479 = 1/2.09）")
    ap.add_argument("--test_resample", type=float, default=0.0,
                    help="真·重采样：测试 kp 在特征提取前重采样到 round(N*f) 帧（f>1 升采样=每帧物理时间变短→每帧速度变小；0=关闭）")
    ap.add_argument("--thermal", nargs="+", default=[], help="Thermal checkpoint")
    ap.add_argument("--skeleton", nargs="+", default=[], help="骨架 MotionBERT checkpoint")
    ap.add_argument("--skel_norm", type=str, default="shoulder", choices=["shoulder", "torso"],
                    help="骨架归一化（与训练对齐：augmented 用 shoulder/torso）")
    ap.add_argument("--skel_clean", action="store_true", help="骨架时间清洗（与训练 --clean 对齐）")
    ap.add_argument("--main_backbone", type=str, default="r2plus1d34",
                    help="主线 backbone（r2plus1d34 为定版）")
    ap.add_argument("--thermal_backbone", type=str, default="r2plus1d34",
                    help="Thermal backbone（r2plus1d34 为定版）")
    ap.add_argument("--frame_diff", action="store_true",
                    help="主线用帧差（4→8ch）；Thermal 固定 3ch 无帧差（帧差对其有害）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--thermal_frames", type=int, default=None,
                    help="thermal 独立帧数(默认与 --num_frames 相同)")
    ap.add_argument("--skeleton_frames", type=int, default=None,
                    help="骨架(MotionBERT) 独立帧数(默认与 --num_frames 相同; 骨架训练 nf24 需显式 24)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--quantize", action="store_true", help="checkpoint 是 int8 量化包")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--time_tta", type=int, default=1,
                    help="时间 TTA：>1 时按 N 个时间偏移窗口分别推理后平均（1=关闭）")
    ap.add_argument("--time_tta_mode", type=str, default="offset", choices=["offset", "roll_jitter"],
                    help="时间 TTA 模式：offset=旧偏移窗口；roll_jitter=v9 fork 时序抖动 ±1（torch.roll）")
    ap.add_argument("--num_infer_samples", type=int, default=1,
                    help="推理时每组独立 segment 采样次数（>1 时 logits 平均，类似训练多采样增强）")
    ap.add_argument("--w_main", type=float, default=1.0)
    ap.add_argument("--w_main_dual", type=float, default=1.0)
    ap.add_argument("--w_thermal", type=float, default=1.0)
    ap.add_argument("--w_skeleton", type=float, default=1.0)
    ap.add_argument("--imu_logits", nargs="+", default=[], help="预计算 IMU test logits .npy 列表(行序=SM_test_ sorted, Nx40); 多个=平分 --w_imu")
    ap.add_argument("--w_imu", type=float, default=1.0)
    ap.add_argument("--val_acc_main", type=float, default=0.0, help="主线 val acc，用于 --auto_weight")
    ap.add_argument("--val_acc_main_dual", type=float, default=0.0,
                    help="主线 dual val acc（SM 融合），用于 --auto_weight")
    ap.add_argument("--val_acc_thermal", type=float, default=0.0)
    ap.add_argument("--val_acc_skeleton", type=float, default=0.0)
    ap.add_argument("--auto_weight", action="store_true",
                    help="用 val acc 的 softmax 自动设权重（覆盖 --w_xxx）")
    ap.add_argument("--prob_avg", action="store_true",
                    help="softmax 后加权平均概率（推荐：不同模型 logit 尺度不同，概率平均更稳）")
    ap.add_argument("--geomean", action="store_true",
                    help="对数域加权平均(log softmax)：fused=Σ w·log p，argmax 等价几何平均。"
                         "fold 扫描 +0.42pt(0.6777→0.6819, 3折正) vs prob_avg；与 --prob_avg 互斥")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="softmax 温度：>1 拉平(低熵成员更确信时更纠偏)、<1 锐化")
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
    ap.add_argument("--save_avg_probs", type=str, default="",
                    help="保存 prob_avg 融合后的 soft 概率(anchor soft, {clip_id: probs[40]})")
    ap.add_argument("--save_main_logits", type=str, default="",
                    help="保存 main 模态 test logits {clip_id: logits[40]} (与锚同源: flip_tta/quantize/crop)")
    ap.add_argument("--save_thermal_logits", type=str, default="",
                    help="保存 thermal 模态 test logits {clip_id: logits[40]} (与锚同源)")
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
        acc_map = {"main": args.val_acc_main, "main_dual": args.val_acc_main_dual,
                   "thermal": args.val_acc_thermal,
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
        args.w_main_dual = weights.get("main_dual", 0.0)
        args.w_thermal = weights.get("thermal", 0.0)
        args.w_skeleton = weights.get("skeleton", 0.0)

    # 各模态分别推理，收集 (权重, logits)
    collected = []
    if args.main:
        _l = infer_main(args, device)
        collected.append((args.w_main, _l))
        if args.save_main_logits:
            import pickle as _p
            cids = [d.name for d in sorted(Path(args.test_root).expanduser().iterdir())
                    if d.is_dir() and d.name.startswith("SM_test_")]
            _p.dump({c: _l[i].astype(np.float32) for i, c in enumerate(cids)},
                    open(Path(args.save_main_logits).expanduser(), "wb"))
            print(f"saved main logits -> {args.save_main_logits}", flush=True)
    if args.gate_fused:
        collected.append((args.w_gate, infer_gate_fused(args, device)))
    if args.main_dual:
        collected.append((args.w_main_dual, infer_main_dual(args, device)))
    if args.thermal:
        _l = infer_thermal(args, device)
        collected.append((args.w_thermal, _l))
        if args.save_thermal_logits:
            import pickle as _p
            cids = [d.name for d in sorted(Path(args.test_root).expanduser().iterdir())
                    if d.is_dir() and d.name.startswith("SM_test_")]
            _p.dump({c: _l[i].astype(np.float32) for i, c in enumerate(cids)},
                    open(Path(args.save_thermal_logits).expanduser(), "wb"))
            print(f"saved thermal logits -> {args.save_thermal_logits}", flush=True)
    if args.skeleton:
        collected.append((args.w_skeleton, infer_skeleton(args, device)))
    if args.imu_logits:
        nf = len(args.imu_logits)
        for f in args.imu_logits:
            imu_l = np.load(Path(f).expanduser())
            if imu_l.shape != (405, 40):
                print(f"警告: {Path(f).name} shape {imu_l.shape} != (405,40), 仍尝试")
            collected.append((args.w_imu / nf, imu_l.astype(np.float32)))
        print(f"loaded {nf} IMU logits w_total={args.w_imu} each={args.w_imu/nf:.3f}",
              flush=True)
    if not collected:
        print("错误: 至少提供一个模态的 checkpoint")
        sys.exit(1)

    wsum = sum(w for w, _ in collected)
    if args.noisy_or:
        # BHaRNet Noisy-OR：per-class 证据池化，非概率分布的 argmax 仍有效
        fused = noisy_or_fuse(collected)
    elif args.geomean:
        # 对数域加权平均（几何平均）：对温度缩放后的 log-softmax 加权求和，argmax 不变
        fused = np.zeros((405, 40), dtype=np.float32)
        for w, l in collected:
            p = l / args.temp
            p = p - p.max(axis=1, keepdims=True)
            lp = p - np.log(np.exp(p).sum(axis=1, keepdims=True) + 1e-12)
            fused += w * lp
        print(f"[geomean] T={args.temp} 对数域加权平均，w_sum={wsum}", flush=True)
        if abs(wsum - 1.0) > 1e-6 and abs(wsum) > 1e-9:
            fused = fused / wsum
    elif args.prob_avg:
        # 概率平均：softmax 后加权，不同模型 logit 尺度不敏感
        fused = np.zeros((405, 40), dtype=np.float32)
        for w, l in collected:
            p = l / args.temp
            p = p - p.max(axis=1, keepdims=True)
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
    if args.save_avg_probs:
        import pickle as _p
        _p.dump({cid: fused[i].astype(np.float32) for i, cid in enumerate(clip_ids)},
                open(Path(args.save_avg_probs).expanduser(), "wb"))
        print(f"saved avg probs -> {args.save_avg_probs}", flush=True)
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
