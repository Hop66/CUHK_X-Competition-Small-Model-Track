#!/usr/bin/env python3
"""SM dual + thermal 融合链路自验（val ground truth，零 LB）

背景：SM dual 提交 0.55223（dual+thermal 融合产物）。此前只自验了 dual 单侧
      static 复现 0.6598，**未验 thermal 单侧与融合链**。本脚本在训练 fold0
      验证集（6 陌生 subject，有标注）上重演「与提交完全一致」的融合：
        dual-only / thermal-only / dual+thermal prob_avg

要点：
  1. 两模态必须用**fold 划分模型**（fold0 ckpt，val 才未见过）——full 模型见过
     val subject 会假 100%。
  2. crop 均用训练原 key（bbox_train/bbox_thermal_train），loader 按同序 zip →
     若顺序错位融合 acc 会显著低于 max(单模态)。
  3. 对照真值：dual fold0 SM=0.6652；thermal 3D fold0≈0.60~0.63。
     融合应 > 两者 → 融合链 OK；融合 ≪ max → 对齐/融合实现 bug。

用法:
  python scripts/selftest_fusion_val.py \
    --dual_ckpt  outputs/main_dual/main_SM_fold0.pth \
    --thermal_ckpt outputs/thermal3d_dual/t0/r2plus1d34_thermal_fold0.pth \
    [--dual_alpha 1.0] [--flip_tta]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex, ThermalVideoDataset,
                         build_train_index)
from src.model import build_model
from src.skeleton_dataset import build_skeleton_index
from src.split import split_by_subject
from src.skeleton_motion import extract_motion_features, load_skeleton
from scripts.ensemble_inference import build_main_dual, load_state


def softmax_row(ls):
    m = ls - ls.max(1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual_ckpt", required=True)
    ap.add_argument("--thermal_ckpt", required=True)
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--dual_alpha", type=float, default=-1.0)
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    tr_idx, va_idx = split_by_subject(clips, n_folds=3)[args.fold]
    va = [clips[i] for i in va_idx]
    print(f"[fusion] fold{args.fold} val clips = {len(va)}", flush=True)

    def _load_crop(p):
        _p = Path(p).expanduser()
        return json.loads(_p.read_text(encoding="utf-8")) if _p.exists() else {}
    mc = _load_crop(args.main_crop)
    tc = _load_crop(args.thermal_crop)
    print(f"[fusion] main_crop={len(mc)} thermal_crop={len(tc)}", flush=True)

    sk = {f"{c.action_id}/{c.subject}/{c.sample}": c.pred_dir
          for c in build_skeleton_index(root)}
    motion = np.zeros((len(va), args.num_frames, 29), np.float32)
    for i, c in enumerate(va):
        pd = sk.get(f"{c.action_id}/{c.subject}/{c.sample}")
        kp, _ = load_skeleton(pd) if pd is not None else (np.zeros((0, 17, 3), np.float32), None)
        motion[i] = extract_motion_features(kp, T=args.num_frames)
    energ = np.abs(motion).sum((1, 2))
    print(f"[fusion] motion energy mean={energ.mean():.3f} "
          f"零={int((energ < 1e-4).sum())}/{len(va)}", flush=True)

    dual = build_main_dual(device).eval()
    dual.load_state_dict(load_state(args.dual_ckpt, False, device))
    th_m = build_model("r2plus1d34", num_classes=40, in_channels=3,
                       n_segment=args.num_frames).to(device).eval()
    th_sd = load_state(args.thermal_ckpt, False, device)
    # 兼容 dual 训练产物（键 static.*）——thermal 目标是裸 R2+1D，剥前缀
    if any(k.startswith("static.") for k in th_sd):
        th_sd = {k[len("static."):]: v for k, v in th_sd.items() if k.startswith("static.")}
    th_m.load_state_dict(th_sd)
    with torch.no_grad():
        a0 = float(torch.sigmoid(dual.a).item())
    a = float(args.dual_alpha) if args.dual_alpha >= 0 else a0
    print(f"[fusion] dual alpha = {a:.3f} (learned={a0:.3f}) | thermal ckpt 就绪", flush=True)

    m_ds = DepthIRVideoDataset(va, args.num_frames, args.size, False, mc)
    # 直接由 depth_dir 推 Thermal 路径（个别 clip 的 Thermal 树缺失用这种方法仍保顺序，
    # ThermalVideoDataset 对空目录返回零张量，不影响对齐）
    def _tdir(c):
        adir = c.depth_dir.parent.parent.name     # 如 "001_Wash_face"
        return root / "Thermal" / adir / c.subject / c.sample
    t_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample, _tdir(c)) for c in va]
    t_ds = ThermalVideoDataset(t_clips, args.num_frames, args.size, False, tc)
    ml = DataLoader(m_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    tl = DataLoader(t_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    N = len(va)
    y = np.zeros(N, np.int64)
    d_l = np.zeros((N, 40), np.float32)
    t_l = np.zeros((N, 40), np.float32)
    with torch.no_grad():
        s = 0
        for (x, lab, _), (xt, _, _) in zip(ml, tl):
            b = len(x)
            x = x.to(device)
            mb = torch.from_numpy(motion[s:s + b]).to(device)
            pm = torch.softmax(dual.motion(mb), -1)
            ps = torch.softmax(dual.static(x), -1)
            p = a * ps + (1 - a) * pm
            o = torch.log(p.clamp_min(1e-12))
            xt = xt.to(device)
            ot = th_m(xt)
            if args.flip_tta:
                psf = torch.softmax(dual.static(torch.flip(x, dims=(-1,))), -1)
                pf = a * psf + (1 - a) * pm
                o = o + torch.log(pf.clamp_min(1e-12))
                ot = ot + th_m(torch.flip(xt, dims=(-1,)))
                o, ot = o / 2, ot / 2
            d_l[s:s + b] = o.float().cpu().numpy()
            t_l[s:s + b] = ot.float().cpu().numpy()
            y[s:s + b] = lab.numpy()
            s += b

    d_p = softmax_row(d_l)
    t_p = softmax_row(t_l)
    f_p = d_p + t_p                       # prob_avg 1:1（提交路径）
    acc_d = (d_l.argmax(1) == y).mean()
    acc_t = (t_l.argmax(1) == y).mean()
    acc_f = (f_p.argmax(1) == y).mean()
    diff = int((t_l.argmax(1) != d_l.argmax(1)).sum())
    print(f"\n[fusion] dual-only      val acc = {acc_d:.4f}  (对照 SM fold0 0.6652)")
    print(f"[fusion] thermal-only   val acc = {acc_t:.4f}  (对照 thermal3D fold0 ≈0.60~0.63)")
    print(f"[fusion] dual+thermal   val acc = {acc_f:.4f}  (--prob_avg 1:1)")
    print(f"[fusion] 两模态预测分歧 clip = {diff}/{N} ({diff/N:.1%})")
    print("\n[判读]")
    print("  acc_d≈0.6652 且 acc_t≈0.60+ 且 acc_f > max(acc_d, acc_t) → 融合链 OK → 0.55223=full-OOD")
    print("  acc_f ≪ max(d,t) → 顺序错位或融合实现 bug（就地修）")
    print("  acc_t ≪ 0.60 → thermal 推理链路/thermal_crop 有问题")


if __name__ == "__main__":
    main()
