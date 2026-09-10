#!/usr/bin/env python3
"""thermal-only 快速自验（避开 fusion_val 的 IO 重链，几十秒出结果）

背景：0.55223 是 dual+thermal 融合产物；dual-only 已在 selftest_main_dual_infer
      三档复现（S 0.6588 / SM 0.6652 / M 0.2853）。唯一缺失 = thermal 单侧在
      陌生 subject val 上的水平。
判读：
  thermal-only ≈ 0.60+（对照 th3d S fold0 训练 0.6009）→ thermal 链 OK，
     融合崩 → 对齐/融合实现问题；或 thermal_s42(full) 测试 OOD
  thermal-only ≪ 0.60 → thermal 推理/crop 有问题（thermal 把融合拖死）

用法:
  python scripts/selftest_thermal_val.py \
    --ckpt outputs/thermal3d_dual/th3d_S_fold0.pth [--flip_tta] [--fold 0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ThermalClipIndex, ThermalVideoDataset, build_train_index
from src.model import build_model
from src.split import split_by_subject
from scripts.ensemble_inference import load_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    tr_idx, va_idx = split_by_subject(clips, n_folds=3)[args.fold]
    va = [clips[i] for i in va_idx]
    print(f"[thermal] fold{args.fold} val clips = {len(va)}", flush=True)

    tc = {}
    _p = Path(args.thermal_crop).expanduser()
    if _p.exists():
        tc = json.loads(_p.read_text(encoding="utf-8"))
        print(f"[thermal] thermal_crop entries = {len(tc)}", flush=True)

    # 直接由 depth_dir 推 Thermal 路径（个别 Thermal 树缺失 → 空目录零张量，顺序不变）
    def _tdir(c):
        adir = c.depth_dir.parent.parent.name
        return root / "Thermal" / adir / c.subject / c.sample

    t_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample, _tdir(c)) for c in va]
    ds = ThermalVideoDataset(t_clips, args.num_frames, args.size, False, tc)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model("r2plus1d34", num_classes=40, in_channels=3,
                        n_segment=args.num_frames).to(device).eval()
    sd = load_state(args.ckpt, False, device)
    if any(k.startswith("static.") for k in sd):   # dual 训练产物 → 剥 static.
        sd = {k[len("static."):]: v for k, v in sd.items() if k.startswith("static.")}
    model.load_state_dict(sd)
    print(f"[thermal] ckpt {Path(args.ckpt).name} loaded", flush=True)

    y = np.zeros(len(va), np.int64)
    logit = np.zeros((len(va), 40), np.float32)
    with torch.no_grad():
        s = 0
        for x, lab, _ in loader:
            b = len(x)
            x = x.to(device)
            o = model(x)
            if args.flip_tta:
                o = o + model(torch.flip(x, dims=(-1,)))
                o = o / 2
            logit[s:s + b] = o.float().cpu().numpy()
            y[s:s + b] = lab.numpy()
            s += b
    acc = (logit.argmax(1) == y).mean()
    n_draw = int((logit.argmax(1) == 0).sum())
    print(f"\n[thermal] thermal-only fold{args.fold} val acc = {acc:.4f}  "
          f"(对照 th3d S fold0 训练 ≈0.6009)")
    print(f"[thermal] 预测类0={n_draw}/{len(va)} 去重类数={logit.argmax(1).max()-logit.argmax(1).min()+1}")
    print("\n[判读]")
    print("  ≈0.60+ → thermal 链 OK → 0.55223 是融合/对齐 or full-OOD 层面")
    print("  ≪0.60 → thermal 推理/crop 与训练不一致 → 修 thermal 侧")


if __name__ == "__main__":
    main()
