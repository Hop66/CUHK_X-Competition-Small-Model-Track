#!/usr/bin/env python3
"""一站式冲刺推理器：main(+thermal) 融合 + 多裁剪空间TTA + flip + 时窗TTA + 可选BN适应。

覆盖 ensemble_inference 之外缺失的两个增量：
  1) 空间多裁剪 TTA（crop 缩放/平移变体 logit 平均）——跨被试常 +0.5~1.5pt
  2) BN 测试时适应（transductive, train()+no_grad 回暖一程）
用法：
  python scripts/infer_multicrop.py --main main_s42 p777 --thermal th_s42 --quantize \
      --multicrop 6 --flip --time_tta_n 3 --bn_passes 1 --output sub_F_stk.csv
"""
import argparse
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
from scripts.ensemble_inference import load_crop, load_state


def trans_crop(box, v):
    """返回变体 v 的裁剪框 (x1,y1,x2,y2) 归一化。v=0 不变。"""
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if v == 1:   f = 1.15                       # 外扩(zoom out)
    elif v == 2: f = 0.85                       # 内缩(zoom in)
    elif v == 3: return (max(cx - w/2 - 0.08*w, 0), y1, min(cx + w/2 - 0.08*w, 1), y2)  # 左移
    elif v == 4: return (max(cx - w/2 + 0.08*w, 0), y1, min(cx + w/2 + 0.08*w, 1), y2)  # 右移
    elif v == 5: return (x1, max(cy - h/2 - 0.08*h, 0), x2, min(cy + h/2 - 0.08*h, 1))  # 上移
    elif v == 6: return (x1, max(cy - h/2 + 0.08*h, 0), x2, min(cy + h/2 + 0.08*h, 1))  # 下移
    elif v == 7: return (max(x1 - 0.05*w, 0), max(y1 - 0.05*h, 0), min(x2 + 0.05*w, 1), min(y2 + 0.05*h, 1))  # 放大框
    else:        return (x1, y1, x2, y2)        # v=0 identity
    nw, nh = w * f, h * f
    return (max(cx - nw / 2, 0), max(cy - nh / 2, 0), min(cx + nw / 2, 1), min(cy + nh / 2, 1))


def shifted_crop_map(base_map, v):
    return {k: trans_crop(b, v) for k, b in base_map.items()}


def adapt_bn(model, loader, device, passes):
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            for x, *_ in loader:
                model(x.to(device))


def infer_stream(args, device, modality, model, ckpts, crop_map, clip_keyfn, ds_ctor):
    """返回 [N,40] logits（对 crops×offsets×ckpts 平均）。"""
    N = len(ds_ctor(crop_map, 0, 0))
    logit = np.zeros((N, 40), np.float32)
    n_f = 0
    offsets = [-1.0] if args.time_tta_n <= 1 else [i / (args.time_tta_n - 1) for i in range(args.time_tta_n)]

    # BN 测试时适应：在基裁剪上对第 1 个 ckpt 适配一次，然后所有评价都切 eval
    if args.bn_passes > 0 and ckpts:
        ds0 = ds_ctor(crop_map, 0, offsets[0])
        model.load_state_dict(load_state(ckpts[0], args.quantize, device))
        loader0 = DataLoader(ds0, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
        adapt_bn(model, loader0, device, args.bn_passes)
        print(f"[{modality}] BN 适应 {args.bn_passes}pass 完成", flush=True)

    for ci, ckpt in enumerate(ckpts):
        model.load_state_dict(load_state(ckpt, args.quantize, device))
        model.eval()
        for v in range(args.multicrop):
            cm = shifted_crop_map(crop_map, v)
            for off in offsets:
                ds = ds_ctor(cm, v, off)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
                with torch.no_grad():
                    s = 0
                    for x, *_ in loader:
                        x = x.to(device)
                        o = model(x)
                        if args.flip:
                            o = o + model(torch.flip(x, dims=(-1,)))
                        logit[s:s + len(o)] += o.float().cpu().numpy()
                        s += len(o)
                n_f += 1
    return logit / max(n_f, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", default=[])
    ap.add_argument("--thermal", nargs="+", default=[])
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--multicrop", type=int, default=1, help="空间裁剪变体数(1=仅原框; 2..7 含缩放/平移)")
    ap.add_argument("--time_tta_n", type=int, default=1, help="时窗 TTA 数(1=关)")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--bn_passes", type=int, default=0, help="BN 测试时适应前向次数(0=关)")
    ap.add_argument("--output", type=str, default="outputs/sub_F_stk.csv")
    args = ap.parse_args()
    args.multicrop = max(1, min(8, args.multicrop))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_root = Path.home() / "Multimodal/data/Testing/data/small_model_track_test"
    raw = build_test_index(test_root)
    clip_ids = [d.name for d in sorted(test_root.iterdir()) if d.name.startswith("SM_test_")]
    collected = []

    def main_ds(cm, v, off):
        clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
        return DepthIRVideoDataset(clips, args.num_frames, args.size, False, cm,
                                   use_frame_diff=False, sample_offset=off)
    def th_ds(cm, v, off):
        clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                "thermal_dir": ddir.parent / "Thermal"})() for (cid, ddir, _) in raw]
        return ThermalVideoDataset(clips, args.num_frames, args.size, False, cm,
                                   use_frame_diff=False, sample_offset=off)

    if args.main:
        m = build_model("r2plus1d34", num_classes=40, in_channels=4,
                        n_segment=args.num_frames).to(device).eval()
        cmap = load_crop("bbox_test.json", raw)
        collected.append(("main", infer_stream(args, device, "main", m, args.main, cmap, None, main_ds)))
    if args.thermal:
        m = build_model("r2plus1d34", num_classes=40, in_channels=3,
                        n_segment=args.num_frames).to(device).eval()
        cmap = load_crop("bbox_thermal_test.json", raw)
        collected.append(("thermal", infer_stream(args, device, "thermal", m, args.thermal, cmap, None, th_ds)))

    # 概率平均融合
    fused = np.zeros((len(raw), 40), np.float32)
    for name, l in collected:
        p = l - l.max(axis=1, keepdims=True)
        p = np.exp(p); p = p / p.sum(axis=1, keepdims=True)
        fused += p
    if collected:
        fused /= len(collected)
    preds = fused.argmax(1).astype(int)

    test_df = pd.read_csv("data/Testing/test_file/test.csv")
    order = test_df["path"].astype(str).map(lambda p: re.search(r"(SM_test_\d+)", p).group(1))
    pred_map = dict(zip(clip_ids, preds))
    test_df["prediction"] = [pred_map.get(k, 0) for k in order]
    out = Path(args.output)
    test_df[["path", "prediction"]].to_csv(out, index=False)
    print(f"saved: {out} ({len(test_df)} rows, 类0={int((test_df['prediction']==0).sum())}, "
          f"去重={test_df['prediction'].nunique()})", flush=True)
    print(f"配置: main={args.main} thermal={args.thermal} | crops={args.multicrop} "
          f"time={args.time_tta_n} bn={args.bn_passes} flip={args.flip}", flush=True)


if __name__ == "__main__":
    main()
