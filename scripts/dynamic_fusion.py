#!/usr/bin/env python3
"""CUHK-X —— main+thermal 动态融合推理（样本级动态加权，零训练、零拟合）。

方法本质：不同样本上两模态可靠性不同——main 高置信时它几乎肯定对（保持 main，
避免 thermal 0.63 拖累）；main 低置信时让 thermal（独立视角）参与互补。

特化 + 教训（类感知/门控因"用全量模型在 val 拟合超参"泄漏失败）：
  → 本脚本**不拟合任何参数**，权重 w 是固定启发式：
      w = sigmoid((conf_main - 0.5) / 0.2)，conf 高 → main 主导，conf 低 → thermal 参与
  → 零过拟合风险，直接应用到 test。

对照：main+thermal 朴素平均 = 0.73（LB）
判读：LB > 0.73 → 动态融合有效

用法:
    python scripts/dynamic_fusion.py \
        --main outputs/pack/main_fold0_int5.pth \
        --thermal outputs/pack/thermal_fold0_int5.pth \
        --quantize --flip_tta \
        --main_crop bbox_train.json --main_test_crop bbox_test.json \
        --thermal_crop bbox_thermal_train.json --thermal_test_crop bbox_thermal_test.json \
        --output submission_dynamic.csv
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
from src.quantize import load_quantized


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_sd(ckpt, quantize, device):
    ckpt = Path(ckpt).expanduser()
    if quantize:
        return load_quantized(ckpt, device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def infer(model, ds, device, flip):
    model.eval()
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    outs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            out = model(x)
            if flip:
                out = out + model(torch.flip(x, dims=(-1,)))
            outs.append(out.float().cpu().numpy())
    return np.concatenate(outs) / (2 if flip else 1)


def dynamic_fuse(main_l, th_l):
    """样本级动态加权：w = sigmoid((conf-0.5)/0.2)，不拟合参数。"""
    p_m = softmax(main_l)
    conf = p_m.max(-1)
    w = 1.0 / (1.0 + np.exp(-(conf - 0.5) / 0.2))
    fused = w[:, None] * p_m + (1.0 - w)[:, None] * softmax(th_l)
    return fused.argmax(-1), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--test_csv", default="~/Multimodal/data/Testing/test_file/test.csv")
    ap.add_argument("--main", required=True)
    ap.add_argument("--thermal", required=True)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--main_test_crop", default="bbox_test.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--thermal_test_crop", default="bbox_thermal_test.json")
    ap.add_argument("--output", default="submission_dynamic.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    n = len(raw)

    main_crop = json.loads(Path(args.main_test_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_test_crop).expanduser().read_text(encoding="utf-8"))

    m_main = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=16).to(device)
    m_main.load_state_dict(load_sd(args.main, args.quantize, device))
    m_th = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=16).to(device)
    m_th.load_state_dict(load_sd(args.thermal, args.quantize, device))

    clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
    ds_m = DepthIRVideoDataset(clips, 16, 128, False, main_crop, use_frame_diff=False)
    main_test = infer(m_main, ds_m, device, args.flip_tta)
    th_clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                               "thermal_dir": ddir.parent / "Thermal"})()
                for (cid, ddir, _) in raw]
    ds_t = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
    th_test = infer(m_th, ds_t, device, args.flip_tta)
    pred, w = dynamic_fuse(main_test, th_test)
    print(f"[test] main={n} 动态融合完成；平均 w={w.mean():.3f}（>0.5=main 主导占比）", flush=True)

    # 对齐 test.csv 的 path 格式
    clip_ids = [d.name for d in sorted(test_root.iterdir())
                if d.is_dir() and d.name.startswith("SM_test_")]
    test_df = pd.read_csv(Path(args.test_csv).expanduser())
    pred_map = dict(zip(clip_ids, pred))

    def _clip_of_path(p):
        m = re.search(r"(SM_test_\d+)", str(p))
        return m.group(1) if m else str(p)

    order = test_df["path"].astype(str).map(_clip_of_path)
    test_df["prediction"] = [pred_map.get(k, 0) for k in order]
    test_df[["path", "prediction"]].to_csv(args.output, index=False)
    print(f"==== 动态融合提交: {args.output} ====", flush=True)
    print(f"  rows={len(test_df)} 类数={test_df.prediction.nunique()} "
          f"类0={int((test_df.prediction == 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
