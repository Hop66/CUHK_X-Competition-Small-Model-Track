#!/usr/bin/env python3
"""CUHK-X —— 类感知 / 置信门控融合（main + thermal）→ 提交 csv。

在 fold0 val 上用（全量/打包）模型学融合超参，应用到 test：
  - prob_avg    : 朴素概率平均（当前 0.73 保底方式，对照）
  - gated       : main 置信度 < τ 才让 thermal 参与（只救不伤，τ 在 val 搜）
  - class_aware : 每类 w[c]（val 上 main/thermal 相对准确率），按类别维度加权
  - gated+class : 组合（推荐）
自动选 val 最优策略应用 test，输出 csv + 报告。

⚠️ 说明：val 用与提交相同的模型推理（val 数据训练见过 → val acc 有泄漏，
   但融合超参 τ/w 反映模型真实相对行为，可用于 test）。

用法（服务器）:
  python scripts/fusion_class_gated.py \
    --main outputs/pack/main_fold0_int5.pth \
    --thermal outputs/pack/thermal_fold0_int5.pth \
    --quantize --flip_tta \
    --main_crop bbox_train.json --main_test_crop bbox_test.json \
    --thermal_crop bbox_thermal_train.json --thermal_test_crop bbox_thermal_test.json \
    --output submission_gated.csv
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

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalClipIndex,
                         ThermalVideoDataset, build_test_index, build_train_index)
from src.model import build_model
from src.quantize import load_quantized
from src.split import split_by_subject


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


def infer_main(model, ds, device, flip):
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


def infer_thermal(model, ds, device, flip):
    return infer_main(model, ds, device, flip)  # 同样的推理逻辑


def build_main_model(ckpt, quantize, device, in_channels=4):
    m = build_model("r2plus1d34", num_classes=40, in_channels=in_channels, n_segment=16).to(device)
    m.load_state_dict(load_sd(ckpt, quantize, device))
    m.eval()
    return m


def build_thermal_model(ckpt, quantize, device, in_channels=3):
    m = build_model("r2plus1d34", num_classes=40, in_channels=in_channels, n_segment=16).to(device)
    m.load_state_dict(load_sd(ckpt, quantize, device))
    m.eval()
    return m


# ---------------- 融合策略 ----------------
def apply_strategy(main_l, th_l, w, tau, name):
    """按策略融合。main_l/th_l: [N,40] logits；w: [40] 类权重；tau: 门控阈值。"""
    p_m, p_t = softmax(main_l), softmax(th_l)
    if "class" in name:
        fused = w * p_m + (1.0 - w) * p_t
    else:
        fused = 0.5 * p_m + 0.5 * p_t
    pred = fused.argmax(-1)
    if name in ("gated", "gated+class"):
        conf = p_m.max(-1)
        keep = conf >= tau          # main 高置信 → 直接用 main（只救不伤）
        pred[keep] = main_l[keep].argmax(-1)
    return pred


def fit_strategy(main_val, th_val, labels):
    """在 val 上拟合最优策略 → (val_acc, name, tau, w)。"""
    p_m = softmax(main_val)
    main_pred = main_val.argmax(-1)
    base = (main_pred == labels).mean()
    n_cls = 40
    # 类感知权重：每类 main/th 相对准确率（平滑）
    w = np.full(n_cls, 0.5)
    for c in range(n_cls):
        m = labels == c
        if m.sum() >= 5:
            ma = (main_val[m].argmax(-1) == c).mean()
            ta = (th_val[m].argmax(-1) == c).mean()
            w[c] = np.clip(0.5 + 0.7 * (ma - ta), 0.05, 0.95)
    best = (base, "prob_avg", None, np.full(n_cls, 0.5))
    for name in ("gated", "class_aware", "gated+class"):
        if name == "class_aware":
            pred = apply_strategy(main_val, th_val, w, None, name)
            acc = (pred == labels).mean()
            if acc > best[0]:
                best = (acc, name, None, w.copy())
            continue
        for tau in np.linspace(0.2, 0.95, 31):
            pred = apply_strategy(main_val, th_val, w, tau, name)
            acc = (pred == labels).mean()
            if acc > best[0]:
                best = (acc, name, float(tau), w.copy())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--test_root", default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--main", required=True, help="main int5/fp32 ckpt")
    ap.add_argument("--thermal", required=True, help="thermal int5/fp32 ckpt")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--main_test_crop", default="bbox_test.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--thermal_test_crop", default="bbox_thermal_test.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", default="submission_gated.csv")
    ap.add_argument("--test_csv", default="~/Multimodal/data/Testing/test_file/test.csv",
                    help="Kaggle test.csv（path 列格式 small_model_track_test/SM_test_XXXX/）")
    ap.add_argument("--force", type=str, default="auto",
                    choices=["auto", "prob_avg", "gated", "class_aware", "gated+class"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    print(f"device={device} quantize={args.quantize} flip_tta={args.flip_tta}", flush=True)

    # ---- 1) fold0 val 推理（学融合超参）----
    main_clips = build_train_index(root)
    _, val_idx = split_by_subject(main_clips, 3)[args.fold]
    va_main = [main_clips[i] for i in val_idx]
    labels = np.array([c.action_id for c in va_main], dtype=np.int64)
    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
    print(f"fold{args.fold} val {len(labels)} clips", flush=True)

    m_main = build_main_model(args.main, args.quantize, device)
    ds_vm = DepthIRVideoDataset(va_main, 16, 128, False, main_crop, use_frame_diff=False)
    main_val = infer_main(m_main, ds_vm, device, args.flip_tta)
    print(f"[val] main acc = {(main_val.argmax(-1) == labels).mean():.4f}", flush=True)

    m_th = build_thermal_model(args.thermal, args.quantize, device)
    th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                 root / "Thermal" / c.depth_dir.parent.parent.name
                                 / c.subject / c.sample) for c in va_main]
    ds_vt = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
    th_val = infer_thermal(m_th, ds_vt, device, args.flip_tta)
    print(f"[val] thermal acc = {(th_val.argmax(-1) == labels).mean():.4f}", flush=True)

    # ---- 2) 拟合策略 ----
    val_acc, name, tau, w = fit_strategy(main_val, th_val, labels)
    if args.force != "auto":
        name = args.force
        # 重算该策略的 val acc（仅报告）
        p_m, p_t = softmax(main_val), softmax(th_val)
        if "class" in name:
            fused = w * p_m + (1 - w) * p_t
        else:
            fused = 0.5 * p_m + 0.5 * p_t
        pred = fused.argmax(-1)
        if name in ("gated", "gated+class"):
            conf = p_m.max(-1)
            keep = conf >= tau
            pred[keep] = main_val[keep].argmax(-1)
        val_acc = (pred == labels).mean()
    print(f"\n==== 融合策略（val 拟合）====", flush=True)
    print(f"  最优: {name} | val_acc={val_acc:.4f} (泄漏仅供参考) | "
          f"tau={tau if tau is not None else '-'} | w 类数={int((w != 0.5).sum())} 非 0.5", flush=True)
    print(f"  val 各类权重 w[c]: {np.round(w, 2)}", flush=True)

    # ---- 3) test 推理 + 应用 ----
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)
    n = len(raw)
    main_test_crop = json.loads(Path(args.main_test_crop).expanduser().read_text(encoding="utf-8"))
    th_test_crop = json.loads(Path(args.thermal_test_crop).expanduser().read_text(encoding="utf-8"))
    clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
    ds_tm = DepthIRVideoDataset(clips, 16, 128, False, main_test_crop, use_frame_diff=False)
    main_test = infer_main(m_main, ds_tm, device, args.flip_tta)
    th_clips_t = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                 "thermal_dir": ddir.parent / "Thermal"})()
                  for (cid, ddir, _) in raw]
    ds_tt = ThermalVideoDataset(th_clips_t, 16, 128, False, th_test_crop, use_frame_diff=False)
    th_test = infer_thermal(m_th, ds_tt, device, args.flip_tta)
    print(f"[test] main/thermal 推理完成 {n} clips", flush=True)

    pred = apply_strategy(main_test, th_test, w, tau, name)
    # ---- 输出对齐 test.csv 的 path 格式（Kaggle 期望 small_model_track_test/SM_test_XXXX/）----
    test_root_p = Path(args.test_root).expanduser()
    clip_ids = [d.name for d in sorted(test_root_p.iterdir())
                if d.is_dir() and d.name.startswith("SM_test_")]
    test_df = pd.read_csv(Path(args.test_csv).expanduser())
    if not (len(test_df) == len(clip_ids) == len(pred)):
        print(f"⚠️ 数量不一致: test_csv={len(test_df)} clip={len(clip_ids)} pred={len(pred)}", flush=True)
    pred_map = dict(zip(clip_ids, pred))

    def _clip_of_path(p):
        m = re.search(r"(SM_test_\d+)", str(p))
        return m.group(1) if m else str(p)

    order = test_df["path"].astype(str).map(_clip_of_path)
    missing = sum(1 for k in order if k not in pred_map)
    if missing:
        print(f"⚠️ {missing} 个 path 未匹配到 clip_id（检查 test.csv 格式）", flush=True)
    test_df["prediction"] = [pred_map.get(k, 0) for k in order]
    test_df[["path", "prediction"]].to_csv(args.output, index=False)
    print(f"==== 提交已保存: {args.output} ====", flush=True)
    print(f"  rows={len(test_df)} 类数={test_df.prediction.nunique()} "
          f"类0={int((test_df.prediction == 0).sum())}", flush=True)
    print(f"  path 示例: {test_df.path.iloc[0]}", flush=True)


if __name__ == "__main__":
    main()
