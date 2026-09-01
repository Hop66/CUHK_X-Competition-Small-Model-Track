#!/usr/bin/env python3
"""CUHK-X —— fold0 val 上 Noisy-OR 融合 vs prob_avg 融合对拍（BHaRNet 重信度融合评估）

main(R2+1D 4ch) + thermal(R2+1D 3ch) 在 fold0 val 推理 → prob_avg 与 Noisy-OR 两种决策融合对比 acc。
- prob_avg: p = 0.5·softmax(main) + 0.5·softmax(thermal)
- Noisy-OR: p_k = 1 - (1-p_m,k)(1-p_t,k)（BHaRNet 重信度：至少一模态置信）

判读：noisy_or acc - prob_avg acc ≥ +0.01 且分歧样本可控 → 换融合 → 拿 0.73 测试线重出 submit
      否则保持 prob_avg（0.73 测试线不动）。
用法: python scripts/eval_fusion_noisyor.py [--main_ckpt ...] [--thermal_ckpt ...]
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
from src.split import split_by_subject


def load_sd(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "main" in sd:
        sd = sd["main"]
    return sd


def predict(model, loader, device):
    model.eval()
    preds, logits = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            o = model(x)
            preds.append(o.argmax(-1).cpu().numpy())
            logits.append(o.cpu().float().numpy())
    return np.concatenate(preds), np.concatenate(logits)


def softmax(l):
    p = l - l.max(1, keepdims=True)
    p = np.exp(p)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--main_ckpt", default="outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth")
    ap.add_argument("--thermal_ckpt", default="outputs/thermal_34/r2plus1d34_thermal_fold0.pth")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal.json")
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

    # main（4ch 无帧差）
    ds = DepthIRVideoDataset(va_main, 16, 128, False, main_crop, use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    main_model = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=16).to(device)
    main_model.load_state_dict(load_sd(args.main_ckpt))
    _, mlog = predict(main_model, loader, device)

    # thermal（3ch）
    th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                 root / "Thermal" / c.depth_dir.parent.parent.name
                                 / c.subject / c.sample) for c in va_main]
    ds_t = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
    loader_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    th_model = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=16).to(device)
    th_model.load_state_dict(load_sd(args.thermal_ckpt))
    _, tlog = predict(th_model, loader_t, device)

    pm, pt = softmax(mlog), softmax(tlog)
    prob_fuse = 0.5 * pm + 0.5 * pt
    nor_fuse = 1.0 - (1.0 - pm) * (1.0 - pt)

    a_prob = float((prob_fuse.argmax(1) == labels).mean())
    a_nor = float((nor_fuse.argmax(1) == labels).mean())
    div = int((prob_fuse.argmax(1) != nor_fuse.argmax(1)).sum())
    # 分歧中哪边对
    p_pred, n_pred = prob_fuse.argmax(1), nor_fuse.argmax(1)
    p_win = int(((p_pred != n_pred) & (n_pred == labels) & (p_pred != labels)).sum())
    n_win = int(((p_pred != n_pred) & (p_pred == labels) & (n_pred != labels)).sum())
    main_acc = float((mlog.argmax(1) == labels).mean())
    th_acc = float((tlog.argmax(1) == labels).mean())

    print(f"\n==== fold{args.fold} val: main={main_acc:.4f} thermal={th_acc:.4f} ====", flush=True)
    print(f"prob_avg acc      = {a_prob:.4f}", flush=True)
    print(f"Noisy-OR acc      = {a_nor:.4f}", flush=True)
    print(f"分歧样本={div}/{len(labels)} | Noisy-OR 独对(prob仅错)={n_win} | prob 独对(noisy仅错)={p_win}", flush=True)
    if a_nor - a_prob >= 0.01:
        print("判读：Noisy-OR ≥ +1pt → 换融合，重出 0.73 测试 submission 对拍 LB", flush=True)
    else:
        print("判读：Noisy-OR 未超 +1pt → 保持 prob_avg（0.73 测试线不动）", flush=True)


if __name__ == "__main__":
    main()
