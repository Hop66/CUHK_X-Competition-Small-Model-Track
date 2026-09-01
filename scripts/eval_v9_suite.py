#!/usr/bin/env python3
"""CUHK-X —— yolo v9 三件套 val 对拍（不烧 LB）：4-pass TTA + median crop + prior decode

在 fold0 val 上一次性评测（main R2+1D + thermal R2+1D，prob_avg 融合=0.73 协议代理）：
  T1   TTA2（orig+hflip，现状） vs TTA4（+temporal jitter ±1，yolo v9 安全改进）
  T2   soft-count imbalance 诊断（>2 才该用 prior）
  T3   prior decode 对拍（PRIOR_LAMBDA=0.3 / conf_gate=0.55 / max_flip 6%，yolo v9 逻辑）
  T4   [可选] median-centre crop：传 --main_crop_median 则用 median bbox 重推 main，
       与现 bbox_train.json 对比（先用 detect --median 生成 median bbox json 再传）
判读: tta4 > tta2 → TTA4 上 0.73 测试线；prior>argmax 且 imbalance>2 → prior 值得测；
      median main 更高 → 重建 bbox 用 median。
用法: python scripts/eval_v9_suite.py [--main_ckpt ...] [--thermal_ckpt ...]
      cp 用法: CUDA_VISIBLE_DEVICES="" python scripts/eval_v9_suite.py --workers 0
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

PRIOR_LAMBDA = 0.30
PRIOR_CONF_GATE = 0.55
PRIOR_MAX_FLIP_FRAC = 0.06


def load_sd(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "main" in sd:
        sd = sd["main"]
    return sd


def predict_tta(model, loader, device, tta=4):
    """返回 logits [N,40]；tta=2: orig+hflip；tta=4:+temporal jitter ±1（yolo v9）。"""
    model.eval()
    logits = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)                       # [B,T,C,H,W]
            o = model(x)
            o = o + model(torch.flip(x, dims=(-1,)))
            if tta == 4:
                o = o + model(torch.roll(x, 1, dims=1))
                o = o + model(torch.roll(x, -1, dims=1))
            logits.append(o.cpu().float().numpy())
    return np.concatenate(logits)


def softmax(l):
    p = l - l.max(1, keepdims=True)
    p = np.exp(p)
    return p / p.sum(1, keepdims=True)


def safe_prior_decode(probabilities, lam=PRIOR_LAMBDA, conf_gate=PRIOR_CONF_GATE,
                      max_flip_frac=PRIOR_MAX_FLIP_FRAC):
    """yolo v9 的 prior 解码（拷贝自其 cell14 逻辑，仅长尾>2 才建议用）。"""
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
    return decoded.astype(np.int64), int(flip.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--main_ckpt", default="outputs/main_track/r2plus1d34_depthir_fold0.pth")
    ap.add_argument("--thermal_ckpt", default="outputs/thermal_notebook14/t0/r2plus1d34_thermal_fold0.pth")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--main_crop_median", default="", help="可选：median-centre bbox json（detect --median）")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--tta", type=int, default=4, choices=[2, 4])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    folds = split_by_subject(main_clips, 3)
    _, val_idx = folds[args.fold]
    va_main = [main_clips[i] for i in val_idx]
    labels = np.array([c.action_id for c in va_main], dtype=np.int64)
    N = len(va_main)
    print(f"fold{args.fold} val clips={N}", flush=True)

    def run_main(crop_file, tta):
        crop = json.loads(Path(crop_file).expanduser().read_text(encoding="utf-8"))
        ds = DepthIRVideoDataset(va_main, 16, 128, False, crop, use_frame_diff=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        m = build_model("r2plus1d34", 40, in_channels=4, n_segment=16).to(device)
        m.load_state_dict(load_sd(args.main_ckpt))
        return predict_tta(m, loader, device, tta)

    def run_thermal(tta):
        th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                     root / "Thermal" / c.depth_dir.parent.parent.name
                                     / c.subject / c.sample) for c in va_main]
        crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
        ds_t = ThermalVideoDataset(th_clips, 16, 128, False, crop, use_frame_diff=False)
        loader_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
        t = build_model("r2plus1d34", 40, in_channels=3, n_segment=16).to(device)
        t.load_state_dict(load_sd(args.thermal_ckpt))
        return predict_tta(t, loader_t, device, tta)

    def fuse(m, t):
        if m is not None and t is not None:
            return 0.5 * softmax(m) + 0.5 * softmax(t)
        return softmax(m if m is not None else t)

    has_main = bool(args.main_ckpt) and Path(args.main_ckpt).expanduser().exists()
    has_th = bool(args.thermal_ckpt) and Path(args.thermal_ckpt).expanduser().exists()
    if not (has_main or has_th):
        print("❌ 至少给一个 --main_ckpt / --thermal_ckpt")
        return

    m2 = run_main(args.main_crop, 2) if has_main else None
    t2 = run_thermal(2) if has_th else None
    m4 = run_main(args.main_crop, 4) if has_main else None
    t4 = run_thermal(4) if has_th else None

    f2, f4 = fuse(m2, t2), fuse(m4, t4)
    a2 = float((f2.argmax(1) == labels).mean())
    a4 = float((f4.argmax(1) == labels).mean())
    main_acc = float((softmax(m4).argmax(1) == labels).mean()) if has_main else 0.0
    th_acc = float((softmax(t4).argmax(1) == labels).mean()) if has_th else 0.0

    probs = f4
    imbalance = float(probs.sum(0).max() / max(probs.sum(0).min(), 1e-9))
    argmax_pred = probs.argmax(1).astype(np.int64)
    prior_pred, flipped = safe_prior_decode(probs)
    a_argmax = float((argmax_pred == labels).mean())
    a_prior = float((prior_pred == labels).mean())
    nuniq_a, nuniq_p = len(np.unique(argmax_pred)), len(np.unique(prior_pred))

    print(f"\n==== fold{args.fold} val · yolo-v9 三件套对拍 ====", flush=True)
    if has_main:
        print(f"main acc (tta4)   = {main_acc:.4f}", flush=True)
    if has_th:
        print(f"thermal acc(tta4) = {th_acc:.4f}", flush=True)
    print(f"[TTA2] fused prob_avg = {a2:.4f}  (orig+hflip，= 现 0.73 协议代理)", flush=True)
    print(f"[TTA4] fused prob_avg = {a4:.4f}  (+temporal jitter ±1，yolo v9)", flush=True)
    print(f"       → tta4 - tta2 = {100 * (a4 - a2):+.2f}pt", flush=True)
    print(f"[prior] soft-count imbalance={imbalance:.2f} | argmax={a_argmax:.4f}(n={nuniq_a}) "
          f"prior={a_prior:.4f}(n={nuniq_p},flip={flipped})", flush=True)
    d = a_prior - a_argmax
    if imbalance > 2 and d > 0:
        print(f"   → imbalance>2 且 prior {100 * d:+.2f}pt → prior 值得上", flush=True)
    else:
        print(f"   → prior 无需 ({'imbalance<2' if imbalance <= 2 else f'无增益{100 * d:+.2f}pt'}) → 用 argmax",
              flush=True)

    if args.main_crop_median and Path(args.main_crop_median).expanduser().exists() and has_main:
        m_m = run_main(args.main_crop_median, 4)
        a_m = float((softmax(m_m).argmax(1) == labels).mean())
        print(f"[median-crop] main acc = {a_m:.4f} vs 现 bbox(tta4) {main_acc:.4f} "
              f"→ {100 * (a_m - main_acc):+.2f}pt", flush=True)


if __name__ == "__main__":
    main()
