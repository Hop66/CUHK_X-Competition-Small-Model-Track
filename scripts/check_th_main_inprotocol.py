#!/usr/bin/env python3
"""检验: th 是否拖累 main —— 真锚协议(fold0 s42 + flip_tta + crop)下 main 单模 vs main+th。

背景: OOF(aug2) 显示 main 单模 0.6658 > main+th 0.6468 (+1.9pt, th 净改错28)。
但锚 test 用 s42(fold0, int5, flip_tta) 与 aug2 不同源 —— 需 in-protocol 核验。

方案: 用 s42 fold0 fp32 (main_full/r2plus1d34_depthir_full_seed42) 跑 train fold0 的 val 折
      (subjects 10,11,25,26 留出), 评估协议内 main 单模 vs main+th prob_avg (均 flip_tta+quantize)。

⚠️❌ P0 协议判废 (2026-09-11, 见 idea.md §B):
  `main_full/..._full_seed42.pth` 是 FULL(全量 18 被试) 训练的模型, 已见过所有 18 subjects
  (含所谓 "fold0 val 折") → 拿它在训练集 fold0 上评 "OOF" 是**泄漏**, 结论无效。
  `quantize_pack` 对单 ckpt 命名为 *_fold0_int5 只是列表下标, 不代表它是 fold0 模型。
  此脚本数字**作废**, 仅保留作为"协议错误"的考古记录; 修复需重训真正的 fold0 模型。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DepthIRVideoDataset, ThermalVideoDataset, build_train_index, build_thermal_index
from src.split import split_by_subject
from src.model import build_model
from scripts.ensemble_inference import load_state


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_ckpt", default="outputs/main_full/r2plus1d34_depthir_full_seed42.pth")
    ap.add_argument("--th_ckpt", default="outputs/th_nf32_full/r2plus1d34_thermal_full_seed42.pth")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--th_crop", default="bbox_thermal_train.json")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--th_frames", type=int, default=32, help="thermal 帧数(锚=32)")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--flip", action="store_true", help="与锚 test flip_tta 同构")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="outputs/oof/s42_fold0_inprotocol.pkl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("data/Training/HAR")
    mclips = build_train_index(root)
    thclips = build_thermal_index(root)
    folds = split_by_subject(mclips, 3)
    tr_idx, va_idx = folds[0]              # fold0
    va_m = [mclips[i] for i in va_idx]
    # thermal index 与 main 同 subject 结构 → 要 key 对齐(用相同 sample)
    thmap = {}
    for c in thclips:
        thmap[(c.subject, c.sample)] = c

    crop_cache = json.loads(open(args.main_crop, encoding="utf-8").read())
    th_crop = json.loads(open(args.th_crop, encoding="utf-8").read())

    # ---------- main 推理 ----------
    model = build_model("r2plus1d34", num_classes=40, in_channels=4).to(device).eval()
    sd = load_state(args.main_ckpt, False, device)
    model.load_state_dict(sd)
    ds = DepthIRVideoDataset(va_m, args.num_frames, args.size, False, crop_cache, use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    M = np.zeros((len(va_m), 40), np.float32)
    s = 0
    with torch.no_grad():
        for b in loader:
            x = b[0].to(device)
            o = model(x)
            if args.flip:
                o = o + model(torch.flip(x, dims=(-1,)))
            M[s:s + len(o)] = o.float().cpu().numpy(); s += len(o)
    print(f"main fold0 val logits {M.shape} (flip={args.flip})", flush=True)

    # ---------- thermal 推理 (可选) ----------
    T = None
    if args.th_ckpt:
        tmodel = build_model("r2plus1d34", num_classes=40, in_channels=3).to(device).eval()
        tsd = load_state(args.th_ckpt, False, device)
        tmodel.load_state_dict(tsd)
        # thermal clips 对齐 va_m
        va_t = []
        for c in va_m:
            tt = thmap.get((c.subject, c.sample))
            va_t.append(tt if tt is not None else type("C", (), dict(thermal_dir=Path("") ))())
        tds = ThermalVideoDataset(va_t, args.th_frames, args.size, False, th_crop, use_frame_diff=False)
        tloader = DataLoader(tds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        T = np.zeros((len(va_m), 40), np.float32)
        s = 0
        with torch.no_grad():
            for b in tloader:
                x = b[0].to(device)
                o = tmodel(x)
                if args.flip:
                    o = o + tmodel(torch.flip(x, dims=(-1,)))
                T[s:s + len(o)] = o.float().cpu().numpy(); s += len(o)
        print(f"thermal fold0 val logits {T.shape} (flip={args.flip})", flush=True)

    keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in va_m]
    out = {"keys": keys, "M": M, "T": T}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(args.out, "wb"))
    print(f"saved {args.out} n={len(keys)}", flush=True)

    # ---------- 立即评估 ----------
    lab = np.array([int(k.split("/")[0]) for k in keys])
    predM = M.argmax(-1)
    aM = (predM == lab).mean()
    if T is not None:
        p = softmax(M) + softmax(T); aMT = (p.argmax(-1) == lab).mean()
        fix, brk = 0, 0
        flip = predM != p.argmax(-1)
        fix = ((p.argmax(-1) == lab) & (predM != lab) & flip).sum()
        brk = ((predM == lab) & (p.argmax(-1) != lab) & flip).sum()
        print(f"\n=== 真锚协议(fold0 s42, flip={args.flip}) ===")
        print(f"  main 单模   acc = {aM:.4f}")
        print(f"  main+th     acc = {aMT:.4f}   Δ(main-th)= {aM-aMT:+.4f}")
        print(f"  th融合 改对={fix} 改错={brk} 净{fix-brk:+}  (flip 总 {int(flip.sum())})")
    else:
        print(f"  main 单模 acc = {aM:.4f}  (无 th 对照)")


if __name__ == "__main__":
    main()
