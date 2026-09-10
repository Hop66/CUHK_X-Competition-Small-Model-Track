#!/usr/bin/env python3
"""TENT —— 完全测试时自适应（熵最小化, Wang et al. ICLR21 做跨域）。

在未标注测试集上, 仅用 熵损失 L=E[-Σ p log p] 微调 BN 的 scale/shift(少量参数),
直接对抗协变量偏移(cross-subject)。BN-stats 适应(tta_bn.py)是其 BN 统计量特例。

用法:
  1) fold 代理验证: python scripts/tta_tent.py --modality main --mode fold --fold 0
       → 输出 [fold0] no-adapt vs TENT(k步) val acc, 正差才值得上真测试
  2) 真测试出 CSV: python scripts/tta_tent.py --modality main --mode test --steps 200
       --output outputs/sub_G_main_tent.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalVideoDataset,
                         build_test_index, build_train_index)
from src.split import split_by_subject
from src.model import build_model
from scripts.ensemble_inference import load_state

MAIN_FOLD_CKPTS = {0: "outputs/main_baseline/baseline_aug2_fold0.pth",
                   1: "outputs/main_baseline/baseline_aug2_fold1.pth",
                   2: "outputs/main_baseline/baseline_aug2_fold2.pth"}
MAIN_TEST_CKPT = "outputs/main_full/r2plus1d34_depthir_full_seed42.pth"
THERMAL_TEST_CKPT = "outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth"


def build_model_for(modality, device, nf=16):
    if modality == "main":
        return build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=nf).to(device).eval()
    return build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=nf).to(device).eval()


def bn_params(model):
    """只取 BN 的 scale/shift(weight/bias), TENT 标准做法。"""
    out = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p in m.parameters():
                if p is not None and p.requires_grad:
                    out.append(p)
    return out


def tent_step(model, loader, device, steps, lr=1e-3, accum=1):
    params = bn_params(model)
    print(f"[TENT] 可调 BN 参数 {len(params)}（lr={lr}, steps={steps}）", flush=True)
    opt = torch.optim.SGD(params, lr=lr)
    model.train()
    it = 0
    for _ in range(steps):
        for x, *_ in loader:
            x = x.to(device)
            logit = model(x)
            p = torch.softmax(logit, -1)
            loss = -(p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % 50 == 0:
                print(f"  [TENT] it{it} H={loss.item():.4f}", flush=True)
    model.eval()


def eval_acc(model, loader, device, flip=True):
    model.eval()
    n_c = n_t = 0
    with torch.no_grad():
        for b in loader:
            x = b[0].to(device); lab = b[1]   # (x, label, subject)
            o = model(x)
            if flip:
                o = o + model(torch.flip(x, dims=(-1,)))
            n_c += (o.argmax(1).cpu() == lab).sum().item(); n_t += lab.numel()
    return n_c / max(n_t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["main", "thermal"], default="main")
    ap.add_argument("--mode", choices=["fold", "test"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200, help="熵最小化迭代步数")
    ap.add_argument("--tent_lr", type=float, default=1e-3)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--time_tta_n", type=int, default=1)
    ap.add_argument("--output", default="outputs/sub_G_main_tent.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.num_frames
    root = Path.home() / "Multimodal/data/Training/HAR"
    model = build_model_for(args.modality, device, T)

    if args.mode == "fold":
        clips = build_train_index(root)
        tr_idx, va_idx = split_by_subject(clips, n_folds=3)[args.fold]
        va = [clips[i] for i in va_idx]
        import json
        crop_cache = json.loads(Path("bbox_train.json").read_text(encoding="utf-8"))
        ckpt = args.ckpt or MAIN_FOLD_CKPTS[args.fold]
        model.load_state_dict(load_state(ckpt, False, device))
        ds = DepthIRVideoDataset(va, T, args.size, False, crop_cache, use_frame_diff=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        acc0 = eval_acc(model, loader, device, flip=args.flip)
        print(f"[fold{args.fold}] no-adapt val acc = {acc0:.4f}", flush=True)
        tent_step(model, loader, device, args.steps, args.tent_lr)
        acc1 = eval_acc(model, loader, device, flip=args.flip)
        print(f"[fold{args.fold}] TENT val acc = {acc1:.4f} (Δ={acc1-acc0:+.4f})", flush=True)
    else:
        ckpt = args.ckpt or (MAIN_TEST_CKPT if args.modality == "main" else THERMAL_TEST_CKPT)
        model.load_state_dict(load_state(ckpt, False, device))
        test_root = Path.home() / "Multimodal/data/Testing/data/small_model_track_test"
        raw = build_test_index(test_root)
        import json
        crop_p = "bbox_test.json" if args.modality == "main" else "bbox_thermal_test.json"
        crop_cache = json.loads(Path(crop_p).read_text(encoding="utf-8"))
        if args.modality == "main":
            clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
            ds = DepthIRVideoDataset(clips, T, args.size, False, crop_cache, use_frame_diff=False)
        else:
            clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                    "thermal_dir": ddir.parent / "Thermal"})() for (cid, ddir, _) in raw]
            ds = ThermalVideoDataset(clips, T, args.size, False, crop_cache, use_frame_diff=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        offsets = [-1.0] if args.time_tta_n <= 1 else [i / (args.time_tta_n - 1) for i in range(args.time_tta_n)]
        def run_infer():
            model.eval()
            logits = np.zeros((len(raw), 40), np.float32)
            s = 0
            with torch.no_grad():
                for b in loader:
                    x = b[0].to(device)
                    o = model(x)
                    if args.flip:
                        o = o + model(torch.flip(x, dims=(-1,)))
                    logits[s:s + len(o)] = o.float().cpu().numpy(); s += len(o)
            return logits
        logit_no = run_infer()
        pred_no = logit_no.argmax(1)
        tent_step(model, loader, device, args.steps, args.tent_lr)
        logit_ad = run_infer()
        pred_ad = logit_ad.argmax(1)
        n_flip = int((pred_no != pred_ad).sum())
        print(f"[{args.modality}] TENT {args.steps}步: 翻转 {n_flip}/{len(pred_no)} ({n_flip/len(pred_no):.1%})", flush=True)
        clip_ids = [d.name for d in sorted(test_root.iterdir()) if d.name.startswith("SM_test_")]
        test_df = pd.read_csv("data/Testing/test_file/test.csv")
        import re
        order = test_df["path"].astype(str).map(lambda p: re.search(r"(SM_test_\d+)", p).group(1))
        pred_map = dict(zip(clip_ids, pred_ad.astype(int)))
        test_df["prediction"] = [pred_map.get(k, 0) for k in order]
        out = Path(args.output)
        test_df[["path", "prediction"]].to_csv(out, index=False)
        print(f"TENT CSV saved: {out} ({len(test_df)} rows, 类0={int((test_df['prediction']==0).sum())})", flush=True)


if __name__ == "__main__":
    main()
