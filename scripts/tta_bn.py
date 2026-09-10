#!/usr/bin/env python3
"""BN 测试时自适应（transductive BN adaptation / TENT 的 BN-stats 变体）。

动机：训练/测试跨被试(域移)。测试集 405 clip 推理时全部可得 → 可在推理前用
      unlabeled 测试视频更新每条 BN 的 running_mean/var（train()+no_grad 回暖一程），
      以对抗协变量偏移。零训练、只多 1~2 次全量前向，代价≈两次推理。

用法：
  1) fold 代理验证（跨被试有效性, offline）：
     python scripts/tta_bn.py --modality main --mode fold --fold 0 --flip
     → 输出【val acc: no-adapt vs adapt-1pass vs adapt-2pass】，正差才值得上测试
  2) 真测试集出 CSV：
     python scripts/tta_bn.py --modality main --mode test --flip --time_tta 1 \
                              --output outputs/sub_E_main_bn.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
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


def build_model_for(modality, device):
    if modality == "main":
        return build_model("r2plus1d34", num_classes=40, in_channels=4,
                           n_segment=16).to(device).eval()
    return build_model("r2plus1d34", num_classes=40, in_channels=3,
                       n_segment=16).to(device).eval()


def make_loader(ds, bs, workers):
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      num_workers=workers, pin_memory=True)


def adapt_bn(model, loader, device, passes=1):
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            for x, *_ in loader:
                model(x.to(device))


def eval_acc(model, loader, device, flip=True, thermal_dir=False):
    model.eval()
    n_c = n_t = 0
    ys = []
    with torch.no_grad():
        for batch in loader:
            x, lab = batch[0].to(device), batch[1]   # (x, label, subject)
            out = model(x)
            if flip:
                out = out + model(torch.flip(x, dims=(-1,)))
            n_c += (out.argmax(1).cpu() == lab).sum().item()
            n_t += lab.numel()
            ys.append(lab)
    return n_c / max(n_t, 1)


def eval_logits(model, loader, device, flip=True, time_tta=1, offset_mode="offset"):
    model.eval()
    logits = None
    s = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            out = None
            offsets = [-1.0] if time_tta <= 1 else [i / (time_tta - 1) for i in range(time_tta)]
            for off in offsets:
                # offset 由外部生成 loader（每 offset 一个 loader）；此处简化：time_tta>1 由外部循环
                o = model(x)
                if flip:
                    o = o + model(torch.flip(x, dims=(-1,)))
                out = o if out is None else out + o
            out = out / len(offsets)
            if logits is None:
                logits = np.zeros((len(loader.dataset), out.shape[1]), np.float32)
            logits[s:s + len(out)] = out.float().cpu().numpy()
            s += len(out)
    return logits


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
    ap.add_argument("--adapt_passes", type=int, default=2)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--output", default="outputs/sub_E_main_bn.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.num_frames
    root = Path.home() / "Multimodal/data/Training/HAR"
    model = build_model_for(args.modality, device)

    if args.mode == "fold":
        assert args.modality == "main", "fold 代理目前只支持 main（thermal fold ckpt 也齐后放开）"
        clips = build_train_index(root)
        tr_idx, va_idx = split_by_subject(clips, n_folds=3)[args.fold]
        va = [clips[i] for i in va_idx]
        print(f"[{args.modality}] fold{args.fold} val clips={len(va)}", flush=True)
        import json
        crop_cache = json.loads(Path("bbox_train.json").read_text(encoding="utf-8"))
        ckpt = args.ckpt or MAIN_FOLD_CKPTS[args.fold]
        model.load_state_dict(load_state(ckpt, False, device))

        ds = DepthIRVideoDataset(va, T, args.size, False, crop_cache, use_frame_diff=False)
        loader = make_loader(ds, args.batch_size, args.workers)

        acc0 = eval_acc(model, loader, device, flip=args.flip)
        print(f"[fold{args.fold}] no-adapt  val acc = {acc0:.4f}", flush=True)
        for p in range(1, args.adapt_passes + 1):
            adapt_bn(model, loader, device, passes=1)     # 每次加一程（累积）
            acc = eval_acc(model, loader, device, flip=args.flip)
            print(f"[fold{args.fold}] adapt {p}pass val acc = {acc:.4f} (Δ={acc-acc0:+.4f})", flush=True)
    else:
        ckpt = args.ckpt or (MAIN_TEST_CKPT if args.modality == "main" else THERMAL_TEST_CKPT)
        model.load_state_dict(load_state(ckpt, False, device))
        test_root = Path.home() / "Multimodal/data/Testing/data/small_model_track_test"
        raw = build_test_index(test_root)
        crop_p = "bbox_test.json" if args.modality == "main" else "bbox_thermal_test.json"
        import json
        crop_cache = json.loads(Path(crop_p).read_text(encoding="utf-8"))
        if args.modality == "main":
            clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
            ds = DepthIRVideoDataset(clips, T, args.size, False, crop_cache, use_frame_diff=False)
        else:
            clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                    "thermal_dir": ddir.parent / "Thermal"})()
                     for (cid, ddir, _) in raw]
            ds = ThermalVideoDataset(clips, T, args.size, False, crop_cache, use_frame_diff=False)
        loader = make_loader(ds, args.batch_size, args.workers)

        # 1) 先无适应推理
        logit_no = eval_logits(model, loader, device, flip=args.flip)
        pred_no = logit_no.argmax(1)
        # 2) BN 适应后推理
        adapt_bn(model, loader, device, passes=args.adapt_passes)
        logit_ad = eval_logits(model, loader, device, flip=args.flip)
        pred_ad = logit_ad.argmax(1)
        n_flip = int((pred_no != pred_ad).sum())
        print(f"[{args.modality}] BN 适应 {args.adapt_passes}pass: 预测翻转 {n_flip}/{len(pred_no)} "
              f"({n_flip/len(pred_no):.1%}), 类0 {int((pred_ad==0).sum())}", flush=True)

        # 写 CSV（用适应后的）
        test_df = pd.read_csv("data/Testing/test_file/test.csv")
        clip_ids = [d.name for d in sorted(test_root.iterdir())
                    if d.is_dir() and d.name.startswith("SM_test_")]
        import re
        order = test_df["path"].astype(str).map(lambda p: re.search(r"(SM_test_\d+)", p).group(1))
        pred_map = dict(zip(clip_ids, pred_ad.astype(int)))
        test_df["prediction"] = [pred_map.get(k, 0) for k in order]
        out = Path(args.output)
        test_df[["path", "prediction"]].to_csv(out, index=False)
        print(f"BN-TTA CSV saved: {out} ({len(test_df)} rows, 类0={int((test_df['prediction']==0).sum())})", flush=True)


if __name__ == "__main__":
    main()
