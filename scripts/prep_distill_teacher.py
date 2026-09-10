#!/usr/bin/env python3
"""蒸馏教师 logits 提取（训练集/测试集共用）。

teacher = 多条 main seed（或 thermal seed）在无增强推理下的 prob-avg soft 标签。
输出 pkl: { "<action>/<subject>/<sample>": probs[40] float32 }
  - 训练集 clip key = build_train_index 的 (action_id, subject, sample)
  - 测试集 clip key 前缀加 "test:" 防与训练键冲突（供 transductive 蒸馏）
用法:
  python scripts/prep_distill_teacher.py --modality main \
      --ckpts main_s42 main_s777 main_avg --root train \
      --out outputs/teacher_main_train.pkl
  python scripts/prep_distill_teacher.py --modality main \
      --ckpts main_s42 main_s777 --root test \
      --out outputs/teacher_main_test.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalVideoDataset,
                         build_test_index, build_train_index, build_thermal_index)
from src.model import build_model
from scripts.ensemble_inference import load_state, load_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["main", "thermal"], required=True)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="老师权重路径（int5 pack 需加 --quantize；也可给 fp32 路径）")
    ap.add_argument("--root", choices=["train", "test"], default="train")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.num_frames
    if args.modality == "main":
        model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                            n_segment=T).to(device).eval()
    else:
        model = build_model("r2plus1d34", num_classes=40, in_channels=3,
                            n_segment=T).to(device).eval()

    HOME = Path.home() / "Multimodal"
    import json
    if args.root == "train":
        root = HOME / "data/Training/HAR"
        crop_p = "bbox_train.json" if args.modality == "main" else "bbox_thermal_train.json"
        crop_cache = json.loads(Path(crop_p).read_text(encoding="utf-8"))
        if args.modality == "main":
            clips = build_train_index(root)
            ds = DepthIRVideoDataset(clips, T, args.size, False, crop_cache,
                                     use_frame_diff=False)
            keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in clips]
        else:
            clips = build_thermal_index(root)
            ds = ThermalVideoDataset(clips, T, args.size, False, crop_cache,
                                     use_frame_diff=False)
            keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in clips]
    else:
        test_root = HOME / "data/Testing/data/small_model_track_test"
        raw = build_test_index(test_root)
        crop_p = "bbox_test.json" if args.modality == "main" else "bbox_thermal_test.json"
        cmap = load_crop(crop_p, raw)
        if args.modality == "main":
            clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
            ds = DepthIRVideoDataset(clips, T, args.size, False, cmap, use_frame_diff=False)
        else:
            clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                    "thermal_dir": ddir.parent / "Thermal"})()
                     for (cid, ddir, _) in raw]
            ds = ThermalVideoDataset(clips, T, args.size, False, cmap, use_frame_diff=False)
        keys = [f"test:{cid}" for (cid, _, _) in raw]

    N = len(clips)
    logit = np.zeros((N, 40), np.float32)
    for ci, ckpt in enumerate(args.ckpts):
        model.load_state_dict(load_state(ckpt, args.quantize, device))
        model.eval()
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            s = 0
            for x, *_ in loader:
                x = x.to(device)
                o = model(x)
                if args.flip:
                    o = o + model(torch.flip(x, dims=(-1,)))
                # fp32 累加太大会爆显存→软标签渐进平均即可（概率）
                p = torch.softmax(o, -1)
                logit[s:s + len(p)] += p.float().cpu().numpy() / len(args.ckpts)
                s += len(p)
        print(f"[teacher] {ckpt} done ({(ci+1)}/{len(args.ckpts)})", flush=True)

    out = {k: v for k, v in zip(keys, logit.astype(np.float32))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"[teacher] {len(out)} keys -> {args.out} (prob-avg of {len(args.ckpts)})", flush=True)


if __name__ == "__main__":
    main()


