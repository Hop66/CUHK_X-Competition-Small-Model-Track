#!/usr/bin/env python3
"""main(R2+1D 4ch) fold0 复核：给定 n_segment ckpt 在 fold0 val 上推理 acc。

用于 main_nf32(32帧) 这类训练器只存裸 state_dict、out 又没打印 best 的场景。
用法:
  python scripts/eval_main_fold0.py --ckpt outputs/main_nf32/r2plus1d34_depthir_fold0.pth \
      --num_frames 32 [--train_root ...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import DepthIRVideoDataset, build_train_index
from src.model import build_model
from src.split import split_by_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=3)
    _, va_idx = folds[args.fold]
    va_clips = [clips[i] for i in va_idx]
    crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop,
                             use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, timeout=300)
    model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                        n_segment=args.num_frames).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    print(f"== main fold{args.fold} nf={args.num_frames} 复核 acc = "
          f"{correct / max(total, 1):.4f} (对照 16f=0.6695) ==", flush=True)


if __name__ == "__main__":
    main()
