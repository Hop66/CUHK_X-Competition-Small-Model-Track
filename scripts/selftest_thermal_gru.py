#!/usr/bin/env python3
"""th_gru(ResNet2D+BiGRU+Attention) thermal-only 快速自验。

复刻 selftest_thermal_val.py 的纪律：用训练时 fold0 val 复核"推理链路"
(ckpt 加载 / crop / dataset / 归一化 / 帧采样) 是否与训练收敛一致。
判读: 复核 acc ≈ ckpt['best_acc'](±0.5pt) → 链路 OK, 可全量/上 LB。
用法:
  python scripts/selftest_thermal_gru.py --ckpt outputs/th_gru/th_gru_fold0.pth \
      --num_frames 16 [--flip_tta]
注意: 推理/测试阶段 crop 用 bbox_thermal_test.json; 训练自验用 bbox_thermal_train.json。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ThermalClipIndex, ThermalVideoDataset, build_train_index
from src.split import split_by_subject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_thermal_gru_attn import ThermalGRUNet  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, flip_tta=False):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device)
        p = torch.softmax(model(x), -1)
        if flip_tta:
            p = (p + torch.softmax(model(torch.flip(x, dims=(4,))), -1)) / 2
        correct += (p.argmax(-1).cpu() == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=3)
    tr_idx, va_idx = folds[args.fold]
    va_clips = [clips[i] for i in va_idx]
    th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                 root / "Thermal" / c.depth_dir.parent.parent.name /
                                 c.subject / c.sample) for c in va_clips]
    crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
    ds = ThermalVideoDataset(th_clips, args.num_frames, args.size, False, crop)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, timeout=300)
    model = ThermalGRUNet().to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"ckpt best_acc(训练)={ck.get('best_acc', '?')} epoch={ck.get('epoch', '?')} "
          f"n_frames={args.num_frames}", flush=True)
    acc = evaluate(model, loader, device, flip_tta=args.flip_tta)
    print(f"== th_gru fold{args.fold} 复核 acc = {acc:.4f} (crop={args.thermal_crop}) ==", flush=True)


if __name__ == "__main__":
    main()
