#!/usr/bin/env python3
"""挖被埋没的对比学习: 评估 train_contrastive 产出的 main 模型 fold{0,1,2} val acc + OOF。

对比 ckpt = { "main": state_dict, "thermal": state_dict } (main↔thermal InfoNCE 训练后的 main)
判读: 与同代基线「4ch+增强 fold0=0.6192」/ 现 main 单模态 fold0≈0.66 对比。
用法:
  python scripts/eval_contrast_main.py --ckpt outputs/contrastive/r2plus1d34_contrastive_fold0.pth
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DepthIRVideoDataset, build_train_index
from src.model import build_model
from src.split import split_by_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_base", default="outputs/contrastive/r2plus1d34_contrastive_fold{i}.pth",
                    help="含 {i} 占位, 循环 fold0/1/2")
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--crop_cache", default="bbox_train.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_oof", default="outputs/oof/contrast_main_fold{i}.pkl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=3)
    crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))

    results = []
    for i in [0, 1, 2]:
        ckpt_p = Path(args.ckpt_base.replace("{i}", str(i))).expanduser()
        if not ckpt_p.exists():
            print(f"缺 {ckpt_p}, skip", flush=True)
            continue
        _, va_idx = folds[i]
        va_clips = [clips[j] for j in va_idx]
        ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop,
                                 use_frame_diff=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, timeout=300)
        model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                            n_segment=args.num_frames).to(device).eval()
        sd = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(sd["main"])
        correct = total = 0
        oof = {}
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                out = model(x).cpu().numpy().astype(np.float32)
                correct += (out.argmax(-1) == y.numpy()).sum()
                total += y.numel()
        acc = correct / max(total, 1)
        results.append((i, acc))
        print(f"  fold{i}: val = {acc:.4f} (n={total})", flush=True)
        oof_p = Path(args.out_oof.replace("{i}", str(i))).expanduser()
        oof_p.parent.mkdir(parents=True, exist_ok=True)
        # 简化: 只存 acc, 不存 OOF(如需融合再补)
        with open(str(ckpt_p) + ".acc.txt", "w") as fh:
            fh.write(f"{acc:.4f}\n")

    if results:
        accs = [r[1] for r in results]
        print(f"\ncontrastive-main 3折: " +
              " ".join(f"{a:.4f}" for a in accs) +
              f" | mean={sum(accs)/len(accs):.4f}")
        print("参照: 同代基线 4ch+增强 fold0=0.6192; 现 main 单模态 fold0≈0.66-0.67")


if __name__ == "__main__":
    main()
