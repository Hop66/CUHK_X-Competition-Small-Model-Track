#!/usr/bin/env python3
"""骨架(MotionBert finetune) fold0 val OOF logits 提取 —— 供「低权重 α 插值」验证。

产出: outputs/oof/skel_fold0.pkl = { "<action>/<subj>/<sample>": logits[40] }
与 main_oof.pkl / thermal_oof.pkl 同格式、同 key 规则, 可直接对齐做 α 加权融合。
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.split import split_by_subject
from train_skeleton_motionbert import ActionNet, BASE, load_pretrained_backbone


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ckpt", type=str, default="outputs/skeleton_finetune/finetune_fold0.pth")
    ap.add_argument("--pretrained", type=str, default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--out", type=str, default="outputs/oof/skel_fold0.pkl")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0, help=">0 只跑前 N 个样本(冒烟)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)

    folds = split_by_subject(clips, n_folds=3)
    tr_idx, va_idx = folds[args.fold]
    va_clips = [clips[i] for i in va_idx]
    print(f"device={device} fold{args.fold} val clips={len(va_clips)}", flush=True)

    va_ds = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=True)
    if args.max_samples > 0:
        va_ds = torch.utils.data.Subset(va_ds, list(range(min(args.max_samples, len(va_ds)))))
    loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
    model = ActionNet(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                      dropout_ratio=0.5, version="class",
                      hidden_dim=args.hidden_dim, num_joints=17).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd["model"])
    model.eval()
    print(f"loaded {args.ckpt} (best_acc={sd.get('best_acc', '?')})", flush=True)

    out, t0 = {}, time.time()
    for bi, (x, y, _) in enumerate(loader):
        x = x.to(device).unsqueeze(1)          # [N,T,17,3] -> [N,1,T,17,3]
        lg = model(x).cpu().numpy().astype(np.float32)
        base = bi * args.batch_size
        for j in range(len(x)):
            clip = va_clips[base + j]
            out[f"{clip.action_id}/{clip.subject}/{clip.sample}"] = lg[j]
        print(f"batch {bi+1}/{len(loader)} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)
    print(f"saved {args.out} n={len(out)} acc={np.mean([np.argmax(v)==int(k.split('/')[0]) for k,v in out.items()]):.4f}", flush=True)


if __name__ == "__main__":
    main()
