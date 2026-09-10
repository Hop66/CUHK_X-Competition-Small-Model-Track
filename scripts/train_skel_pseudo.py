#!/usr/bin/env python3
"""Noisy-Student 式 骨架伪标签微调(用户: 骨架也试同款)。

动机: 骨架(MotionBERT-ActionNet)判负根因=对全新 test 被试域无信号(域反噬)。
用 0.75锚 soft 伪标签(test_teacher_avg_probs.pkl) 把骨架从"训练域姿态"拉向"test 域意见" → 融合从反噬(-0.38)转中性/正。

阶段1: init = 全量骨架 finetune_full.pth(已有, 全量train 40ep)  —— 只做 sanity 打印
阶段2: [train 硬标签 交替 + test soft 伪标签] 小 lr 微调 6ep
产物: outputs/oof/skel_pseudo_test_logits.npy + 微调前后 skeleton↔teacher 一致率
硬伤: 见过 test 分布 → 无法离线验证 → 只能 LB 实测。
"""
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.skeleton_dataset import MotionBertSkeletonDataset, SkeletonClipIndex, build_skeleton_index
from src.split import split_by_subject
from train_skeleton_motionbert import ActionNet, BASE, load_pretrained_backbone


def load_full_state(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        return sd["model"]
    return sd


def build_test_skeleton_clips(test_root):
    clips = []
    for d in sorted(Path(test_root).expanduser().iterdir()):
        if not d.is_dir() or not d.name.startswith("SM_test_"):
            continue
        skel = d / "Skeleton"
        pred = skel / "predictions" if (skel / "predictions").is_dir() else skel
        clips.append(SkeletonClipIndex(-1, d.name, d.name, pred))
    return clips


@torch.no_grad()
def predict(model, ds, device, batch_size=64):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=True)
    out = np.zeros((len(ds), 40), np.float32)
    s = 0
    for x, _, _ in loader:
        lg = model(x.to(device).unsqueeze(1)).float().cpu().numpy()
        out[s:s + len(x)] = lg
        s += len(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--test_root", default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--init_ckpt", default="outputs/skeleton_finetune_full/finetune_full.pth")
    ap.add_argument("--teacher_pkl", default="outputs/test_teacher_avg_probs.pkl")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs2", type=int, default=6)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/oof/skel_pseudo_test_logits.npy")
    ap.add_argument("--max_samples", type=int, default=0, help="冒烟用(>0 只跑前N train/test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    troot = Path(args.train_root).expanduser()
    clips = build_skeleton_index(troot)
    if args.max_samples > 0:
        clips = clips[:args.max_samples]
    tr_ds = MotionBertSkeletonDataset(clips, args.num_frames, True, input3d=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=False)

    test_clips = build_test_skeleton_clips(args.test_root)
    if args.max_samples > 0:
        test_clips = test_clips[:args.max_samples]
    clip_ids = [c.sample for c in test_clips]
    idx_map = {c: i for i, c in enumerate(clip_ids)}
    te_ds = MotionBertSkeletonDataset(test_clips, args.num_frames, False, input3d=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    raw = pickle.load(open(args.teacher_pkl, "rb"))
    teacher = {re.sub(r"^test:", "", k): np.asarray(v, np.float32) for k, v in raw.items()}
    tgt = np.stack([teacher[c] for c in clip_ids])
    print(f"train clips={len(clips)} test clips={len(test_clips)}", flush=True)

    backbone = load_pretrained_backbone(Path("weights/mb_lite_latest_epoch.bin").expanduser(),
                                        device)
    model = ActionNet(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                      dropout_ratio=0.5, version="class", hidden_dim=512, num_joints=17).to(device)
    model.load_state_dict(load_full_state(Path(args.init_ckpt).expanduser(), device))
    print(f"loaded init {args.init_ckpt}", flush=True)

    before = predict(model, te_ds, device, args.batch_size)
    agree1 = float((before.argmax(1) == tgt.argmax(1)).mean())
    print(f"phase1(init) test skel↔teacher 一致率: {agree1:.3f}", flush=True)

    # 阶段2: 交替 train 硬 / test soft
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr2, weight_decay=1e-4)
    crit_h = nn.CrossEntropyLoss(label_smoothing=0.1)
    tr_it = iter(tr_loader)
    te_it = iter(te_loader)
    n_te = (len(te_ds) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for ep in range(args.epochs2):
        run_loss, n = 0.0, 0
        for i in range(n_te * 2):
            if i % 2 == 0:
                try:
                    x, y, _ = next(tr_it)
                except StopIteration:
                    tr_it = iter(tr_loader)
                    x, y, _ = next(tr_it)
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit_h(model(x.unsqueeze(1)), y)
                loss.backward()
                opt.step()
                run_loss += loss.item() * y.numel()
                n += y.numel()
            else:
                try:
                    x, _, subj = next(te_it)
                except StopIteration:
                    te_it = iter(te_loader)
                    x, _, subj = next(te_it)
                p = tgt[[idx_map[s] for s in subj]]
                p = torch.from_numpy(p).to(device)
                opt.zero_grad()
                logp = F.log_softmax(model(x.to(device).unsqueeze(1)), dim=-1)
                loss = (-(p * logp).sum(-1)).mean()
                loss.backward()
                opt.step()
                run_loss += loss.item()
                n += 1
        print(f"[p2] ep{ep+1}/{args.epochs2} loss={run_loss/max(n,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    after = predict(model, te_ds, device, args.batch_size)
    agree2 = float((after.argmax(1) == tgt.argmax(1)).mean())
    print(f"phase2 test skel↔teacher 一致率: {agree2:.3f} (↑=dia对齐test域)", flush=True)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, after)
    print(f"saved {out} {after.shape}", flush=True)


if __name__ == "__main__":
    main()
