#!/usr/bin/env python3
"""Noisy-Student 式伪标签: 用 0.75 锚(强) 在 test 上的 soft 伪标签微调 IMUCNN, 修其跨被试域偏移。

思路(用户提出): 骨架/IMU 判负的唯一根因=对全新 test 被试域无可用信号(域偏移反噬)。
用强模型(0.75锚链 main+th) 在 test 405 的 soft 伪标签 把 IMU 拉向其"test 域意见" →
测试融合从噪声变对齐, 可能从反噬(-0.38)变中性甚至正。

阶段1: 全量训练域硬标签 30ep(balanced) —— 保存 phase1 基线
阶段2: [train 硬标签 交替 + test soft 伪标签] 小 lr 微调 8ep —— 对齐 test 域
产物: outputs/oof/imu_pseudo_test_logits.npy + 微调前后 IMU↔teacher 一致率对照
设定: 伪标签训练后 IMU 见过 test 分布 → 无法离线干净验证 → 只能 LB 实测(1名额)。
"""
import argparse
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

from src.dataset import build_balanced_sampler, build_train_index
from src.imu_dataset import IMUDataset, IMUClipIndex, build_imu_index
from train_imu import IMUCNN


def make_train_opt(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


def run_epoch_hard(model, loader, opt, device):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    run_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        run_loss += loss.item() * y.numel()
        n += y.numel()
    return run_loss / max(n, 1)


def run_epoch_pseudo(model, tr_loader, te_loader, teacher, opt, device, steps):
    """交替: 偶数步=train 硬标签, 奇数步=test soft 伪标签。"""
    model.train()
    crit_h = nn.CrossEntropyLoss(label_smoothing=0.1)
    tr_it = iter(tr_loader)
    te_it = iter(te_loader)
    run_loss, n = 0.0, 0
    for i in range(steps):
        if i % 2 == 0:
            try:
                x, y, _ = next(tr_it)
            except StopIteration:
                tr_it = iter(tr_loader)
                x, y, _ = next(tr_it)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit_h(model(x), y)
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
            p = np.stack([teacher[s] for s in subj])
            p = torch.from_numpy(p).to(device)
            opt.zero_grad()
            logp = F.log_softmax(model(x.to(device)), dim=-1)
            loss = (-(p * logp).sum(-1)).mean()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(x)
            n += len(x)
    return run_loss / max(n, 1)


@torch.no_grad()
def predict_test(model, test_ds, device, T, batch_size):
    model.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    out = np.zeros((len(test_ds), 40), np.float32)
    s = 0
    for x, _, subj in loader:
        lg = model(x.to(device)).cpu().numpy().astype(np.float32)
        out[s:s + len(x)] = lg
        s += len(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--test_root", default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--teacher_pkl", default="outputs/test_teacher_avg_probs.pkl")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--epochs1", type=int, default=30)
    ap.add_argument("--epochs2", type=int, default=8)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--out", default="outputs/oof/imu_pseudo_test_logits.npy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # ---- 训练域 ----
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    tr_clips = build_imu_index(root, main_clips)
    tr_ds = IMUDataset(tr_clips, args.T, True, seed=42, jitter=True)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)

    # ---- test 域 + teacher ----
    troot = Path(args.test_root).expanduser()
    clip_ids = [d.name for d in sorted(troot.iterdir())
                if d.is_dir() and d.name.startswith("SM_test_")]
    te_clips = [IMUClipIndex(-1, cid, cid, troot / cid / "IMU") for cid in clip_ids]
    te_ds = IMUDataset(te_clips, args.T, False)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    raw = pickle.load(open(Path(args.teacher_pkl).expanduser(), "rb"))
    teacher = {re.sub(r"^test:", "", k): np.asarray(v, np.float32) for k, v in raw.items()}
    print(f"train clips={len(tr_clips)} test clips={len(te_clips)} teacher keys={len(teacher)}",
          flush=True)

    model = IMUCNN().to(device)
    # ---- 阶段1: 训练域硬标签 ----
    opt1 = make_train_opt(model, args.lr1)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs1)
    t0 = time.time()
    for ep in range(args.epochs1):
        l = run_epoch_hard(model, tr_loader, opt1, device)
        sched1.step()
        print(f"[p1] ep{ep+1}/{args.epochs1} loss={l:.4f} ({time.time()-t0:.0f}s)", flush=True)
    p1_logits = predict_test(model, te_ds, device, args.T, args.batch_size)
    agree1 = float((p1_logits.argmax(1) == np.argmax([teacher[c] for c in clip_ids], -1)).mean())
    print(f"phase1 test IMU↔teacher 一致率: {agree1:.3f}", flush=True)

    # ---- 阶段2: + test soft 伪标签 微调 ----
    opt2 = make_train_opt(model, args.lr2)
    nsteps = int(np.ceil(len(te_ds) / args.batch_size))
    for ep in range(args.epochs2):
        l = run_epoch_pseudo(model, tr_loader, te_loader, teacher, opt2, device,
                             steps=max(nsteps * 2, 16))
        print(f"[p2] ep{ep+1}/{args.epochs2} loss={l:.4f} ({time.time()-t0:.0f}s)", flush=True)
    p2_logits = predict_test(model, te_ds, device, args.T, args.batch_size)
    agree2 = float((p2_logits.argmax(1) == np.argmax([teacher[c] for c in clip_ids], -1)).mean())
    print(f"phase2 test IMU↔teacher 一致率: {agree2:.3f} (微调后应显著↑=对齐test域)", flush=True)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, p2_logits)
    print(f"saved {out} {p2_logits.shape}", flush=True)


if __name__ == "__main__":
    main()
