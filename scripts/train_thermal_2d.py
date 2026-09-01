#!/usr/bin/env python3
"""CUHK-X —— 诊断：热像 2D 单分支（确定热像 2D 骨架的动作识别天花板）

背景（2026-08-23）：双分支训练暴露 2D 分支 acc 恒定 ~0.1（接近随机），3D 分支正常 0.4+。
需区分根因：①热像 2D 骨架质量差（数据） vs ②共享 backbone+fused loss 下 3D 主导压制 2D（架构）。
此脚本跑"只喂热像 2D"的单分支，隔离数据因素。

模式：
  default/训练：DSTformer + ActionNet，只读热像 npz [x,y,conf]（与双分支 2D 分支同协议+同增强）。
    → 若 acc 也 ~0.1-0.2：根因是①（热像骨架质量差）；若 acc 到 0.3+：根因是②（共享压制）。
  --mode stats：不训练，扫描全部热像 npz 输出骨架质量统计（关节 conf / 有效比例 / 坐标范围）。
    → 训练前快速判断数据质量。

用法:
    python scripts/train_thermal_2d.py --pretrained weights/mb_lite_latest_epoch.bin --folds 3 --balanced
    python scripts/train_thermal_2d.py --mode stats
"""

import argparse
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.dual_dataset import Thermal2DSkeletonDataset, build_dual_pairs
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.split import split_by_subject

BASE = dict(dim_in=3, dim_out=3, dim_rep=512, depth=5,
            num_heads=8, num_joints=17, maxlen=243)
ARCH_LITE = dict(dim_feat=256, mlp_ratio=4)
ARCH_RELEASE = dict(dim_feat=512, mlp_ratio=2)


def _strip_prefix(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            out[k[len("backbone."):]] = v
        elif k.startswith("head."):
            continue
        else:
            out[k] = v
    return out


def _infer_arch(state: dict) -> dict:
    for k, v in state.items():
        if k.endswith("joints_embed.weight") and v.ndim >= 2:
            return dict(ARCH_RELEASE) if v.shape[0] == 512 else dict(ARCH_LITE)
    return dict(ARCH_LITE)


def load_pretrained_backbone(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_pos" in ckpt:
        state = _strip_prefix(ckpt["model_pos"])
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = _strip_prefix(ckpt["model"])
    else:
        state = _strip_prefix(ckpt) if isinstance(ckpt, dict) else ckpt
    if not state:
        raise RuntimeError(f"权重解析失败：{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    arch = _infer_arch(state)
    print(f"架构 {'MB_release(dim_feat=512)' if arch['dim_feat'] == 512 else 'MB_lite(dim_feat=256)'} "
          f"（{len(state)} 个参数）", flush=True)
    cfg = dict(BASE)
    cfg.update(arch)
    backbone = DSTformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **cfg)
    model_dict = backbone.state_dict()
    matched = discarded = 0
    for k, v in state.items():
        if k in model_dict and model_dict[k].size() == v.size():
            model_dict[k] = v
            matched += 1
        else:
            discarded += 1
    backbone.load_state_dict(model_dict)
    print(f"pretrained loaded: matched={matched} discarded={discarded}", flush=True)
    return backbone


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device).unsqueeze(1)
        y = y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_fold(model, train_loader, val_loader, device, epochs, fold, save_path,
               lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.01,
               label_smoothing=0.1, patience=8):
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.head.parameters(), "lr": lr_head},
    ], lr=lr_backbone, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best = 0.0
    no_improve = 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device).unsqueeze(1)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate(model, val_loader, device)
        if acc > best:
            best = acc
            no_improve = 0
            if save_path is not None:
                torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= patience:
            print(f"[fold{fold}] early stop @ ep{ep+1}（{patience} ep 无提升）", flush=True)
            break
    return best


def run_stats(root: Path, thermal_out: Path):
    """扫描热像 npz：关节 conf / 有效比例 / 坐标范围 / 帧数 —— 训练前判断数据质量。"""
    pairs, _ = build_dual_pairs(root, thermal_out)
    print(f"==== 热像 2D 骨架质量统计（{len(pairs)} clip）====", flush=True)
    confs, nframes, kp_std = [], [], []
    for i, c in enumerate(pairs):
        npz = Path(thermal_out) / f"{c.action_id}/{c.subject}/{c.sample}.npz"
        z = np.load(npz)
        kp, conf = z["kp"], z["conf"]
        nframes.append(kp.shape[0])
        confs.append(conf)
        kp_std.append(kp.std())
        if (i + 1) % 500 == 0 or i == len(pairs) - 1:
            print(f"  scanned {i+1}/{len(pairs)}", flush=True)
    cf = np.concatenate(confs)
    ks = np.array(kp_std)
    nf = np.array(nframes)
    joint_conf = np.concatenate([np.mean(c, axis=0)[None, :] for c in confs], 0).mean(0)
    print(f"  clip 帧数: mean={nf.mean():.0f} min={nf.min()} max={nf.max()}", flush=True)
    print(f"  kp 坐标 std: mean={ks.mean():.4f}（<0.05 提示坐标退化/静止）", flush=True)
    print(f"  conf: mean={cf.mean():.4f} >0.5占比={(cf>0.5).mean():.4f} "
          f"<0.0占比={(cf<0).mean():.4f}", flush=True)
    print(f"  每关节 conf 均值(H36M 顺序): "
          f"{np.round(joint_conf, 3).tolist()}", flush=True)
    # 每关节有效比例（conf>0.5）
    eff = np.concatenate([(np.mean(c, axis=0) > 0.5)[None, :] for c in confs], 0).mean(0)
    print(f"  每关节有效比例(conf均值>0.5): {np.round(eff, 3).tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--thermal_out", type=str, default="outputs/thermal_skeleton")
    ap.add_argument("--pretrained", type=str, default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "stats"])
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--conf_norm", type=str, default="sigmoid",
                    choices=["sigmoid", "none"], help="热像 conf 归一化")
    ap.add_argument("--save_dir", type=str, default="outputs/thermal_2d")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    thermal_out = Path(args.thermal_out).expanduser()

    if args.mode == "stats":
        run_stats(root, thermal_out)
        return

    pairs, _ = build_dual_pairs(root, thermal_out)
    print(f"device={device} 热像 2D 单分支 clips={len(pairs)} conf_norm={args.conf_norm}", flush=True)
    folds = split_by_subject(pairs, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    accs = []
    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_clips = [pairs[i] for i in tr_idx]
        va_clips = [pairs[i] for i in va_idx]
        tr_ds = Thermal2DSkeletonDataset(tr_clips, thermal_out, args.num_frames, True,
                                         conf_norm=args.conf_norm)
        va_ds = Thermal2DSkeletonDataset(va_clips, thermal_out, args.num_frames, False,
                                         conf_norm=args.conf_norm)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               shuffle=sampler is None,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
        backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
        model = ActionNet(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                          dropout_ratio=0.5, version="class",
                          hidden_dim=args.hidden_dim, num_joints=17).to(device)
        if fi == target[0]:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"ActionNet total params ~{n_params/1e6:.1f}M", flush=True)
        save_path = save_dir / f"motionbert_fold{fi}.pth"
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, fi, save_path,
                          label_smoothing=args.label_smoothing, patience=args.patience)
        accs.append(best)
        print(f"== fold {fi} best val = {best:.4f} -> {save_path}")

    if len(accs) > 1:
        print(f"\n==== mean val across {len(accs)} folds: {np.mean(accs):.4f} "
              f"(std {np.std(accs):.4f}) ====", flush=True)


if __name__ == "__main__":
    main()
