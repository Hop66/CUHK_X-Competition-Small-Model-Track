#!/usr/bin/env python3
"""
CUHK-X —— Skeleton MotionBERT 预训练微调（H3.6M-17 拓扑，2D+conf 输入）

- MotionBERT-Lite 预训练权重（dim_feat=256, mlp_ratio=4）初始化 DSTformer
- ActionNet 40 类分类头，lr_backbone=1e-4 / lr_head=1e-3（AdamW + 指数衰减）
- subject-fold 3 折交叉验证

用法:
    python scripts/train_skeleton_motionbert.py --pretrained weights/mb_lite_latest_epoch.bin
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.skeleton_smooth import SmoothMotionBertSkeletonDataset
from src.split import split_by_subject

# MotionBERT 架构超参（dim_feat/mlp_ratio 因预训练权重而异，其余固定）
BASE = dict(dim_in=3, dim_out=3, dim_rep=512, depth=5,
            num_heads=8, num_joints=17, maxlen=243)
ARCH_LITE = dict(dim_feat=256, mlp_ratio=4)     # MB_lite (H36M/AMASS 预训练, 61MB)
ARCH_RELEASE = dict(dim_feat=512, mlp_ratio=2)  # MB_release / NTU 动作权重 (162MB)


def _strip_prefix(state: dict) -> dict:
    """剥离 module./backbone./head. 前缀，得到 DSTformer backbone 的 state_dict。

    兼容三种来源：
      - DataParallel 保存（module.backbone.xxx / module.head.xxx）
      - ActionNet 直接保存（backbone.xxx / head.xxx）
      - 裸 backbone state_dict（joints_embed.weight 等）
    """
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            out[k[len("backbone."):]] = v
        elif k.startswith("head."):
            continue  # 丢弃分类头（我们自建 40 类 head）
        else:
            out[k] = v
    return out


def _infer_arch(state: dict) -> dict:
    """按 joints_embed.weight 的 out_features 推断架构（512=MB_release, 256=MB_lite）。

    比按 key 格式推断更可靠：无论权重是 model_pos / model+backbone. / 裸 state_dict，
    都能根据实际尺寸确定架构，避免把 MB_lite 误判为 MB_release。
    """
    for k, v in state.items():
        if k.endswith("joints_embed.weight") and v.ndim >= 2:
            return dict(ARCH_RELEASE) if v.shape[0] == 512 else dict(ARCH_LITE)
    return dict(ARCH_LITE)


def load_pretrained_backbone(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # 提取 backbone state_dict（兼容三种格式）
    if isinstance(ckpt, dict) and "model_pos" in ckpt:
        # MotionBERT 官方预训练 checkpoint（key=model_pos），可能带 module. 前缀（DataParallel）
        state = _strip_prefix(ckpt["model_pos"])
        kind = "MB_lite 预训练(model_pos)"
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        # ActionNet checkpoint（NTU 动作权重 / 我们自己训练的），剥离 module./backbone./head.
        state = _strip_prefix(ckpt["model"])
        kind = "ActionNet 动作权重(model)"
    else:
        state = _strip_prefix(ckpt) if isinstance(ckpt, dict) else ckpt
        kind = "裸 state_dict"

    if not state:
        raise RuntimeError(
            f"权重解析失败：未提取到任何 backbone 参数。"
            f"ckpt 结构: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    arch = _infer_arch(state)
    arch_name = "MB_release(dim_feat=512)" if arch["dim_feat"] == 512 else "MB_lite(dim_feat=256)"
    print(f"检测到 {kind}，架构 {arch_name}（{len(state)} 个参数）")

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
    print(f"pretrained loaded: matched={matched} discarded={discarded} "
          f"(total params ~{sum(p.numel() for p in backbone.parameters())/1e6:.1f}M)")
    return backbone


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device).unsqueeze(1)   # [N,T,17,3] -> [N,1,T,17,3]
        y = y.to(device)
        out = model(x)
        correct += (out.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_fold(model, train_loader, val_loader, device, epochs, fold,
               save_path, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.01,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--pretrained", type=str, default="weights/mb_lite_latest_epoch.bin",
                    help="MB_lite 预训练(82MB fp32 符合约束)；也可传 NTU 权重自动检测")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8, help="val 连续无提升即早停")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--repr", type=str, default="3d", choices=["2d", "3d"],
                    help="骨架输入表示：3d=原生3D[水平,垂直,深度]（推荐，匹配 model_pos 3D 预训练）；2d=旧[x,y,conf]")
    ap.add_argument("--smooth", type=int, default=1,
                    help="骨架时间平滑窗口（默认 1=关，行为不变；5=0.5s 滑动平均去噪，可复位）")
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_mb")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    print(f"device={device} clips={len(clips)} "
          f"subjects={len(set(c.subject for c in clips))} "
          f"classes={len(set(c.action_id for c in clips))}")

    folds = split_by_subject(clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    accs = []
    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_clips = [clips[i] for i in tr_idx]
        va_clips = [clips[i] for i in va_idx]
        input3d = (args.repr == "3d")
        if args.smooth > 1:
            tr_ds = SmoothMotionBertSkeletonDataset(tr_clips, args.num_frames, True,
                                                    input3d=input3d, smooth_window=args.smooth)
            va_ds = SmoothMotionBertSkeletonDataset(va_clips, args.num_frames, False,
                                                    input3d=input3d, smooth_window=args.smooth)
        else:
            tr_ds = MotionBertSkeletonDataset(tr_clips, args.num_frames, True, input3d=input3d)
            va_ds = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=input3d)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               shuffle=sampler is None,  # 无 sampler 时也必须重排，否则同一样本增强每 epoch 确定性
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        # 每个 fold 重新加载预训练权重并重建模型（避免跨 fold 状态泄漏）
        backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
        model = ActionNet(backbone=backbone, dim_rep=BASE["dim_rep"], num_classes=40,
                          dropout_ratio=0.5, version="class",
                          hidden_dim=args.hidden_dim, num_joints=17).to(device)
        if fi == target[0]:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"ActionNet total params ~{n_params/1e6:.1f}M "
                  f"(fp32 ~{n_params*4/1e6:.0f}MB)")

        save_path = save_dir / f"motionbert_fold{fi}.pth"
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, fi, save_path,
                          label_smoothing=args.label_smoothing, patience=args.patience)
        accs.append(best)
        print(f"== fold {fi} best val = {best:.4f} -> {save_path}")

    if len(accs) > 1:
        print(f"\n==== mean val across {len(accs)} folds: {np.mean(accs):.4f} "
              f"(std {np.std(accs):.4f}) ====")


if __name__ == "__main__":
    main()
