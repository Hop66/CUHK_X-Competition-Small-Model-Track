#!/usr/bin/env python3
"""CUHK-X —— main+thermal 双流特征融合（特征级中融合，真正没做过的融合层级）。

方法本质：多模态融合三层次——输入级(main 已 concat Depth+IR)/特征级(本脚本)/决策级(0.73 logits 平均)。
特征级融合让模型**内部学到跨模态交互**（thermal 独立视角如何补充 Depth+IR），
比事后 logits 平均更强（不丢失模态间相关性）。

特化到我们任务（2891 小数据、cross-subject、100MB）：
  1. 双 backbone 均 IG65M 预训练（强先验，防小数据过拟合）
  2. 融合用简单 concat + MLP（不引入复杂注意力，避免小数据过拟合）
  3. 特征取各自 GAP 后 [B,512] → concat [B,1024] → Linear(40)
  4. 低 lr + 早停（双流联合训练小数据易崩）

对照：main+thermal logits 平均 = 0.73（LB）
判读：fold0 融合 acc > 0.6609(main) 且 > 0.6339(thermal) 且优于 logits 平均 → 特征级融合有效

大小：2×62M fp32=124M，int5 ~80MB ✅（与分开打包一样）

用法:
    python scripts/train_dual_fusion.py --fold 0 --weights ig65m_r2plus1d34.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalClipIndex,
                         ThermalVideoDataset, build_train_index)
from src.model import R2Plus1D34
from src.split import split_by_subject


class DualDataset(Dataset):
    """同一 clip 的 main(4ch) + thermal(3ch) 配对输出。"""

    def __init__(self, main_clips, num_frames, size, is_train, main_crop, th_crop):
        th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                     Path(".") / "Thermal" / c.depth_dir.parent.parent.name
                                     / c.subject / c.sample) for c in main_clips]
        self.ds_m = DepthIRVideoDataset(main_clips, num_frames, size, is_train,
                                        main_crop, use_frame_diff=False)
        self.ds_t = ThermalVideoDataset(th_clips, num_frames, size, is_train,
                                        th_crop, use_frame_diff=False)

    def __len__(self):
        return len(self.ds_m)

    def __getitem__(self, i):
        xm, y, _ = self.ds_m[i]          # [T,4,H,W]
        xt, _, _ = self.ds_t[i]          # [T,3,H,W]
        return xm, xt, y


class DualFusion(nn.Module):
    """main/thermal 双流 → GAP 特征 concat → MLP。

    mod_drop: 模态 dropout（训练时以 p 随机丢弃单模态特征），
    让每个单流独立变强 + 模态鲁棒（多模态融合经典技巧，防过拟合）。
    """

    def __init__(self, weights_path, mod_drop: float = 0.5):
        super().__init__()
        self.main = R2Plus1D34(40, 4, weights_path)
        self.th = R2Plus1D34(40, 3, weights_path)
        self.fuse = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 40))
        self.mod_drop = mod_drop

    def forward(self, xm, xt, is_train: bool = False):
        fm = self.main.encoder(xm.permute(0, 2, 1, 3, 4))   # [B,512]
        ft = self.th.encoder(xt.permute(0, 2, 1, 3, 4))     # [B,512]
        if is_train and self.mod_drop > 0:
            if torch.rand(1).item() < self.mod_drop:
                fm = torch.zeros_like(fm)      # 丢 main（学 thermal 独立）
            if torch.rand(1).item() < self.mod_drop:
                ft = torch.zeros_like(ft)      # 丢 thermal（学 main 独立）
        return self.fuse(torch.cat([fm, ft], -1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xm, xt, y in loader:
        xm, xt, y = xm.to(device), xt.to(device), y.to(device)
        correct += (model(xm, xt).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mod_drop", type=float, default=0.5,
                    help="模态 dropout 概率（训练时随机丢弃单模态，提升鲁棒+防过拟合）")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--save_dir", type=str, default="outputs/dual_fusion")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]
    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))

    tr_ds = DualDataset(tr_clips, args.num_frames, args.size, True, main_crop, th_crop)
    va_ds = DualDataset(va_clips, args.num_frames, args.size, False, main_crop, th_crop)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    print(f"device={device} train={len(tr_ds)} val={len(va_ds)} "
          f"steps/ep={len(tr_loader)}", flush=True)

    model = DualFusion(args.weights, mod_drop=args.mod_drop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DualFusion params ~{n_params/1e6:.0f}M "
          f"(fp32 ~{n_params*4/1e6:.0f}MB → int5 ~{n_params*4/1e6/8:.0f}MB)", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.main.parameters(), "lr": args.lr},       # backbone 1e-4 保 IG65M
        {"params": model.th.parameters(), "lr": args.lr},         # backbone 1e-4 保 IG65M
        {"params": model.fuse.parameters(), "lr": args.lr * 10},  # 融合头 1e-3 新学要快
    ], lr=args.lr, weight_decay=1e-4)
    # StepLR 不衰减到 0（CosineAnnealing 在 50ep 内衰减到 ~0，后 20ep 不学习 → 训练无效）
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"dualfusion_fold{args.fold}.pth"
    best, no_improve = 0.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for xm, xt, y in tr_loader:
            xm, xt, y = xm.to(device), xt.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(xm, xt, is_train=True), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate(model, va_loader, device)
        if acc > best:
            best, no_improve = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"fusion_val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= args.patience:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== 双流融合 fold{args.fold} best val = {best:.4f} "
          f"（对照 main 0.6609/thermal 0.6339/融合 0.73；>0.68 则特征级融合有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
