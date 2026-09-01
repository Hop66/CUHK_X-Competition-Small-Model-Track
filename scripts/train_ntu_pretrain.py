#!/usr/bin/env python3
"""CUHK-X —— 阶段1：NTU 骨架预训练（域对齐 17 关节，A001-A040 全量 37,920）。

方法本质：骨架预训练让 backbone 吸收 NTU 40 被试的运动多样性（跨被试运动表征），
微调阶段只需适配我们域（比 multi-task/混合训练干净）。

特化：
  1. 25→17 关节映射 + 中心化 + 肩宽归一化（与我们数据完全一致 → 域对齐）
  2. lr 大（1e-3 backbone）+ StepLR（transformer 微调标准，不衰减到 0）
  3. 保存 backbone（DSTformer）state_dict，供阶段2微调

用法:
    python scripts/train_ntu_pretrain.py --ntu_root data/external/ntu/ntu60 \
        --pretrained weights/mb_lite_latest_epoch.bin --save_dir outputs/ntu_pretrain
"""

import argparse
import re
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.motionbert.action_net import ActionHeadClassification
from src.motionbert.dstformer import DSTformer
from src.ntu_dataset import NTUSkeletonDataset

from train_skeleton_motionbert import BASE, load_pretrained_backbone

FNAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


class NTUModel(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=40, hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        self.head = ActionHeadClassification(0.5, dim_rep, num_classes, 17, hidden_dim)

    def forward(self, x):
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        feat = self.backbone.get_representation(x)
        feat = feat.reshape(N, M, T, J, -1)
        return self.head(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntu_root", default="data/external/ntu/ntu60")
    ap.add_argument("--pretrained", default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/ntu_pretrain")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NTUSkeletonDataset(args.ntu_root, args.num_frames, True, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"NTU 预训练: {len(ds)} 样本 (A001-A040) steps/ep={len(loader)}", flush=True)

    backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
    model = NTUModel(backbone, dim_rep=BASE["dim_rep"], num_classes=40, hidden_dim=512).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * 10},
    ], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ntu_pretrained_backbone.pth"
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device).unsqueeze(1), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        print(f"[ntu-pretrain] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)", flush=True)

    # 只保存 backbone（DSTformer），供阶段2微调
    torch.save({"backbone": model.backbone.state_dict(),
                "dim_rep": BASE["dim_rep"]}, save_path)
    print(f"== NTU 预训练完成 → {save_path}（backbone 已保存）", flush=True)


if __name__ == "__main__":
    main()
