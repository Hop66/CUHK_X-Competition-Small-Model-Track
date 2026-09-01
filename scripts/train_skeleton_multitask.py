#!/usr/bin/env python3
"""CUHK-X —— 骨架 multi-task 混合训练（NTU A001-A040 外部补充数据）。

思路：NTU 40 类与我们 40 类语义不对齐（只有 ~13 类一致），不能直接共享标签。
     → 共享 MotionBERT backbone(DSTformer) + 两个分类头（我们 40 类 / NTU 40 类），
       各自样本走各自头，encoder 共享 → NTU 37,920 样本(40 被试)训练跨被试运动表示。

对照：骨架基线（MB_lite, 2891 样本）= fold0 ~0.45-0.51
判读：fold0 > 0.51 则 NTU 补充有效 → 全量骨架 + NTU 重训 → 参与三模态融合

用法:
    python scripts/train_skeleton_multitask.py --fold 0 --ntu_root data/external/ntu/ntu60
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import train_skeleton_motionbert

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler
from src.motionbert.action_net import ActionHeadClassification
from src.skeleton_dataset import MotionBertSkeletonDataset, build_skeleton_index
from src.split import split_by_subject
from src.ntu_dataset import NTUSkeletonDataset

from train_skeleton_motionbert import BASE, load_pretrained_backbone


class MultiTaskActionNet(nn.Module):
    """共享 backbone + 我们 40 类头 + NTU 40 类头。"""

    def __init__(self, backbone, dim_rep: int = 512, n_ours: int = 40, n_ntu: int = 40,
                 hidden_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.head_ours = ActionHeadClassification(0.5, dim_rep, n_ours, 17, hidden_dim)
        self.head_ntu = ActionHeadClassification(0.5, dim_rep, n_ntu, 17, hidden_dim)

    def _feat(self, x):
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        feat = self.backbone.get_representation(x)          # (N*M, T, J, dim_rep)
        return feat.reshape(N, M, T, J, -1)

    def forward_ours(self, x):
        return self.head_ours(self._feat(x))

    def forward_ntu(self, x):
        return self.head_ntu(self._feat(x))


@torch.no_grad()
def evaluate_ours(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x = x.to(device).unsqueeze(1)
        y = y.to(device)
        correct += (model.forward_ours(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ntu_root", type=str, default="data/external/ntu/ntu60",
                    help="NTU 骨架根目录（含 *.skeleton）")
    ap.add_argument("--pretrained", type=str, default="weights/mb_lite_latest_epoch.bin")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--ntu_weight", type=float, default=0.5,
                    help="NTU CE 损失权重（防辅助数据主导）")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="outputs/skeleton_multitask")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    folds = split_by_subject(clips, n_folds=args.folds)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    # 我们数据（fold0，H3.6M-17）
    tr_ours = MotionBertSkeletonDataset(tr_clips, args.num_frames, True, input3d=True)
    va_ours = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=True)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips])
    ours_loader = DataLoader(tr_ours, batch_size=args.batch_size, sampler=sampler,
                             num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ours, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    # NTU 数据（A001-A040）
    ntu_ds = NTUSkeletonDataset(args.ntu_root, args.num_frames, True, seed=42)
    ntu_loader = DataLoader(ntu_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"device={device} ours_clips={len(tr_clips)} ntu_files={len(ntu_ds)} "
          f"ours_steps={len(ours_loader)} ntu_steps={len(ntu_loader)}", flush=True)

    backbone = load_pretrained_backbone(Path(args.pretrained).expanduser(), device)
    model = MultiTaskActionNet(backbone, dim_rep=BASE["dim_rep"], n_ours=40, n_ntu=40,
                               hidden_dim=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MultiTask params ~{n_params/1e6:.1f}M (fp32 ~{n_params*4/1e6:.0f}MB)", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head_ours.parameters(), "lr": args.lr_head},
        {"params": model.head_ntu.parameters(), "lr": args.lr_head},
    ], lr=args.lr_backbone, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"multitask_fold{args.fold}.pth"

    best, no_improve = 0.0, 0
    steps_per_ep = min(len(ours_loader), len(ntu_loader))
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        it_o, it_n = iter(ours_loader), iter(ntu_loader)
        run_loss, n = 0.0, 0
        for _ in range(steps_per_ep):
            xo, yo, _ = next(it_o)
            xn, yn, _ = next(it_n)
            xo, yo = xo.to(device).unsqueeze(1), yo.to(device)
            xn, yn = xn.to(device).unsqueeze(1), yn.to(device)
            opt.zero_grad()
            lo = crit(model.forward_ours(xo), yo)
            ln = args.ntu_weight * crit(model.forward_ntu(xn), yn)
            (lo + ln).backward()
            opt.step()
            run_loss += (lo.item() + ln.item()) * yo.numel()
            n += yo.numel()
        sched.step()
        acc = evaluate_ours(model, va_loader, device)
        if acc > best:
            best, no_improve = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best,
                        "head": "ours"}, save_path)
        else:
            no_improve += 1
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
              f"ours_val={acc:.4f} best={best:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= args.patience:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== multi-task fold{args.fold} best ours_val = {best:.4f} "
          f"（对照骨架基线 0.51；>0.51 则 NTU 补充有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
