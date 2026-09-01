#!/usr/bin/env python3
"""CUHK-X —— 辅助 3D 位姿头（"3D 骨架加强 2D 视频"，multi-task 首选方法）

机制：视频模型除动作 CE 外，加小头回归同相机对齐的原生 3D 骨架（MPJPE）。
  → 模型被迫编码显式人体几何，把动作从外观/身份解耦 → 提升 cross-subject 泛化。
损失 = CE(main) + λ·MPJPE(pose_pred, gt_3d_skel)。测试摘头 → 零开销、模型不变。

用法:
    python scripts/train_auxpose.py --weights ig65m_r2plus1d34.pth --lr 1e-4 \
        --epochs 60 --fold 0 --crop_cache bbox_train.json \
        --lambda_pose 0.1 --save_dir outputs/auxpose
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalVideoDataset, build_balanced_sampler,
                         build_thermal_index, build_train_index)
from src.model import build_model
from src.pose_aux import PoseAlignedVideoDataset, ThermalPoseAlignedVideoDataset
from src.skeleton_dataset import SkeletonClipIndex
from src.split import split_by_subject


class PoseHead(nn.Module):
    """[B,C,T]（已插值到 16 帧）→ [B,T,17*3]（MLP 回归每帧 3D 骨架）。"""

    def __init__(self, feat: int = 256, out_dim: int = 17 * 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(feat, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, out_dim, 1),
        )

    def forward(self, z):  # z: [B,C,T]
        return self.mlp(z).permute(0, 2, 1)  # [B,T,17*3]


@torch.no_grad()
def evaluate_main(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def encoder_features(model, x, pose_layer: int = 3):
    """复刻 R2Plus1D encoder 前向。
    返回 (h_pose, feat)：h_pose = 指定层（默认 layer3，空间细节好、能定位关节）
    [B,C,T',H',W']；feat = layer4 池化 [B,512]（分类用）。
    """
    enc = model.encoder
    h = enc.stem(x)
    h = enc.layer1(h)
    h = enc.layer2(h)
    h3 = enc.layer3(h)
    h4 = enc.layer4(h3)                  # [B,512,T',H',W']
    feat = torch.flatten(enc.avgpool(h4), 1)  # [B,512]
    return (h3 if pose_layer == 3 else h4), feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--backbone", default="r2plus1d34")
    ap.add_argument("--weights", default="", help="IG-65M 预训练权重（R2+1D-34 用）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop_cache", default="bbox_train.json",
                    help="main 用 bbox_train.json；thermal 用 bbox_thermal_train.json")
    ap.add_argument("--modality", type=str, default="main", choices=["main", "thermal"],
                    help="监督的视频模态：main=Depth+IR(4ch，同相机逐帧对齐)；thermal=热像(3ch，跨相机比例对齐)")
    ap.add_argument("--lambda_pose", type=float, default=0.05, help="MPJPE 辅助损失权重 λ（小=轻正则）")
    ap.add_argument("--pose_layer", type=int, default=3, choices=[3, 4],
                    help="位姿头特征层：3=layer3（空间细节，推荐）；4=layer4（旧）")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--save_dir", default="outputs/auxpose")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    if args.modality == "main":
        video_clips = build_train_index(root)
        dir_attr = "depth_dir"
        in_channels = 4
    else:
        video_clips = build_thermal_index(root)
        dir_attr = "thermal_dir"
        in_channels = 3
    skel_clips = []
    for c in video_clips:
        action = getattr(c, dir_attr).parent.parent.name
        skel_dir = root / "Skeleton" / action / c.subject / c.sample / "predictions"
        skel_clips.append(SkeletonClipIndex(c.action_id, c.subject, c.sample, skel_dir))
    crop = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))
    print(f"device={device} modality={args.modality} clips={len(video_clips)} "
          f"λ_pose={args.lambda_pose} frames={args.num_frames} "
          f"（稠密 16 帧位姿监督 + 缺失掩码）", flush=True)

    folds = split_by_subject(video_clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_main = [video_clips[i] for i in tr_idx]
        tr_skel = [skel_clips[i] for i in tr_idx]
        va_main = [video_clips[i] for i in va_idx]

        if args.modality == "main":
            tr_ds = PoseAlignedVideoDataset(tr_main, tr_skel, args.num_frames, args.size, True,
                                            crop, aug_strength=args.aug_strength, seed=42)
            va_ds = DepthIRVideoDataset(va_main, args.num_frames, args.size, False, crop,
                                        use_frame_diff=False)
        else:
            tr_ds = ThermalPoseAlignedVideoDataset(tr_main, tr_skel, args.num_frames, args.size,
                                                   True, crop, aug_strength=args.aug_strength,
                                                   seed=42)
            va_ds = ThermalVideoDataset(va_main, args.num_frames, args.size, False, crop,
                                        use_frame_diff=False)
        sampler = build_balanced_sampler([c.action_id for c in tr_main])
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

        model = build_model(args.backbone, num_classes=40, in_channels=in_channels,
                            n_segment=args.num_frames,
                            weights_path=args.weights or None).to(device)
        pose_dim = 256 if args.pose_layer == 3 else 512
        pose_head = PoseHead(feat=pose_dim, out_dim=17 * 3).to(device)
        params = list(model.parameters()) + list(pose_head.parameters())
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best = 0.0
        save_path = save_dir / f"{args.backbone}_auxpose_fold{fi}.pth"
        for ep in range(args.epochs):
            model.train()
            pose_head.train()
            run_loss = run_ce = run_mpjpe = 0.0
            n = 0
            for x, skel, mask_t, y, _ in tr_loader:
                x, skel, mask_t, y = x.to(device), skel.to(device), mask_t.to(device), y.to(device)
                opt.zero_grad()
                h, feat = encoder_features(model, x.permute(0, 2, 1, 3, 4), args.pose_layer)
                ce = crit(model.head(feat), y)
                # 稠密 16 帧监督：空间 GAP → 时间插值回 T=16（消除 off-by-one）→ 逐帧回归
                z = h.mean(dim=(3, 4))                                        # [B,C,T']
                z = F.interpolate(z, size=x.shape[1], mode="linear",
                                  align_corners=False)                        # [B,C,16]
                pred = pose_head(z).view(x.shape[0], x.shape[1], 17, 3)       # [B,16,17,3]
                m = mask_t.unsqueeze(-1)                                       # [B,16,1]
                err = (pred - skel).norm(dim=-1) * m                           # [B,16,17]
                mpjpe = err.sum() / m.sum().clamp(min=1)                      # 掩码 MPJPE
                loss = ce + args.lambda_pose * mpjpe
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(y)
                run_ce += ce.item() * len(y)
                run_mpjpe += mpjpe.item() * len(y)
                n += len(y)
            sched.step()
            acc = evaluate_main(model, va_loader, device)
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "best_acc": best,
                            "lambda_pose": args.lambda_pose}, save_path)
            print(f"[fold{fi}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"ce={run_ce/max(n,1):.4f} mpjpe={run_mpjpe/max(n,1):.4f} "
                  f"main_val={acc:.4f} best={best:.4f}", flush=True)
        print(f"== fold {fi} best main val = {best:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
