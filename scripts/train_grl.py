#!/usr/bin/env python3
"""CUHK-X —— 梯度反转层（GRL）主体不变性训练

核心思想：在特征提取器后加一个"主体分类头"，但通过梯度反转层（Gradient Reversal Layer）
让主干网络学习**对主体身份不敏感**的特征表示，从而提升跨人泛化能力。

架构：
    输入 → [Backbone R2+1D34] → features → CE 动作分类头（正常梯度）
                                    ↘ GRL → 主体分类头（反转梯度，λ 控制强度）

损失：L = L_action + λ * L_subject（但 GRL 使主干收到 -λ∇L_subject）

用法（fold0 快速验证）：
    python scripts/train_grl.py --fold 0 --lambda_grl 0.5 --save_dir outputs/grl_test

参考：NTU RGB+D cross-subject 标准做法；GAN 领域经典 GRL 机制。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DepthIRVideoDataset, ThermalVideoDataset, build_balanced_sampler
from src.model import build_model
from src.split import split_by_subject


class SubjectIndexDataset(torch.utils.data.Dataset):
    """P0修复(G): GRL 需要 (x, y, subject_idx) 三元组。
    包装底层 Dataset, __getitem__ 从类上解析得 subject_idx —— 替代无效的
    `train_ds.__getitem__ = new_getitem` monkey-patch (special method 从类查,
    实例赋值不生效)。
    """
    def __init__(self, base_ds, subject_index):
        self.base_ds = base_ds
        self.subject_index = subject_index  # clip i -> subject int index

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        x, y, _ = self.base_ds[i]
        return x, y, self.subject_index[i]


# ------------------------------------------------------------------ GRL 模块
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转：前向恒等，反向乘 -lambda。"""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grl(x, lambd=1.0):
    """应用梯度反转层。"""
    return GradientReversalLayer.apply(x, lambd)


# ------------------------------------------------------------------ 模型封装
class ActionNetWithSubjectHead(nn.Module):
    """动作分类 + 主体分类（带 GRL）双头模型。

    测试时只保留动作头，主体头可丢弃。
    """
    def __init__(self, backbone_name, num_classes=40, num_subjects=18, in_channels=4,
                 n_segment=16, weights_path=None, feature_dim=512):
        super().__init__()
        # 骨干 + 动作头（复用现有 build_model，取其 backbone 和 action head）
        self.backbone = build_model(backbone_name, num_classes=num_classes,
                                    in_channels=in_channels, n_segment=n_segment,
                                    weights_path=weights_path or None)
        # 提取特征维度（R2+1D34 的 fc 前是 512）
        self.feature_dim = feature_dim
        # 主体分类头（接在 backbone 的 avgpool 之后、fc 之前）
        # 需要 hook 到 backbone 的全局池化输出
        self.subject_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_subjects)
        )
        self.num_subjects = num_subjects
        self._feature_cache = None
        # Hook 到 backbone 的 avgpool 输出
        self._register_hook()

    def _register_hook(self):
        """注册 hook 捕获 backbone 的全局池化特征。"""
        # R2+1D34 的结构：... → GlobalAvgPool → Flatten → FC(num_classes)
        # 我们需要在 Flatten 之后、FC 之前捕获特征
        # 由于 build_model 返回的是完整模型，我们通过 module name 查找
        for name, module in self.backbone.named_modules():
            if "avgpool" in name.lower() and isinstance(module, (nn.AdaptiveAvgPool3d, nn.AdaptiveAvgPool2d)):
                module.register_forward_hook(self._hook_fn)
                break

    def _hook_fn(self, module, input, output):
        """Hook：保存全局池化后的特征（展平前）。"""
        # output shape: [B, C, T, H, W] 或 [B, C, 1, 1, 1]
        self._feature_cache = output.flatten(start_dim=1)  # [B, C*T*H*W] → 但通常 C 就是特征维

    def forward(self, x, return_subject=False):
        """前向：动作 logits + （可选）主体 logits。

        Args:
            x: [B, T, C, H, W]  (与 train_step1 / 数据集一致)
            return_subject: 是否计算主体 logits（训练时需要，推理时不需要）

        Returns:
            action_logits: [B, 40]
            subject_logits: [B, num_subjects] 或 None
        """
        # 注意: build_model("r2plus1d")/R2Plus1D18 期望 [B,T,C,H,W](数据集格式), 无需 permute
        self._feature_cache = None
        action_logits = self.backbone(x)

        if not return_subject or self._feature_cache is None:
            return action_logits, None

        # 通过 GRL 连接主体头
        features = grl(self._feature_cache, lambd=getattr(self, "_grl_lambda", 1.0))
        subject_logits = self.subject_head(features)
        return action_logits, subject_logits

    def set_grl_lambda(self, lambd):
        """设置 GRL 强度（可在训练中 annealing）。"""
        self._grl_lambda = lambd


# ------------------------------------------------------------------ 训练循环
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        action_logits, _ = model(x.to(device))
        correct += (action_logits.argmax(-1) == y.to(device)).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_grl(model, train_loader, val_loader, device, epochs, lr, fold, save_path=None,
              lambda_grl=0.5, label_smoothing=0.0):
    """GRL 双任务训练。

    Args:
        lambda_grl: GRL 强度系数（主体损失的梯度放大倍数）
    """
    opt = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": model.subject_head.parameters(), "lr": lr * 10}  # 主体头用更高 lr
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit_action = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    crit_subject = nn.CrossEntropyLoss()

    model.set_grl_lambda(lambda_grl)
    best = 0.0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        run_loss_action, run_loss_subject, n = 0.0, 0.0, 0
        for x, y, subj_idx in train_loader:
            x, y = x.to(device), y.to(device)
            subj_idx = subj_idx.to(device)  # 主体索引（用于主体分类）

            opt.zero_grad()
            action_logits, subject_logits = model(x, return_subject=True)

            loss_action = crit_action(action_logits, y)
            loss_subject = crit_subject(subject_logits, subj_idx)
            # P0修复(G): 外层不再乘 λ —— GRL 层内 backward 已乘 lambd(标准 DANN)。
            # 原版 loss_action + lambda_grl*loss_subject 会让 backbone 收到近似 λ² 的对抗强度。
            loss = loss_action + loss_subject

            loss.backward()
            opt.step()

            run_loss_action += loss_action.item() * y.numel()
            run_loss_subject += loss_subject.item() * y.numel()
            n += y.numel()

        sched.step()
        avg_loss_a = run_loss_action / max(n, 1)
        avg_loss_s = run_loss_subject / max(n, 1)

        if val_loader is not None:
            acc = evaluate(model, val_loader, device)
            if acc > best:
                best = acc
                if save_path:
                    torch.save({
                        "model": model.backbone.state_dict(),  # 只存 backbone（测试用）
                        "best_acc": best,
                        "epoch": ep + 1
                    }, save_path)
            print(f"[fold{fold}] ep{ep+1}/{epochs} "
                  f"loss_a={avg_loss_a:.4f} loss_s={avg_loss_s:.4f} "
                  f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
        else:
            print(f"[full] ep{ep+1}/{epochs} "
                  f"loss_a={avg_loss_a:.4f} loss_s={avg_loss_s:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    return best


# ------------------------------------------------------------------ 主函数
def main():
    ap = argparse.ArgumentParser(description="GRL 主体不变性训练")
    ap.add_argument("--backbone", type=str, default="r2plus1d34")
    ap.add_argument("--weights", type=str, default="ig65m_r2plus1d34.pth")
    ap.add_argument("--modality", type=str, default="depthir", choices=["depthir", "thermal"])
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1, help="-1=全折；否则只跑指定折")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop_cache", type=str, default="", help="bbox cache json")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--sample_mode", type=str, default="segment", choices=["uniform", "segment"])
    ap.add_argument("--gray_norm", action="store_true", help="thermal 用灰度归一化")
    ap.add_argument("--lambda_grl", type=float, default=0.5,
                    help="GRL 强度系数（主体损失梯度放大倍数；0=关闭 GRL，1=标准强度）")
    ap.add_argument("--save_dir", type=str, default="outputs/grl")
    ap.add_argument("--full", action="store_true", help="全量模式：无 val，保存最后 epoch")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, modality={args.modality}, lambda_grl={args.lambda_grl}", flush=True)

    root = Path("~/Multimodal/data/Training/HAR").expanduser()
    crop_cache = {}
    if args.crop_cache:
        import json
        crop_cache = json.loads(Path(args.crop_cache).expanduser().read_text(encoding="utf-8"))

    if args.modality == "thermal":
        in_channels = 3
        idx_cls = lambda r: [type("C", (), {
            "action_id": int(p.parent.parent.name.split("_")[0]),
            "subject": p.parent.name,
            "sample": p.name,
            "thermal_dir": p
        })() for p in sorted((r / "Thermal").rglob("*")) if p.is_dir() and len(p.name) > 2]
        ds_cls = ThermalVideoDataset
    else:
        in_channels = 4
        from src.dataset import build_train_index
        idx_cls = build_train_index
        ds_cls = DepthIRVideoDataset

    clips = idx_cls(root)
    folds = split_by_subject(clips, n_folds=args.folds)

    # 构建主体索引映射（subject string → int）
    all_subjects = sorted(set(c.subject for c in clips))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
    num_subjects = len(all_subjects)
    print(f"subjects={num_subjects}: {all_subjects[:5]}...", flush=True)

    target_folds = range(args.folds) if args.fold < 0 else [args.fold]

    if args.full:
        tr_clips = clips
        ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
        if args.modality == "thermal":
            train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           aug_strength=args.aug_strength, sample_mode=args.sample_mode,
                                           mean_std=ms)
        else:
            train_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           aug_strength=args.aug_strength, sample_mode=args.sample_mode)
        # P0修复(G): 先包装(返回 subject_idx), 再建 loader —— 包装必须在 DataLoader 之前
        subject_index = [subject_to_idx[c.subject] for c in tr_clips]
        train_ds = SubjectIndexDataset(train_ds, subject_index)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

        model = ActionNetWithSubjectHead(args.backbone, num_classes=40, num_subjects=num_subjects,
                                         in_channels=in_channels, n_segment=args.num_frames,
                                         weights_path=args.weights or None).to(device)
        save_path = Path(args.save_dir).expanduser() / f"{args.backbone}_{args.modality}_grl_full.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        train_grl(model, train_loader, None, device, args.epochs, args.lr, "full",
                  save_path, lambda_grl=args.lambda_grl, label_smoothing=args.label_smoothing)
        print(f"==== GRL FULL DONE → {save_path} ====", flush=True)
        return

    fold_accs = []
    for fi in target_folds:
        tr_idx, va_idx = folds[fi]
        tr_clips = [clips[i] for i in tr_idx]
        va_clips = [clips[i] for i in va_idx]

        ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
        if args.modality == "thermal":
            train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           aug_strength=args.aug_strength, sample_mode=args.sample_mode,
                                           mean_std=ms)
            val_ds = ThermalVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                         sample_mode=args.sample_mode, mean_std=ms)
        else:
            train_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           aug_strength=args.aug_strength, sample_mode=args.sample_mode)
            val_ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                         sample_mode=args.sample_mode)

        # P0修复(G): 先包装(返回 subject_idx), 再建 loader
        subject_index = [subject_to_idx[c.subject] for c in tr_clips]
        train_ds = SubjectIndexDataset(train_ds, subject_index)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

        model = ActionNetWithSubjectHead(args.backbone, num_classes=40, num_subjects=num_subjects,
                                         in_channels=in_channels, n_segment=args.num_frames,
                                         weights_path=args.weights or None).to(device)

        save_path = Path(args.save_dir).expanduser() / f"{args.backbone}_{args.modality}_grl_fold{fi}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best = train_grl(model, train_loader, val_loader, device, args.epochs, args.lr, fi,
                         save_path, lambda_grl=args.lambda_grl, label_smoothing=args.label_smoothing)
        fold_accs.append(best)
        print(f"== fold {fi} GRL best val = {best:.4f} → {save_path}")

    if len(fold_accs) > 1:
        print(f"\n==== GRL mean val across {len(fold_accs)} folds: {np.mean(fold_accs):.4f} "
              f"(std {np.std(fold_accs):.4f}) ====")


if __name__ == "__main__":
    main()
