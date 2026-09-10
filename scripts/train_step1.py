#!/usr/bin/env python3
"""
CUHK-X Step 1 —— 运动优先主线训练（Depth_Color+IR 4ch → 视频模型）

流程：
  1. 建训练索引（2891 clip / 18 被试 / 40 类）
  2. subject-fold 交叉验证（3 折，每折 6 被试验证）
  3. 类平衡采样 + Adam + cosine
  4. 每折训练/验证，输出 val 准确率

用法:
    python scripts/train_step1.py --backbone r2plus1d --num_frames 16 --epochs 30
    python scripts/train_step1.py --backbone tsm_resnet18 --num_frames 16
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import (ClipIndex, DepthIRVideoDataset, build_train_index,
                         ThermalVideoDataset, build_thermal_index,
                         build_balanced_sampler)
from src.split import split_by_subject
from src.model import build_model


# 难对集（twin_v3 数据驱动审计 + 骨架 v3 特征 62-65% 判别力验证）:
# 骨架作为"难对选择器"（LUPI 修正函数）：训练时难对类样本 CE 梯度加权，
# 逼 main 在易混淆类上更专注；纯 loss 侧，不动输入/架构/监督 → 规避历史雷区。
HARD_PAIRS = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
HPID = {a for p in HARD_PAIRS for a in p}


def _hp_weights(y, w):
    import numpy as _np
    _hp = _np.isin(y.cpu().numpy(), list(HPID))
    m = torch.ones_like(y.float())
    m[_hp] = float(w)
    return m


def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    correct = total = 0
    run_loss = 0.0
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            run_loss += crit(out, y).item() * y.numel()
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1), run_loss / max(total, 1)


def _set_freeze(model, trainable_substr):
    """All params frozen except those whose name contains any of trainable_substr."""
    for n, p in model.named_parameters():
        p.requires_grad_(any(k in n for k in trainable_substr))


def _random_erasing_cube(x, prob=0.5, min_area=0.02, max_area=1.0 / 3.0):
    """pytorchvideo 风格随机时空 cube 擦除(同一 patch 覆盖全部 T 帧), mode≈rand.
    输入 [B,C,T,H,W]; 就地修改. prob=0 与操作修改无关(不会调用)."""
    B, C, T, H, W = x.shape
    for i in range(B):
        if np.random.random() > prob:
            continue
        for _ in range(10):
            target_area = np.random.uniform(min_area, max_area) * H * W
            aspect = np.random.uniform(0.3, 1.0 / 0.3)
            eh = max(int(round(np.sqrt(target_area * aspect))), 1)
            ew = max(int(round(np.sqrt(target_area / aspect))), 1)
            if ew < W and eh < H:
                top = np.random.randint(0, H - eh)
                left = np.random.randint(0, W - ew)
                fill = torch.empty(C, T, eh, ew, dtype=x.dtype, device=x.device).normal_()
                x[i, :, :, top:top + eh, left:left + ew] = fill
                break
    return x


def _cutmix3d(x, y_orig, perm, alpha):
    """Spatial CutMix on [B,C,T,H,W] (mmaction2 标配, alpha~1).
    返回: x_permuted_mixed, lam_eff (标签混合权重)."""
    _, C, T, H, W = x.shape
    lam = float(np.random.beta(alpha, alpha))
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = max(int(cx - cut_w / 2), 0)
    bby1 = max(int(cy - cut_h / 2), 0)
    bbx2 = min(int(cx + cut_w / 2), W)
    bby2 = min(int(cy + cut_h / 2), H)
    xc = x.clone()
    xc[..., bbx1:bbx2, bby1:bby2] = x[perm][..., bbx1:bbx2, bby1:bby2]
    lam_eff = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1)) / float(W * H)
    return xc, lam_eff


def _make_opt(optim_type, model, lr, wd=1e-4):
    if optim_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def train_fold(model, train_loader, val_loader, device, epochs, lr, fold, save_path=None,
               label_smoothing=0.0, full=False, save_every=10,
               cb_loss_params=None, freeze_epochs=0, freeze_scale=0.3,
               freeze_train=("stem", "layer3", "layer4"),
               mixup_alpha=0.0, cutmix_alpha=0.0, erase_prob=0.0, ema_decay=0.0,
               grad_clip=0.0, rsc_drop=0.0, hardpair_w=0.0, optim_type="adam"):
    """cb_loss_params: dict with 'class_counts' and 'beta', or None for standard CE."""
    opt = _make_opt(optim_type, model, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if freeze_epochs > 0:
        _set_freeze(model, freeze_train)
        print(f"[fold{fold}] 分阶段: ep<= {freeze_epochs} 只训 {freeze_train}, "
              f"之后全解冻 lr x{freeze_scale}", flush=True)
        print(f"[fold{fold}] trainable={sum(1 for p in model.parameters() if p.requires_grad)}/"
              f"{len(list(model.parameters()))}", flush=True)
    if cb_loss_params is not None:
        from src.cb_loss import ClassBalancedLoss
        crit = ClassBalancedLoss(
            class_counts=cb_loss_params["class_counts"],
            num_classes=40,
            beta=cb_loss_params.get("beta", 0.9999),
            label_smoothing=label_smoothing
        )
        print(f"[fold{fold}] using ClassBalancedLoss (beta={cb_loss_params['beta']})", flush=True)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ema_state = None
    if ema_decay > 0:
        ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        print(f"[fold{fold}] EMA decay={ema_decay}, MixUp alpha={mixup_alpha}", flush=True)
    best = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            if erase_prob > 0:
                x = _random_erasing_cube(x, erase_prob)
            if rsc_drop > 0:
                # RSC (Self-Challenging, ECCV'20): 丢弃每样本主导像素 → 逼学非主导(跨域稳健)
                x1 = x.clone().requires_grad_(True)
                opt.zero_grad()
                l1 = crit(model(x1), y)
                l1.backward()
                sal = x1.grad.abs().mean(dim=(1, 2))          # [B,H,W] (平均 C,T)
                Bsz, H, W = sal.shape
                k = max(int(round(rsc_drop * H * W)), 1)
                thr = sal.view(Bsz, -1).kthvalue(H * W - k + 1, dim=1).values
                mask = (sal >= thr[:, None, None]).float()    # 1=主导像素(丢弃)
                x2 = (x1.detach() * (1 - mask[:, None, None])).requires_grad_(True)
                opt.zero_grad()
                loss = crit(model(x2), y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            else:
                opt.zero_grad()
                if mixup_alpha > 0 or cutmix_alpha > 0:
                    # batch blending (maction2 标配): mixup / cutmix 二选一或独立
                    blend = None
                    if mixup_alpha > 0 and cutmix_alpha > 0:
                        blend = np.random.choice(["mixup", "cutmix"])
                    elif mixup_alpha > 0:
                        blend = "mixup"
                    else:
                        blend = "cutmix"
                    perm = torch.randperm(x.size(0), device=x.device)
                    if blend == "mixup":
                        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                        xb = lam * x + (1 - lam) * x[perm]
                        out_m = model(xb)
                        loss = lam * crit(out_m, y) + (1 - lam) * crit(out_m, y[perm])
                    else:
                        xb, lam = _cutmix3d(x, y, perm, cutmix_alpha)
                        out_m = model(xb)
                        loss = lam * crit(out_m, y) + (1 - lam) * crit(out_m, y[perm])
                else:
                    out = model(x)
                    if hardpair_w > 0 and int((np.isin(y.cpu(), list(HPID))).sum()) > 0:
                        # 难对区样本 CE 梯度加权（per-sample；非难对保持权重1）
                        l = F.cross_entropy(out, y, reduction='none', label_smoothing=label_smoothing)
                        w = _hp_weights(y, hardpair_w)
                        loss = (l * w).mean()
                    else:
                        loss = crit(out, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            if ema_state is not None:
                with torch.no_grad():
                    md = model.state_dict()
                    for k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(md[k], alpha=1.0 - ema_decay)
            run_loss += loss.item() * y.numel()
            n += y.numel()
        if freeze_epochs > 0 and ep + 1 == freeze_epochs:
            # 解冻全层, lr 缩放, cosine 周期改为剩余 epoch
            for p in model.parameters():
                p.requires_grad_(True)
            opt = _make_opt(optim_type, model, lr * freeze_scale)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - freeze_epochs)
            print(f"[fold{fold}] ep{ep+1}: unfreeze ALL, lr->{lr*freeze_scale:.2e}", flush=True)
        sched.step()
        if full:
            # 全量模式：无 val，每 save_every ep 保存快照 + 最后 ep 保存最终（CosineAnnealing 收敛）
            if save_path is not None and ((ep + 1) % save_every == 0 or ep == epochs - 1):
                if ema_state is not None:
                    online = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema_state)
                    torch.save(model.state_dict(), save_path)
                    model.load_state_dict(online)   # 恢复在线权重, 下一 ep 训练不受污染
                else:
                    torch.save(model.state_dict(), save_path)
            print(f"[{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
                  f"(FULL) saved={save_path} ({time.time()-t0:.1f}s)", flush=True)
            continue
        online = None
        if ema_state is not None:
            # EMA 评估: 保存在线权重, 临时切到 EMA, 评估后恢复在线(避免训练被 EMA 权重污染)
            online = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
        acc, val_loss = evaluate(model, val_loader, device, label_smoothing)
        if online is not None:
            model.load_state_dict(online)
        if acc > best:
            best = acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val_loss={val_loss:.4f} val={acc:.4f} best={best:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--backbone", type=str, default="r2plus1d",
                    choices=["r2plus1d", "r2plus1d34", "tsm_resnet18"])
    ap.add_argument("--weights", type=str, default="", help="IG-65M 预训练权重路径（r2plus1d34 用）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop_cache", type=str, default="", help="bbox_cache.json 路径，缺省不裁剪")
    ap.add_argument("--ir_mask", action="store_true", help="IR Otsu 人像掩码抑制 Depth 背景")
    ap.add_argument("--skel_heat", action="store_true", help="M1: 骨架 heatmap 通道并入 main(4ch→5ch)")
    ap.add_argument("--bright_alpha", type=float, default=1.0, help="M7: 训练输入亮度 alpha(匹配 test 0.462)")
    ap.add_argument("--bright_beta", type=float, default=0.0, help="M7: 训练输入亮度 beta")
    ap.add_argument("--frame_diff", action="store_true", help="追加帧差运动通道（显式抓运动，4→8ch）")
    ap.add_argument("--no_balanced", action="store_true", help="关闭类平衡采样")
    ap.add_argument("--modality", type=str, default="depthir", choices=["depthir", "thermal"],
                    help="depthir=Depth+IR 4ch；thermal=Thermal 3ch")
    ap.add_argument("--save_dir", type=str, default="", help="保存 best checkpoint 的目录（缺省不保存）")
    ap.add_argument("--fold", type=int, default=-1, help="只跑指定折（-1=全折）")
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="CrossEntropy 标签平滑（缓解过拟合/校准，0.1 推荐）")
    ap.add_argument("--cb_loss", action="store_true",
                    help="启用 Class-Balanced Loss（有效样本数加权，针对长尾分布；与 label_smoothing 兼容）")
    ap.add_argument("--cb_beta", type=float, default=0.9999,
                    help="CB Loss 的 beta 参数（0~1），越大越接近逆频率加权；典型值 0.9/0.99/0.999/0.9999")
    ap.add_argument("--full", action="store_true",
                    help="全量训练：全部数据不分折、无 val，保存最后 epoch（全量冲刺用）")
    ap.add_argument("--full_save_every", type=int, default=10, help="全量模式快照保存间隔")
    ap.add_argument("--freeze_epochs", type=int, default=0,
                    help=">0: 前N ep 只训 stem/layer3/layer4, 之后全解冻 lr x --freeze_scale (分阶段训练)")
    ap.add_argument("--freeze_scale", type=float, default=0.3, help="解冻后 lr 缩放 (默认0.3)")
    ap.add_argument("--mixup_alpha", type=float, default=0.0,
                    help=">0: 时空 MixUp (Beta(α,α) 插值输入+标签混合 CE)")
    ap.add_argument("--cutmix_alpha", type=float, default=0.0,
                    help=">0: 空间 CutMix (mmaction2 标配 α~1, 保局部特征提升鲁棒)")
    ap.add_argument("--erase_prob", type=float, default=0.0,
                    help=">0: RandomErasing 时空 cube(pytorchvideo 标配 prob0.5); 擦除随机区域加强鲁棒")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help=">0: EMA 权重指数平均(0.99x常用); 训练稳定+val 用 EMA 评估")
    ap.add_argument("--grad_clip", type=float, default=0.0,
                    help=">0: grad 剪枝(权威库标配 40), 稳定长训练")
    ap.add_argument("--rsc_drop", type=float, default=0.0,
                    help=">0: RSC(Self-Challenging ECCV20) 丢主导像素比例如0.3, 提升跨域泛化")
    ap.add_argument("--hardpair_w", type=float, default=0.0,
                    help=">0: 难对区样本 CE 梯度加权 ×w（骨架难对选择器, LUPI 修正函数精神；"
                         "HARD 5 对=13/12,22/21,8/10,18/17,26/24。纯 loss 侧不动架构）")
    ap.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"],
                    help="优化器: adam(现状) / sgd(maction2 标配, momentum0.9; SGD 时用较大 lr 如 1e-2)")
    ap.add_argument("--seed", type=int, default=42, help="全量模式随机种子（多 seed 集成用）")
    ap.add_argument("--aug_strength", type=int, default=2,
                    help="训练增强强度档位 0/1/2/3（0=无,1=温和,2=当前强,3=更强）")
    ap.add_argument("--sample_mode", type=str, default="uniform", choices=["uniform", "segment"],
                    help="时间采样：uniform=endpoint 全程均匀（现状）；segment=全球覆盖分段"
                         "（均分 num_frames 段每段 1 帧，train 段内随机/eval 段中，14th-place 移植）")
    ap.add_argument("--gray_norm", action="store_true",
                    help="仅 thermal：用灰度拉伸归一化 (x-0.5)/0.25 替代 Kinetics norm"
                         "（14th-place baseline 用；thermal 单通道灰度更匹配）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} backbone={args.backbone} frames={args.num_frames}")

    root = Path(args.train_root).expanduser()
    if args.modality == "thermal":
        clips = build_thermal_index(root)
        in_channels = 6 if args.frame_diff else 3
    else:
        clips = build_train_index(root)
        in_channels = (8 if args.frame_diff else 4) + (3 if args.skel_heat else 0)
    print(f"modality={args.modality} clips={len(clips)} "
          f"subjects={len(set(c.subject for c in clips))} "
          f"classes={len(set(c.action_id for c in clips))}")

    crop_cache = {}
    if args.crop_cache:
        crop_cache = json.loads(Path(args.crop_cache).read_text(encoding="utf-8"))
        print(f"crop cache loaded: {len(crop_cache)} entries")

    skel_map = {}
    if args.skel_heat:
        from src.skeleton_dataset import build_skeleton_index as _bsk, SkeletonClipIndex
        _sk = (_bsk(root) if root else [])
        skel_map = {f"{c.action_id}/{c.subject}/{c.sample}": str(c.pred_dir) for c in _sk}
        print(f"skel_map: {len(skel_map)} clips", flush=True)

    folds = split_by_subject(clips, n_folds=args.folds)
    target_folds = range(len(folds)) if args.fold < 0 else [args.fold]

    # ================= FULL 全量训练（不分折，多 seed） =================
    if args.full:
        tr_clips = clips
        if args.modality == "thermal":
            ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
            train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_frame_diff=args.frame_diff, seed=args.seed,
                                           aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode, mean_std=ms)
        else:
            train_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_ir_mask=args.ir_mask, use_frame_diff=args.frame_diff,
                                           seed=args.seed, aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode,
                                           skel_map=skel_map or None,
                                           brightness_alpha=args.bright_alpha, brightness_beta=args.bright_beta)
        sampler = None
        if not args.no_balanced:
            sampler = build_balanced_sampler([c.action_id for c in tr_clips])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        model = build_model(args.backbone, num_classes=40, in_channels=in_channels,
                            n_segment=args.num_frames,
                            weights_path=args.weights or None).to(device)
        save_path = None
        if args.save_dir:
            d = Path(args.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            save_path = d / f"{args.backbone}_{args.modality}_full_seed{args.seed}.pth"

        # CB Loss 参数准备
        cb_params = None
        if args.cb_loss:
            from collections import Counter
            class_counts = [0] * 40
            for c in tr_clips:
                class_counts[c.action_id] += 1
            cb_params = {"class_counts": class_counts, "beta": args.cb_beta}
            print(f"[full] ClassBalancedLoss params: counts={class_counts}, beta={args.cb_beta}", flush=True)

        print(f"==== FULL 全量训练（{len(tr_clips)} clips, seed={args.seed}, "
              f"epochs={args.epochs}, lr={args.lr}, label_smoothing={args.label_smoothing}）====", flush=True)
        train_fold(model, train_loader, None, device, args.epochs, args.lr, "full",
                   save_path, label_smoothing=args.label_smoothing,
                   full=True, save_every=args.full_save_every,
                   cb_loss_params=cb_params,
                   freeze_epochs=args.freeze_epochs, freeze_scale=args.freeze_scale,
                   mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
                   erase_prob=args.erase_prob,
                   ema_decay=args.ema_decay, grad_clip=args.grad_clip,
                   rsc_drop=args.rsc_drop, hardpair_w=args.hardpair_w,
                   optim_type=args.optim)
        print(f"==== FULL DONE: {save_path} ====", flush=True)
        return

    fold_accs = []
    for fi in target_folds:
        train_idx, val_idx = folds[fi]
        tr_clips = [clips[i] for i in train_idx]
        va_clips = [clips[i] for i in val_idx]

        if args.modality == "thermal":
            ms = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) if args.gray_norm else None
            train_ds = ThermalVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_frame_diff=args.frame_diff,
                                           aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode, mean_std=ms)
            val_ds = ThermalVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                         use_frame_diff=args.frame_diff,
                                         sample_mode=args.sample_mode, mean_std=ms)
        else:
            train_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                           use_ir_mask=args.ir_mask, use_frame_diff=args.frame_diff,
                                           aug_strength=args.aug_strength,
                                           sample_mode=args.sample_mode,
                                           skel_map=skel_map or None,
                                           brightness_alpha=args.bright_alpha, brightness_beta=args.bright_beta)
            val_ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                         use_ir_mask=args.ir_mask, use_frame_diff=args.frame_diff,
                                         sample_mode=args.sample_mode,
                                         skel_map=skel_map or None,
                                         brightness_alpha=args.bright_alpha, brightness_beta=args.bright_beta)

        sampler = None
        if not args.no_balanced:
            sampler = build_balanced_sampler([c.action_id for c in tr_clips])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

        model = build_model(args.backbone, num_classes=40, in_channels=in_channels,
                            n_segment=args.num_frames,
                            weights_path=args.weights or None).to(device)
        save_path = None
        if args.save_dir:
            d = Path(args.save_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            save_path = d / f"{args.backbone}_{args.modality}_fold{fi}.pth"

        # CB Loss 参数准备（每折独立计算，因为训练集不同）
        cb_params = None
        if args.cb_loss:
            from collections import Counter
            class_counts = [0] * 40
            for c in tr_clips:
                class_counts[c.action_id] += 1
            cb_params = {"class_counts": class_counts, "beta": args.cb_beta}
            print(f"[fold{fi}] ClassBalancedLoss params: counts={class_counts}, beta={args.cb_beta}", flush=True)

        best = train_fold(model, train_loader, val_loader, device, args.epochs, args.lr, fi,
                          save_path, label_smoothing=args.label_smoothing,
                          cb_loss_params=cb_params,
                          freeze_epochs=args.freeze_epochs, freeze_scale=args.freeze_scale,
                          mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
                          erase_prob=args.erase_prob,
                          ema_decay=args.ema_decay, grad_clip=args.grad_clip,
                          rsc_drop=args.rsc_drop, hardpair_w=args.hardpair_w,
                          optim_type=args.optim)
        fold_accs.append(best)
        print(f"== fold {fi} best val = {best:.4f} -> {save_path}")

    if len(fold_accs) > 1:
        print(f"\n==== mean val across {len(fold_accs)} folds: {np.mean(fold_accs):.4f} "
              f"(std {np.std(fold_accs):.4f}) ====")


if __name__ == "__main__":
    main()
