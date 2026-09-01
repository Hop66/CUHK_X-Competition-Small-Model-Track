#!/usr/bin/env python3
"""CUHK-X —— main+thermal 双流 MISA 式多任务正则（对齐 declare-lab/MISA ACMMM20）

对照：dual_fusion（concat+MLP）fold0、0.73 logits 平均
MISA 核心（小数据多模态正则）：
  - 每模态特征 → 投影 → 拆 shared（跨模态不变）+ private（模态特定）
  - diff_loss: private 与 private/shared 正交（DiffLoss：L2 归一化后 Gram 矩阵平方）
  - sim_loss:   shared 跨模态对齐（CMD：前 5 阶中心矩距离）
  - recon_loss: shared+private 重构回原特征（自监督，稳定）
  - cls:        fusion(shared_m,shared_t,private_m,private_t) → 40 类
  - total = cls + 0.3*diff + 1.0*sim + 1.0*recon（MISA 默认权重）
特化：双 IG65M backbone 微调 lr 1e-4 + 融合头 1e-3 + StepLR + 模态 dropout + 早停

判读：fold0 > 0.68（dual_fusion 判据）且 > dual_fusion 本体 → MISA 正则有效
用法:
    python scripts/train_misa_dual.py --fold 0 --weights ig65m_r2plus1d34.pth
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
from torch.utils.data import DataLoader

from src.dataset import (ClipIndex, DepthIRVideoDataset, ThermalClipIndex,
                         ThermalVideoDataset, build_train_index)
from src.model import R2Plus1D34
from src.split import split_by_subject


class DualDataset:
    """同一 clip 的 main(4ch) + thermal(3ch) 配对。"""

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


# ---------------- MISA 损失组件（从 declare-lab/MISA src/utils/functions.py 移植） ----------------
class DiffLoss(nn.Module):
    """私有/共享特征正交：零均值 + L2 归一化后 Gram 矩阵平方。"""

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1 = input1 - torch.mean(input1, dim=0, keepdims=True)
        input2 = input2 - torch.mean(input2, dim=0, keepdims=True)
        i1 = input1.div(torch.norm(input1, p=2, dim=1, keepdim=True).detach().expand_as(input1) + 1e-6)
        i2 = input2.div(torch.norm(input2, p=2, dim=1, keepdim=True).detach().expand_as(input2) + 1e-6)
        return torch.mean((i1.t().mm(i2)).pow(2))


class CMD(nn.Module):
    """Central Moment Discrepancy：前 n_moments 阶中心矩距离（共享特征跨模态对齐）。
    ⚠️ 修复：欧几里得范数 pow(dist, 0.5) 在 dist=0 时梯度 0.5*0^(-0.5)=inf → NaN
      （模态 dropout 后两模态都成常量 → 中心矩相等 dist=0 触发，set_detect_anomaly 定位）。加 eps。
    """

    def forward(self, x1, x2, n_moments=5):
        mx1, mx2 = torch.mean(x1, 0), torch.mean(x2, 0)
        sx1, sx2 = x1 - mx1, x2 - mx2
        out = torch.pow(torch.pow(mx1 - mx2, 2).sum() + 1e-6, 0.5)
        for k in range(2, n_moments + 1):
            out += self._scm(sx1, sx2, k)
        return out

    def _scm(self, sx1, sx2, k):
        ss1, ss2 = sx1.pow(k).mean(0), sx2.pow(k).mean(0)
        return torch.pow(torch.pow(ss1 - ss2, 2).sum() + 1e-6, 0.5)


class MISAHead(nn.Module):
    """双流特征 → shared/private 分离 + 多任务正则 + 融合分类。"""

    def __init__(self, feat_dim=512, hid=512, mod_drop=0.5):
        super().__init__()
        self.mod_drop = mod_drop
        self.proj_m = nn.Sequential(nn.Linear(feat_dim, hid), nn.LayerNorm(hid))
        self.proj_t = nn.Sequential(nn.Linear(feat_dim, hid), nn.LayerNorm(hid))
        self.shared_m = nn.Sequential(nn.Linear(hid, hid), nn.Sigmoid())
        self.shared_t = nn.Sequential(nn.Linear(hid, hid), nn.Sigmoid())
        self.private_m = nn.Sequential(nn.Linear(hid, hid), nn.Sigmoid())
        self.private_t = nn.Sequential(nn.Linear(hid, hid), nn.Sigmoid())
        self.recon_m = nn.Linear(2 * hid, feat_dim)
        self.recon_t = nn.Linear(2 * hid, feat_dim)
        self.fusion = nn.Sequential(nn.Dropout(0.5), nn.Linear(4 * hid, 40))

    def forward(self, fm, ft, is_train=False):
        # 模态 dropout（鲁棒 + 防过拟合；MISA 原版无，我们保留 dual_fusion 的增强）
        if is_train and self.mod_drop > 0:
            if torch.rand(1).item() < self.mod_drop:
                fm = torch.zeros_like(fm)
            if torch.rand(1).item() < self.mod_drop:
                ft = torch.zeros_like(ft)
        hm, ht = self.proj_m(fm), self.proj_t(ft)
        sm, pm = self.shared_m(hm), self.private_m(hm)
        st, pt = self.shared_t(ht), self.private_t(ht)
        # 重构回原特征（自监督）
        rm = self.recon_m(torch.cat([sm, pm], -1))
        rt = self.recon_t(torch.cat([st, pt], -1))
        logits = self.fusion(torch.cat([sm, st, pm, pt], -1))
        return logits, rm, rt, sm, st, pm, pt


class MISADual(nn.Module):
    def __init__(self, weights_path, mod_drop=0.5):
        super().__init__()
        self.main = R2Plus1D34(40, 4, weights_path)
        self.th = R2Plus1D34(40, 3, weights_path)
        self.head = MISAHead(mod_drop=mod_drop)

    def forward(self, xm, xt, is_train=False):
        fm = self.main.encoder(xm.permute(0, 2, 1, 3, 4))   # [B,512]
        ft = self.th.encoder(xt.permute(0, 2, 1, 3, 4))     # [B,512]
        return self.head(fm, ft, is_train)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xm, xt, y in loader:
        xm, xt, y = xm.to(device), xt.to(device), y.to(device)
        logits, *_ = model(xm, xt)
        correct += (logits.argmax(-1) == y).sum().item()
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
    ap.add_argument("--mod_drop", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--save_dir", type=str, default="outputs/misa_dual")
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

    model = MISADual(args.weights, mod_drop=args.mod_drop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MISADual params ~{n_params/1e6:.0f}M "
          f"(fp32 ~{n_params*4/1e6:.0f}MB → int5 ~{n_params*4/1e6/8:.0f}MB)", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.main.parameters(), "lr": args.lr},
        {"params": model.th.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * 10},   # 新学 head 要快
    ], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    recon_crit = nn.MSELoss()
    diff_crit = DiffLoss()
    cmd_crit = CMD()

    # MISA 默认权重
    diff_w, sim_w, recon_w = 0.3, 1.0, 1.0

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"misa_fold{args.fold}.pth"
    best, no_improve = 0.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, run_cls, run_dif, run_sim, run_rec, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for xm, xt, y in tr_loader:
            xm, xt, y = xm.to(device), xt.to(device), y.to(device)
            opt.zero_grad()
            logits, rm, rt, sm, st, pm, pt = model(xm, xt, is_train=True)
            fm = model.main.encoder(xm.permute(0, 2, 1, 3, 4)).detach()
            ft = model.th.encoder(xt.permute(0, 2, 1, 3, 4)).detach()
            l_cls = cls_crit(logits, y)
            l_dif = diff_crit(pm, sm) + diff_crit(pt, st) + diff_crit(pm, pt)
            l_sim = cmd_crit(sm, st, 5)
            # recon 在 LayerNorm 特征上做：encoder 特征是大值 O(10~100)，直接 MSE 会爆炸（本地复现
            #   rec~2500 + CMD 数值膨胀 → step3 NaN）。LayerNorm 到 O(1) 稳定（MISA 原版特征是 z-norm 过的）。
            rm_n = F.layer_norm(rm, rm.shape[1:])
            fm_n = F.layer_norm(fm, fm.shape[1:])
            rt_n = F.layer_norm(rt, rt.shape[1:])
            ft_n = F.layer_norm(ft, ft.shape[1:])
            l_rec = (recon_crit(rm_n, fm_n) + recon_crit(rt_n, ft_n)) / 2.0
            loss = l_cls + diff_w * l_dif + sim_w * l_sim + recon_w * l_rec
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            run_cls += l_cls.item() * y.numel()
            run_dif += l_dif.item() * y.numel()
            run_sim += l_sim.item() * y.numel()
            run_rec += l_rec.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate(model, va_loader, device)
        if acc > best:
            best, no_improve = acc, 0
            torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
        else:
            no_improve += 1
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} "
              f"cls={run_cls/max(n,1):.4f} diff={run_dif/max(n,1):.4f} "
              f"sim={run_sim/max(n,1):.4f} recon={run_rec/max(n,1):.4f} "
              f"fusion_val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if no_improve >= args.patience:
            print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
            break

    print(f"== MISA 双流 fold{args.fold} best val = {best:.4f} "
          f"（对照 dual_fusion / main 0.6609 / thermal 0.6339 / 融合 0.73；"
          f">0.68 且 > dual_fusion → MISA 正则有效）→ {save_path}", flush=True)


if __name__ == "__main__":
    main()
