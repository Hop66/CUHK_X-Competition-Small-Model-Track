#!/usr/bin/env python3
"""P0 conditional residual fusion (2026-09-12, 审计 P0 最看好方向).

动机(C 统计实证): main×thermal 单模 3折, 热模态可修 main 错区 26% (C=242), 
热模态并非与 main 冗余 → 融合应在 main 不可靠时用热模态的 residual 修正,
而非全局概率平均(会把 B=14.5% 拖累区也带进来)。

结构 (审计公式):
    z_m = main_encoder(x_main)         # [B,512]
    z_t = thermal_encoder(x_thermal)   # [B,512]
    g   = sigmoid(gate_mlp([z_m,z_t])) # [B,512] 条件门控
    dr  = residual_mlp(z_t)            # [B,512] 热→残差
    z   = z_m + g * dr                 # main + 受控修正
    logits = head(z)                   # [B,40] 单 head

- loss = CE(logits) + λ_m CE(main_logits) [主塔温强约束, 防 residual 推翻太多]
- 不用第二个独立 40 分类 head 强迫热模态独学(审计点4)
用法:
  python scripts/train_cond_residual.py --fold 0 [--epochs 60] [--lambda_main 0.3]
  --freeze_enc 1 (默认冻结 encoder 前段, 显式学 fusion)
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
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalVideoDataset,
                         build_thermal_index, build_train_index, build_balanced_sampler)
from src.model import build_model
from src.split import split_by_subject


class DualStreamDataset(torch.utils.data.Dataset):
    """main+thermal 天然对齐: 一个 Dataset 同时返回两流, 避免双 DataLoader 错位。
    (训练用同一 index → 平衡采样依然正确)
    """
    def __init__(self, main_ds, th_ds):
        assert len(main_ds) == len(th_ds)
        self.main_ds = main_ds
        self.th_ds = th_ds

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, i):
        xm, y, _ = self.main_ds[i]
        xt, _, _ = self.th_ds[i]
        return xm, xt, y


class CondResidualFusion(nn.Module):
    """main+thermal 融合 (审计 P0/P1).
    mode='residual': z = z_m + gate·R(z_t)  (热模态修 main)
    mode='concat'  : z = MLP(concat[z_m,z_t]) (纯中间融合对照)
    freeze_pct: 冻结 encoder 层的前 X% (审计: 冻结80%查融合本身价值)
    """
    def __init__(self, in_ch_m=4, in_ch_t=3, n_segment=16, weights_m="ig65m_r2plus1d34.pth",
                 weights_t="ig65m_r2plus1d34.pth", dim=512, num_classes=40,
                 freeze_enc=0, mode="residual"):
        super().__init__()
        self.mode = mode
        self.main_enc = build_model("r2plus1d34", num_classes=num_classes, in_channels=in_ch_m,
                                    n_segment=n_segment, weights_path=weights_m)
        self.th_enc = build_model("r2plus1d34", num_classes=num_classes, in_channels=in_ch_t,
                                  n_segment=n_segment, weights_path=weights_t)
        if mode == "residual":
            self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid())
            self.resid = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.head = nn.Linear(dim, num_classes)
        else:  # concat 中间融合
            self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, num_classes))
        if freeze_enc > 0:
            # 冻结 encoder 前 1-freeze_enc 比例层(main/th 一致); 只留最后一小块 + 融合头可训练
            for enc in (self.main_enc, self.th_enc):
                names = list(enc.named_parameters())
                keep_n = max(0, int((1 - freeze_enc) * len(names)))
                keep_set = {n for n, _ in names[len(names) - keep_n:]}
                for n, p in enc.named_parameters():
                    p.requires_grad_(False if n not in keep_set else True)

    def forward(self, xm, xt, ret_main=False):
        zm = self.main_enc.encoder(xm.permute(0, 2, 1, 3, 4))
        zt = self.th_enc.encoder(xt.permute(0, 2, 1, 3, 4))
        if self.mode == "residual":
            g = self.gate(torch.cat([zm, zt], -1))
            zm = zm + g * self.resid(zt)
            lg = self.head(zm)
        else:
            lg = self.fuse(torch.cat([zm, zt], -1))
        if ret_main:
            return lg, self.main_enc.head(zm)
        return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_main", type=float, default=0.3)
    ap.add_argument("--freeze_enc", type=float, default=1.0,
                    help="0~1: 冻结 encoder 层数值(如0.8=冻结80%, 默认1.0=全冻结只训融合头)")
    ap.add_argument("--mode", type=str, default="residual",
                    choices=["residual", "concat"],
                    help="residual=z_m+gate*R(z_t); concat=MLP(concat[z_m,z_t])")
    ap.add_argument("--save_dir", default="outputs/cond_resid")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()

    # 双流 clip 对齐: main(DepthIR) 与 thermal 同 key(action/subject/sample)
    clips = build_train_index(root)
    th_clips = build_thermal_index(root)
    th_by_key = {f"{c.action_id}/{c.subject}/{c.sample}": c for c in th_clips}
    paired = []
    for c in clips:
        k = f"{c.action_id}/{c.subject}/{c.sample}"
        if k in th_by_key:
            paired.append((c, th_by_key[k]))
    print(f"main+thermal 双流对齐 clips: {len(paired)}/{len(clips)}", flush=True)
    if len(paired) < 100:
        raise SystemExit("❌ thermal 对齐样本太少")

    # 环境平衡 fold (dev_folds 优先; 无则回退 split_by_subject)
    FS = Path("outputs/dev_folds.json")
    if FS.exists():
        spec = json.loads(FS.read_text())
        lockset = set(spec.get("locked", []))
        dfold = spec["dev_folds"][args.fold]
        tr_s, va_s = set(dfold["tr"]), set(dfold["va"])
        tr = [p for p in paired if p[0].subject in tr_s and p[0].subject not in lockset]
        va = [p for p in paired if p[0].subject in va_s and p[0].subject not in lockset]
        print(f"[protocol] dev_folds fold{args.fold}: tr={len(tr)} va={len(va)} "
              f"locked排除={sorted(lockset)}", flush=True)
    else:
        folds = split_by_subject(clips, n_folds=args.folds)
        tr_idx, va_idx = folds[args.fold]
        tr_s = {clips[i].subject for i in tr_idx}; va_s = {clips[i].subject for i in va_idx}
        tr = [p for p in paired if p[0].subject in tr_s]
        va = [p for p in paired if p[0].subject in va_s]

    mc = json.loads(Path("bbox_train.json").read_text())
    tc = json.loads(Path("bbox_thermal_train.json").read_text())
    m_clips = [p[0] for p in tr]; t_clips = [p[1] for p in tr]
    mva = [p[0] for p in va]; tva = [p[1] for p in va]
    tr_ds = DepthIRVideoDataset(m_clips, args.num_frames, args.size, True, mc)
    tth_ds = ThermalVideoDataset(t_clips, args.num_frames, args.size, True, tc)
    va_m = DepthIRVideoDataset(mva, args.num_frames, args.size, False, mc)
    va_t = ThermalVideoDataset(tva, args.num_frames, args.size, False, tc)

    # 双流对齐: 用 DualStreamDataset(单 loader, 天然同步) — 避免双 DataLoader 错位
    tr_dual = DualStreamDataset(tr_ds, tth_ds)
    va_dual = DualStreamDataset(va_m, va_t)
    sampler = build_balanced_sampler([c.action_id for c in m_clips])
    tr_ld = DataLoader(tr_dual, batch_size=args.batch_size, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_dual, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = CondResidualFusion(4, 3, args.num_frames, freeze_enc=args.freeze_enc,
                              mode=args.mode).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def valid():
        model.eval()
        c = t = 0
        for xm, xt, y in va_ld:
            xm, xt, y = xm.to(device), xt.to(device), y.to(device)
            lg = model(xm, xt)
            c += (lg.argmax(-1) == y).sum().item()
            t += len(y)
        model.train()
        return c / max(t, 1)

    best = 0.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = n = 0
        for xm, xt, y in tr_ld:
            xm, xt, y = xm.to(device), xt.to(device), y.to(device)
            lg, lg_m = model(xm, xt, ret_main=True)
            loss = crit(lg, y) + args.lambda_main * crit(lg_m, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * y.numel(); n += y.numel()
        sched.step()
        va_acc = valid()
        if va_acc > best:
            best = va_acc
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(args.save_dir) / f"condr_fold{args.fold}.pth")
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run/max(n,1):.4f} "
              f"val={va_acc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"== fold{args.fold} best={best:.4f} ==")


if __name__ == "__main__":
    main()
