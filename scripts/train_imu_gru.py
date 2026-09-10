#!/usr/bin/env python3
"""IMUGRUNet 复刻 AB(专 notebook skeleton-imu-specialist) —— 结构效应单变量对照。

用户质疑: 我们对 IMUCN(3conv+GAP) 判融合正, 但他的 IMUGRUNet(Conv1DBlock残差+BiGRU+attention) 结构不同, 不能外推。
→ 同输入(30ch 时间对齐)、同 split、同训练器(Adam 1e-3/30ep/balanced), 只换结构 = IMUGRUNet。
产物: outputs/imu_gru/imu_gru_fold{i}.pth + outputs/oof/imu_gru_fold{i}.pkl(key 与 main OOF 对齐)。
对照: outputs/imu/imu_fold{0,1,2}.pth(IMUCNN) 的 main+th α 融合三折全正 +1.26。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler, build_train_index
from src.imu_dataset import FEAT_DIM, IMUDataset, build_imu_index
from src.split import split_by_subject


class Conv1DBlock(nn.Module):
    """notebook IMUGRUNet 残差 conv 块。"""
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, s, k // 2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, k, 1, k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, s, bias=False), nn.BatchNorm1d(out_ch)
        ) if (in_ch != out_ch or s != 1) else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(x) + self.skip(x))


class IMUGRUNet(nn.Module):
    """专复刻: 4×Conv1DBlock(32/64/128/256) + AdaptiveAvgPool1d(16) + 2层BiGRU + attention + MLP head。
    输入 [B, T, 30](沿用我们 30ch 时间对齐, 只换结构)。"""
    def __init__(self, num_classes: int = 40, in_ch: int = FEAT_DIM, dropout: float = 0.3):
        super().__init__()
        convs = []
        ch = in_ch
        for out in (32, 64, 128, 256):
            convs.append(Conv1DBlock(ch, out, s=(1 if out == 32 else 2)))
            ch = out
        self.cnn = nn.Sequential(*convs, nn.AdaptiveAvgPool1d(16))
        self.gru = nn.GRU(256, 128, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(256, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):  # [B, T, 30]
        f = self.cnn(x.permute(0, 2, 1))      # [B, 256, 16]
        f = f.permute(0, 2, 1)                 # [B, 16, 256]
        g, _ = self.gru(f)                     # [B, 16, 256]
        w = torch.softmax(self.attn(g), dim=1)
        return self.head((g * w).sum(dim=1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_fold(model, tr_loader, va_loader, device, epochs, fold, save_path, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_sd = 0.0, None
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        acc = evaluate(model, va_loader, device)
        if acc > best:
            best = acc
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[fold{fold}] ep{ep+1}/{epochs} loss={run_loss/max(n,1):.4f} "
              f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
    if save_path is not None:
        torch.save({"model": best_sd, "best_acc": best}, save_path)
    return best


@torch.no_grad()
def oof_logits(model, va_clips, loader, device, T, path):
    model.eval()
    out = {}
    s = 0
    for x, _, subj in loader:
        lg = model(x.to(device)).cpu().numpy().astype(np.float32)
        for j in range(len(x)):
            c = va_clips[s + j]
            out[f"{c.action_id}/{c.subject}/{c.sample}"] = lg[j]
        s += len(x)
    v = np.array([k.split('/')[0] for k in out])
    acc = float((np.argmax(np.stack(list(out.values())), -1) == v.astype(int)).mean())
    with open(path, "wb") as fh:
        import pickle
        pickle.dump(out, fh)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--no_jitter", action="store_true")
    ap.add_argument("--save_dir", default="outputs/imu_gru")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    clips = build_imu_index(root, main_clips)
    print(f"device={device} clips={len(clips)}", flush=True)

    folds = split_by_subject(clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_clips = [clips[i] for i in tr_idx]
        va_clips = [clips[i] for i in va_idx]
        tr_ds = IMUDataset(tr_clips, args.T, True, seed=42, jitter=not args.no_jitter)
        va_ds = IMUDataset(va_clips, args.T, False)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
        model = IMUGRUNet().to(device)
        ck = save_dir / f"imu_gru_fold{fi}.pth"
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, fi, ck, args.lr)
        # best ckpt 推理 val OOF
        sd = torch.load(ck, map_location=device)
        model.load_state_dict(sd["model"])
        ok = f"outputs/oof/imu_gru_fold{fi}.pkl"
        oof_acc = oof_logits(model, va_clips, va_loader, device, args.T, Path(ok))
        print(f"== fold{fi} best={best:.4f} oof_acc={oof_acc:.4f} -> {ck}", flush=True)


if __name__ == "__main__":
    main()
