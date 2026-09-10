#!/usr/bin/env python3
"""Radar 单模态摸底 + OOF(与 IMU 同款管线, 结构复用 IMUCNN 3conv+GAP, 输入 [B,T,13])。
产物: outputs/radar/radar_fold{i}.pth + outputs/oof/radar_fold{i}.pkl(Main/thermal OOF 同 key)。
判据: main+th+radar α 低权融合 3 折是否正(类比 IMU)。
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_balanced_sampler, build_train_index
from src.radar_dataset import FEAT_DIM, RadarDataset, build_radar_index
from src.split import split_by_subject


class RadarCNN(nn.Module):
    """1D CNN(仿 IMUCNN): 输入 [B, T, 13] -> [B, 40]。"""
    def __init__(self, num_classes: int = 40, feat_dim: int = FEAT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.mean(dim=-1)
        return self.head(x)


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
def oof_logits(model, va_clips, loader, device, path):
    model.eval()
    out = {}
    s = 0
    for x, _, subj in loader:
        lg = model(x.to(device)).cpu().numpy().astype(np.float32)
        for j in range(len(x)):
            c = va_clips[s + j]
            out[f"{c.action_id}/{c.subject}/{c.sample}"] = lg[j]
        s += len(x)
    v = np.array([int(k.split('/')[0]) for k in out])
    acc = float((np.argmax(np.stack(list(out.values())), -1) == v).mean())
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--save_dir", default="outputs/radar")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    clips = build_radar_index(root, main_clips)
    print(f"device={device} clips={len(clips)}", flush=True)

    folds = split_by_subject(clips, n_folds=args.folds)
    target = range(len(folds)) if args.fold < 0 else [args.fold]
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs/oof").mkdir(parents=True, exist_ok=True)

    for fi in target:
        tr_idx, va_idx = folds[fi]
        tr_clips = [clips[i] for i in tr_idx]
        va_clips = [clips[i] for i in va_idx]
        tr_ds = RadarDataset(tr_clips, args.T, True, seed=42)
        va_ds = RadarDataset(va_clips, args.T, False)
        sampler = build_balanced_sampler([c.action_id for c in tr_clips]) if args.balanced else None
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
        model = RadarCNN().to(device)
        ck = save_dir / f"radar_fold{fi}.pth"
        best = train_fold(model, tr_loader, va_loader, device, args.epochs, fi, ck, args.lr)
        sd = torch.load(ck, map_location=device)
        model.load_state_dict(sd["model"])
        ok = f"outputs/oof/radar_fold{fi}.pkl"
        oof_acc = oof_logits(model, va_clips, va_loader, device, Path(ok))
        print(f"== fold{fi} best={best:.4f} oof_acc={oof_acc:.4f} -> {ck}", flush=True)


if __name__ == "__main__":
    main()
