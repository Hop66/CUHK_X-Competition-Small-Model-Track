#!/usr/bin/env python3
"""L2: 骨架作为"训练监督"/aux-head 强化 main —— 推理纯相机(砍掉 aux)。
main 的 encoder(全局池化 B,512) 接 小头预测 clip 骨架归一化姿态(51), aux loss 正则主干把姿势结构编进视觉表征。
fold0 main nf16 vs 0.6695; 推理只用 main(model 输出), aux head 只在训练参与。
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> import twin_v3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index, DepthIRVideoDataset
from src.model import build_model
from src.skeleton_dataset import build_skeleton_index, frame_num_of
from src.imu_dataset import build_imu_index, load_imu_sequence, time_align
from src.split import split_by_subject

# 数据驱动的孪生难对集(最强骨架特征 v3+IMU audit 选出的 weak 对)
HARD_PAIRS = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
HPID = {a for p in HARD_PAIRS for a in p}
NHP = len(HARD_PAIRS)

IMU_DIM = 20


def imu_target(imu_dir):
    """IMU [T,30] → 20维(5设备×mean/std/max/p90 幅值), 供 aux 回归。"""
    dev = load_imu_sequence(Path(imu_dir))
    x = time_align(dev, T=128)
    if x is None or not np.any(x):
        return np.zeros(IMU_DIM, np.float32)
    f = []
    for d in range(5):
        ch = np.abs(x[:, d * 6:d * 6 + 6])
        f += [float(ch.mean()), float(ch.std()), float(ch.max()), float(np.percentile(ch, 90))]
    return np.array(f[:IMU_DIM], np.float32)


def norm_pose(pred_dir):
    """clip 骨架(中帧) → 51 维 归一化姿态(k-kp[0])/身高。"""
    fs = sorted(Path(pred_dir).glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    if len(fs) < 2:
        return None
    o = json.loads(fs[len(fs) // 2].read_text("utf-8"))
    fr = o if isinstance(o, dict) else o[0]
    kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
    k = kp - kp[0]
    s = float(np.ptp(kp[:, 2])) + 1e-6
    return (k / s).astype(np.float32).reshape(-1)   # 51


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_aux", type=float, default=0.1)
    ap.add_argument("--aux", default="skel", choices=["skel", "imu", "hardpairs"],
                    help="aux 目标: skel=骨架姿态(51), imu=IMU统计(20), "
                         "hardpairs=难对骨架软判别(5对×2)")
    ap.add_argument("--weights", default="ig65m_r2plus1d34.pth")
    ap.add_argument("--save_dir", default="outputs/main_aux")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    folds = split_by_subject(clips, 3)
    tr_idx, va_idx = folds[args.fold]
    tr_clips = [clips[i] for i in tr_idx]
    va_clips = [clips[i] for i in va_idx]

    if args.aux == "skel":
        aux_map = {f"{c.action_id}/{c.subject}/{c.sample}": str(c.pred_dir)
                   for c in build_skeleton_index(root)}
        aux_dim = 51
        tgt_cache = {}
        for k, pd_ in aux_map.items():
            t = norm_pose(pd_)
            if t is not None:
                tgt_cache[k] = t
    elif args.aux == "imu":
        imu_map = {f"{c.subject}/{c.sample}": c
                   for c in build_imu_index(root, build_train_index(root))}
        aux_dim = IMU_DIM
        tgt_cache = {}
        for c in clips:
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            imc = imu_map.get(f"{c.subject}/{c.sample}")
            if imc is not None:
                tgt_cache[key] = imu_target(imc.imu_dir)
    else:  # hardpairs
        import shutil
        from twin_v3 import CACHE as V3_CACHE
        aux_dim = 2 * len(HARD_PAIRS)
        fe_cache = Path("outputs/twin_v3_cache.pkl")
        if not fe_cache.exists() and V3_CACHE.exists():
            # 登录节点 /tmp 与 slurm 节点不共享 -> 拷到项目内
            shutil.copy(V3_CACHE, fe_cache)
        if not fe_cache.exists():
            raise SystemExit(f"[hardpairs] 需要 outputs/twin_v3_cache.pkl (先跑 twin_v3.py)")
        FE = pickle.load(open(fe_cache, "rb"))
        # 当前 fold 的 train 样本: 对每个难对拟合骨架 GBC -> per-clip soft 目标
        hard = {}   # key -> { (a,b): np.array([pa,pb]) }
        for p in HARD_PAIRS:
            a, b = p
            mti = []
            for c in tr_clips:
                key = f"{c.action_id}/{c.subject}/{c.sample}"
                if key in FE and c.action_id in (a, b):
                    mti.append((key, FE[key]))
            if len(mti) < 12:
                print(f"[hardpairs] pair {p} 样本不足 {len(mti)}, 跳过", flush=True)
                continue
            clf = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=0)
            clf.fit(np.array([fx for _, fx in mti]),
                    [1 if int(k.split('/')[0]) == a else 0 for k, _ in mti])
            cls = list(clf.classes_)
            ia, ib = cls.index(1), cls.index(0)
            for k, _ in mti:
                pr = clf.predict_proba([FE[k]])[0]
                hard.setdefault(k, {})[p] = np.array([pr[ia], pr[ib]], np.float32)
        print(f"[hardpairs] Fitted GBC on {len(hard)} clips; fold{args.fold}",
              flush=True)
    print(f"fold{args.fold} tr={len(tr_clips)} va={len(va_clips)} aux={args.aux} targets={len(tgt_cache) if args.aux!='hardpairs' else len(hard)}",
          flush=True)

    ckpt_dir = Path(args.save_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    crop_cache = json.loads(Path("bbox_train.json").read_text(encoding="utf-8"))
    tr_ds = DepthIRVideoDataset(tr_clips, args.num_frames, args.size, True, crop_cache,
                                aug_strength=2, return_key=True)
    va_ds = DepthIRVideoDataset(va_clips, args.num_frames, args.size, False, crop_cache,
                                return_key=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                        weights_path=args.weights or None).to(device)
    feat_dim = 512
    aux_head = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(),
                             nn.Linear(256, aux_dim)).to(device)
    feats = {}
    def hook(mod, inp, out):
        feats["mid"] = out
    model.encoder.register_forward_hook(hook)

    opt = torch.optim.Adam(list(model.parameters()) + list(aux_head.parameters()),
                           lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    l1 = nn.SmoothL1Loss()

    def run_valid():
        model.eval()
        c = t = 0
        hc = ht = 0
        with torch.no_grad():
            for x, y, _, keys in va_loader:
                x, y = x.to(device), y.to(device)
                ok = (model(x).argmax(-1) == y)
                c += ok.sum().item()
                t += y.numel()
                hp = torch.tensor([int(k.split('/')[0]) in HPID for k in keys])
                if hp.any():
                    hc += ok[hp].sum().item()
                    ht += int(hp.sum().item())
        return c / max(t, 1), hc / max(ht, 1), ht

    best = 0.0
    best_sd = None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = n = 0
        for x, y, _, keys in tr_loader:
            x, y = x.to(device), y.to(device)
            feats.clear()
            logits = model(x)
            aux = aux_head(feats["mid"])
            if args.aux == "hardpairs":
                B = x.size(0)
                tg = torch.zeros(B, aux_dim, device=device)
                mask = torch.zeros(B, aux_dim, device=device)
                for i, k in enumerate(keys):
                    cid = int(k.split('/')[0])
                    hp = hard.get(k)
                    if hp is None or cid not in HPID:
                        continue
                    for pi, p in enumerate(HARD_PAIRS):
                        so = hp.get(p)
                        if so is None:
                            continue
                        tg[i, pi * 2:pi * 2 + 2] = torch.from_numpy(so).to(device)
                        mask[i, pi * 2:pi * 2 + 2] = 1.0
                logsm = F.log_softmax(aux.reshape(B, NHP, 2), dim=-1)
                loss_aux = -(logsm * tg.reshape(B, NHP, 2) * mask.reshape(B, NHP, 2)).sum() \
                    / max(mask.sum(), 1.0)
            else:
                tg = torch.stack([torch.from_numpy(tgt_cache.get(k, np.zeros(aux_dim, np.float32)))
                                  for k in keys]).to(device)
                loss_aux = l1(aux, tg)
            loss = crit(logits, y) + args.lambda_aux * loss_aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * y.numel(); n += y.numel()
        sched.step()
        val, hval, _ = run_valid()
        if val > best:
            best = val
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run/max(n,1):.4f} "
              f"val={val:.4f} best={best:.4f} hardpair={100*hval:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": best_sd, "best_acc": best}, ckpt_dir / f"main_aux_fold{args.fold}.pth")
    print(f"== fold{args.fold} best={best:.4f} hardpair={100*hval:.2f}% "
          f"(对照 main16f=0.6695, 难对区 main16f≈47%) ===", flush=True)


if __name__ == "__main__":
    main()
