#!/usr/bin/env python3
"""CUHK-X —— 伪标签自训练（官方允许：测试数据伪标签 + 自训练，不违规）

用户策略（2026-08-24）：
  1. 严格筛选：多模型 Co-training（main+thermal top1 一致）+ 动态 top-K 置信度
  2. 主动防错：伪标签样本用低训练权重（可靠性加权），fold0 val 早停防过拟合
  3. 防过拟合：仅高置信 top-K 伪样本、低权重、held-out val 监控、单轮自训练

两阶段（单 fold 验证）：
  stage=pseudo   : 用全量 main+thermal（seed42）对 405 测试打伪标签
                   → 一致性筛选 + top-K → pseudo_labels.json
  stage=selftrain: fold0 训练集(真标签) + 伪标签测试子集 重训 → fold0 val 对比
                   （baseline: thermal fold0 0.6339 / main fold0 0.6609）

用法:
  python scripts/pseudo_label_selftrain.py --stage pseudo
  python scripts/pseudo_label_selftrain.py --stage selftrain --modality thermal --fold 0
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
from torch.utils.data import DataLoader, Dataset

from src.dataset import (DepthIRVideoDataset, ThermalVideoDataset, build_balanced_sampler,
                         build_test_index, build_thermal_index, build_train_index)
from src.model import build_model
from src.split import split_by_subject


# ----------------------------------------------------------------------------
# Stage 1：打伪标签（Co-training + 动态 top-K）
# ----------------------------------------------------------------------------
@torch.no_grad()
def _infer_test_probs(model, ds, device, batch_size, workers):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    probs = []
    for x, _, _ in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, dim=-1).float().cpu().numpy())
    return np.concatenate(probs, 0)


def _load_full_model(in_channels, ckpt, device):
    model = build_model("r2plus1d34", num_classes=40, in_channels=in_channels,
                        weights_path=None).to(device)
    sd = torch.load(str(ckpt), map_location=device)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if "model" in sd:
        sd = sd["model"]
    # 兼容 dual(SM) ckpt：键带 static.* 前缀 → 剥前缀取纯 R2+1D（丢弃 motion.*/a），否则原样
    if any(k.startswith("static.") for k in sd):
        sd = {k[len("static."):]: v for k, v in sd.items() if k.startswith("static.")}
    model.load_state_dict(sd)
    return model


def generate_pseudo_labels(args, device):
    test_root = Path(args.test_root).expanduser()
    raw = build_test_index(test_root)  # [(cid, depth_dir, ir_dir)]
    n = len(raw)
    print(f"[pseudo] 测试 clips={n}，加载 main+thermal 全量模型打伪标签", flush=True)

    # main：Depth+IR 4ch
    main_model = _load_full_model(4, args.main_ckpt, device)
    main_crop_raw = json.loads(Path(args.main_test_crop).expanduser().read_text(encoding="utf-8"))
    # 修复：dataset 按 {action_id}/{subject}/{sample} 查键（此处 = -1/{cid}/{cid}），测试缓存键是裸 cid
    #       → 重映射成 dataset 会查的键，否则 teacher 在全帧上推理（与 0.73 提交 crop 分布不一致）
    main_crop = {f"-1/{cid}/{cid}": w for cid, w in main_crop_raw.items()}
    main_clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                 "depth_dir": dd, "ir_dir": ir})()
                  for cid, dd, ir in raw]
    main_ds = DepthIRVideoDataset(main_clips, args.num_frames, args.size, False, main_crop,
                                  use_frame_diff=False)
    main_prob = _infer_test_probs(main_model, main_ds, device, args.batch_size, args.workers)

    # thermal：3ch
    th_model = _load_full_model(3, args.thermal_ckpt, device)
    th_crop_raw = json.loads(Path(args.thermal_test_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = {f"-1/{cid}/{cid}": w for cid, w in th_crop_raw.items()}
    th_clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                               "thermal_dir": dd.parent / "Thermal"})()
                for cid, dd, _ in raw]
    th_ds = ThermalVideoDataset(th_clips, args.num_frames, args.size, False, th_crop,
                                use_frame_diff=False)
    th_prob = _infer_test_probs(th_model, th_ds, device, args.batch_size, args.workers)

    # Co-training 筛选：top1 一致 + 保守置信度（两模型 max prob 的 min）
    m_top1 = main_prob.argmax(-1)
    t_top1 = th_prob.argmax(-1)
    agree = (m_top1 == t_top1)
    m_conf = main_prob.max(-1)
    t_conf = th_prob.max(-1)
    conf = np.minimum(m_conf, t_conf)  # 保守
    cand = np.where(agree)[0]
    cand_conf = conf[cand]
    k = min(args.pseudo_k, len(cand))
    # 动态 top-K：按置信度降序选最有把握的 K 个
    top_idx = cand[np.argsort(-cand_conf)[:k]]
    pseudo = {raw[i][0]: {"label": int(m_top1[i]), "conf": float(conf[i])}
              for i in top_idx}
    out = Path(args.pseudo_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pseudo, indent=1), encoding="utf-8")
    print(f"[pseudo] 一致={agree.sum()}/{n}，采纳 top-{len(pseudo)}（置信度 "
          f"{np.mean(list(p['conf'] for p in pseudo.values())):.3f}）→ {out}", flush=True)


# ----------------------------------------------------------------------------
# Stage 2：自训练（fold0 真标签 + 伪标签低权重）
# ----------------------------------------------------------------------------
class _WeightedDataset(Dataset):
    """包装：每样本返回 (x, y, weight)。真标签 weight=1.0；伪标签 weight=pseudo_weight。"""

    def __init__(self, base, weights):
        self.base = base
        self.weights = weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, _ = self.base[i]
        return x, y, self.weights[i]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        correct += (model(x.to(device)).argmax(-1) == y.to(device)).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def selftrain(args, device):
    root = Path(args.train_root).expanduser()
    crop_path = args.crop_cache or ("bbox_thermal_train.json" if args.modality == "thermal"
                                    else "bbox_train.json")
    crop = json.loads(Path(crop_path).expanduser().read_text(encoding="utf-8"))
    in_channels = 3 if args.modality == "thermal" else 4
    ds_cls = ThermalVideoDataset if args.modality == "thermal" else DepthIRVideoDataset
    idx_cls = build_thermal_index if args.modality == "thermal" else build_train_index

    real_clips = idx_cls(root)
    if args.full:
        tr_real = real_clips
        va_real = []
    else:
        folds = split_by_subject(real_clips, n_folds=args.folds)
        tr_idx, va_idx = folds[args.fold]
        tr_real = [real_clips[i] for i in tr_idx]
        va_real = [real_clips[i] for i in va_idx]

    # 伪标签测试 clip → 构造同模态 clip（action_id=伪标签）
    pseudo = json.loads(Path(args.pseudo_json).expanduser().read_text(encoding="utf-8"))
    test_root = Path(args.test_root).expanduser()
    raw_by_id = {cid: (dd, ir) for cid, dd, ir in build_test_index(test_root)}
    pseudo_clips = []
    for cid, p in pseudo.items():
        dd, ir = raw_by_id[cid]
        if args.modality == "thermal":
            pseudo_clips.append(type("C", (), {"action_id": p["label"], "subject": cid,
                                               "sample": cid, "thermal_dir": dd.parent / "Thermal"}))
        else:
            pseudo_clips.append(type("C", (), {"action_id": p["label"], "subject": cid,
                                               "sample": cid, "depth_dir": dd, "ir_dir": ir}))
    # 修复：伪 clip 真实来源是测试集（裸 cid）→ 把测试 crop 重映射进假训练键 {label}/{cid}/{cid}，
    #       否则伪样本全帧、与真样本有 crop 混用（历史伪标签负结果根因）
    _tc_src = args.thermal_test_crop if args.modality == "thermal" else args.main_test_crop
    _tc = json.loads(Path(_tc_src).expanduser().read_text(encoding="utf-8")) \
        if Path(_tc_src).expanduser().exists() else {}
    for _cid, _p in pseudo.items():
        if _cid in _tc:
            crop[f"{_p['label']}/{_cid}/{_cid}"] = _tc[_cid]
    print(f"[selftrain] modality={args.modality} 真训练={len(tr_real)} "
          f"伪标签={len(pseudo_clips)}（weight={args.pseudo_weight}）", flush=True)

    # 合并训练集 + 权重（真 1.0 / 伪 pseudo_weight）
    all_train = tr_real + pseudo_clips
    weights = [1.0] * len(tr_real) + [args.pseudo_weight] * len(pseudo_clips)
    tr_base = ds_cls(all_train, args.num_frames, args.size, True, crop,
                     aug_strength=args.aug_strength)
    tr_ds = _WeightedDataset(tr_base, weights)
    sampler = build_balanced_sampler([c.action_id for c in all_train])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    va_loader = None
    if not args.full and va_real:
        va_ds = ds_cls(va_real, args.num_frames, args.size, False, crop)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

    model = build_model("r2plus1d34", num_classes=40, in_channels=in_channels,
                        n_segment=args.num_frames,
                        weights_path=args.weights or None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # StepLR 不衰减到 0（CosineAnnealing 衰减到 ~0 → 后段不学；全量 80ep 尤其要持续学习）
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    best, no_improve = 0.0, 0
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = "full" if args.full else f"fold{args.fold}"
    save_path = save_dir / f"{args.modality}_selftrain_{tag}.pth"
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, w in tr_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            loss = (crit(model(x), y) * w).mean()  # 可靠性加权 CE
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        sched.step()
        if va_loader is not None:
            acc = evaluate(model, va_loader, device)
            if acc > best:
                best, no_improve = acc, 0
                torch.save({"model": model.state_dict(), "best_acc": best}, save_path)
            else:
                no_improve += 1
            print(f"[fold{args.fold}] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"val={acc:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if no_improve >= args.patience:
                print(f"[fold{args.fold}] early stop @ ep{ep+1}", flush=True)
                break
        else:
            # 全量模式：无 val，每 ep 覆盖保存（提交用最终模型）
            torch.save({"model": model.state_dict(), "epoch": ep + 1}, save_path)
            print(f"[full] ep{ep+1}/{args.epochs} loss={run_loss/max(n,1):.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print(f"== selftrain {args.modality} {tag} done → {save_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="pseudo", choices=["pseudo", "selftrain"])
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--test_root", type=str,
                    default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--modality", type=str, default="thermal", choices=["thermal", "main"])
    ap.add_argument("--weights", type=str, default="ig65m_r2plus1d34.pth")
    # 打伪标签用模型（全量 seed42）
    ap.add_argument("--main_ckpt", type=str,
                    default="outputs/main_full/r2plus1d34_depthir_full_seed42.pth")
    ap.add_argument("--thermal_ckpt", type=str,
                    default="outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth")
    ap.add_argument("--main_test_crop", type=str, default="bbox_test.json")
    ap.add_argument("--thermal_test_crop", type=str, default="bbox_thermal_test.json")
    # 自训练
    ap.add_argument("--crop_cache", type=str, default="",
                    help="训练 crop 缓存；缺省按 modality 自动选（否则 main 会误用 thermal 缓存）")
    ap.add_argument("--pseudo_json", type=str, default="outputs/pseudo_labels.json")
    ap.add_argument("--pseudo_k", type=int, default=120, help="动态 top-K 伪标签样本数")
    ap.add_argument("--pseudo_weight", type=float, default=0.5,
                    help="伪标签样本 CE 权重（低=防噪声主导，0.3-0.5）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="全量模式：用全部训练数据+伪标签，无 fold val（提交用，epochs 应设 80）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--save_dir", type=str, default="outputs/selftrain")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} stage={args.stage}", flush=True)
    if args.stage == "pseudo":
        generate_pseudo_labels(args, device)
    else:
        selftrain(args, device)


if __name__ == "__main__":
    main()
