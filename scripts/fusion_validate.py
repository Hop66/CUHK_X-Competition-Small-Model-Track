#!/usr/bin/env python3
"""CUHK-X —— fold 级多模态融合验证（回答"单 seed main + 单 seed thermal 能否超过 2×main"）

在 val 折上用同折 ckpt 推理各模态（支持每模态多 ckpt → logits 平均），输出：
  - 各模态单独 acc
  - main+thermal / main+thermal+skeleton 的概率平均、logits 平均、置信门控 acc
  - oracle 完美选择上限
  - 每样本"融合救回 main 的错"数量（只救不伤核心指标）

用法（服务器）:
  python scripts/fusion_validate.py --fold 0 \
    --main_ckpts outputs/xxx/r2plus1d34_depthir_fold0.pth \
    --thermal_ckpts outputs/thermal_v2/r2plus1d34_thermal_fold0.pth \
    [--skeleton_ckpts outputs/skeleton_3d/3d/motionbert_fold0.pth]
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex,
                         ThermalVideoDataset, build_train_index)
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.model import build_model
from src.skeleton_dataset import MotionBertSkeletonDataset
from src.split import split_by_subject


def load_sd(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "main" in sd:
        sd = sd["main"]
    return sd


def load_mb(ckpt, device):
    bb = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                   num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6))
    m = ActionNet(backbone=bb, dim_rep=512, num_classes=40, dropout_ratio=0.5,
                  version="class", hidden_dim=512, num_joints=17).to(device)
    m.load_state_dict(load_sd(ckpt))
    return m


def infer_logits(model, loader, device, is_skel=False):
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            if is_skel:
                x = x.unsqueeze(1)
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs)


def logits_of_ckpts(ckpts, build, loader, device, is_skel=False):
    """多个 ckpt 独立推理 → logits 平均。返回 [N,40] logits。"""
    accs = []
    acc_all = None
    for ckpt in ckpts:
        model = build(ckpt)
        L = infer_logits(model, loader, device, is_skel)
        acc_all = L if acc_all is None else acc_all + L
        accs.append(L)
    return acc_all / len(ckpts), accs


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def report(name, acc):
    print(f"  {name:<28} acc = {acc:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--main_ckpts", nargs="+", required=True,
                    help="main ckpt 列表（同折多折/多 seed → logits 平均）")
    ap.add_argument("--thermal_ckpts", nargs="+", default=[])
    ap.add_argument("--skeleton_ckpts", nargs="+", default=[])
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_train.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    folds = split_by_subject(main_clips, 3)
    _, val_idx = folds[args.fold]
    va_main = [main_clips[i] for i in val_idx]
    labels = np.array([c.action_id for c in va_main], dtype=np.int64)
    n = len(labels)
    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))
    print(f"fold{args.fold} val {n} clips", flush=True)

    def build_main(ckpt):
        m = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=16).to(device)
        m.load_state_dict(load_sd(ckpt))
        return m

    ds = DepthIRVideoDataset(va_main, 16, 128, False, main_crop, use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    main_logits, main_each = logits_of_ckpts(args.main_ckpts, build_main, loader, device)
    print("\n==== 各模态单独 acc ====", flush=True)
    report("main(平均)", (main_logits.argmax(-1) == labels).mean())
    for i, L in enumerate(main_each):
        report(f"main_ckpt{i}", (L.argmax(-1) == labels).mean())

    # thermal
    th_logits = None
    if args.thermal_ckpts:
        def build_th(ckpt):
            m = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=16).to(device)
            m.load_state_dict(load_sd(ckpt))
            return m
        th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                     root / "Thermal" / c.depth_dir.parent.parent.name
                                     / c.subject / c.sample) for c in va_main]
        ds_t = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
        loader_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
        th_logits, th_each = logits_of_ckpts(args.thermal_ckpts, build_th, loader_t, device)
        report("thermal(平均)", (th_logits.argmax(-1) == labels).mean())
        for i, L in enumerate(th_each):
            report(f"thermal_ckpt{i}", (L.argmax(-1) == labels).mean())

    # skeleton
    sk_logits = None
    if args.skeleton_ckpts:
        def build_sk(ckpt):
            return load_mb(ckpt, device)
        skel_clips = [type("C", (), {"action_id": c.action_id, "subject": c.subject,
                                     "sample": c.sample,
                                     "pred_dir": root / "Skeleton"
                                     / c.depth_dir.parent.parent.name
                                     / c.subject / c.sample / "predictions"})() for c in va_main]
        ds_s = MotionBertSkeletonDataset(skel_clips, 16, False)
        loader_s = DataLoader(ds_s, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
        sk_logits, _ = logits_of_ckpts(args.skeleton_ckpts, build_sk, loader_s, device,
                                        is_skel=True)
        report("skeleton(平均)", (sk_logits.argmax(-1) == labels).mean())

    # ---- 融合 ----
    def prob_avg_fusion(pairs, w=None):
        """pairs=[(logits,...)] 概率平均。返回 [N,40] 概率。"""
        ps = [softmax(L) for L in pairs]
        if w is None:
            w = [1.0 / len(ps)] * len(ps)
        return sum(p * wi for p, wi in zip(ps, w))

    def logit_avg_fusion(pairs):
        return sum(softmax(L) for L in pairs) / len(pairs)

    def gated_fusion(main_logits, aux_logits, labels, steps=201):
        main_prob = softmax(main_logits)
        main_conf = main_prob.max(-1)
        main_pred = main_logits.argmax(-1)
        base = (main_pred == labels).mean()
        aux_prob = softmax(aux_logits) if aux_logits is not None else None
        best = (base, 1.0, 0)
        for tau in np.linspace(0.0, 1.0, steps):
            mask = main_conf < tau
            pred = main_pred.copy()
            if aux_prob is not None and mask.sum() > 0:
                fused = softmax(main_logits[mask]) + aux_prob[mask]
                pred[mask] = fused.argmax(-1)
            acc = (pred == labels).mean()
            if acc > best[0] + 1e-9:
                best = (acc, tau, int(mask.sum()))
        return base, best

    print("\n==== 融合对比 ====", flush=True)
    combos = []
    if th_logits is not None:
        combos.append(("main+thermal", [main_logits, th_logits]))
    if sk_logits is not None:
        combos.append(("main+skeleton", [main_logits, sk_logits]))
    if th_logits is not None and sk_logits is not None:
        combos.append(("main+thermal+skeleton", [main_logits, th_logits, sk_logits]))

    for name, pairs in combos:
        p = prob_avg_fusion(pairs)
        report(f"{name} prob_avg", (p.argmax(-1) == labels).mean())
        p2 = logit_avg_fusion(pairs)
        report(f"{name} logit_avg", (p2.argmax(-1) == labels).mean())
        base, (g_acc, tau, n_resc) = gated_fusion(pairs[0], np.mean(pairs[1:], axis=0), labels)
        report(f"{name} gated(τ={tau:.2f})", g_acc)
        print(f"       gated 净增益 = {g_acc - base:+.4f}（救援 {n_resc}/{n}）", flush=True)

    # ---- oracle 上限（完美选择器：每样本选对的那个模态）----
    print("\n==== oracle 上限 ====", flush=True)
    base_main = (main_logits.argmax(-1) == labels)
    for name, pairs in combos:
        # oracle: 每个样本若任一模态对则对
        ok = base_main.copy()
        for L in pairs[1:]:
            ok |= (L.argmax(-1) == labels)
        print(f"  {name:<28} oracle = {ok.mean():.4f}（vs main {base_main.mean():.4f}）", flush=True)

    # ---- 只救不伤诊断：thermal 纠正 main 的错误数 ----
    if th_logits is not None:
        main_pred = main_logits.argmax(-1)
        th_pred = th_logits.argmax(-1)
        rescued = (~base_main) & (th_pred == labels)

        print("\n==== thermal 互补诊断 ====", flush=True)
        print(f"  main 错 thermal 对（可救）: {int(rescued.sum())}/{int((~base_main).sum())} "
              f"({rescued.mean()/max((~base_main).mean(),1e-9):.2%})", flush=True)


if __name__ == "__main__":
    main()
