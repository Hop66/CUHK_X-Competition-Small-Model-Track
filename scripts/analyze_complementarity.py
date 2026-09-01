#!/usr/bin/env python3
"""CUHK-X —— 模态互补性分析（回答"弱模态 val 够高是否该集成"）

在 val 折上分别推理 main 和辅助模态（skeleton/thermal），计算：
  - 各自准确率
  - 错误重叠（both/main_only/skel_only/neither）
  - **互补纠正率 = 辅助模态对、main 错的样本 / main 错误总数**（关键指标）
  - oracle 集成上限

判读：
  互补纠正率高（>0.3）→ 辅助模态能救 main 的错，集成有潜力
  互补纠正率低（≈随机）→ 两者错误重叠，集成无益（thermal/skeleton 已实测拖累）

用法: python scripts/analyze_complementarity.py --fold 0 [--thermal]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
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
    elif isinstance(sd, dict) and "main" in sd:  # 对比学习保存格式
        sd = sd["main"]
    return sd


def load_mb(ckpt, device):
    from functools import partial
    import torch.nn as nn
    bb = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                   num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6))
    m = ActionNet(backbone=bb, dim_rep=512, num_classes=40, dropout_ratio=0.5,
                  version="class", hidden_dim=512, num_joints=17).to(device)
    m.load_state_dict(load_sd(ckpt))
    return m


def predict(model, loader, device, is_skel=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            if is_skel:
                x = x.unsqueeze(1)
            preds.append(model(x).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def analyze(labels, main_pred, aux_pred, aux_name, per_class=False):
    main_ok = main_pred == labels
    aux_ok = aux_pred == labels
    both = (main_ok & aux_ok).sum()
    main_only = (main_ok & ~aux_ok).sum()
    aux_only = (~main_ok & aux_ok).sum()
    neither = (~main_ok & ~aux_ok).sum()
    n = len(labels)
    main_err = (~main_ok).sum()
    # 关键：辅助模态对 main 错误的纠正率（简单加权集成的判据）
    rescue_rate = aux_only / max(main_err, 1)
    # oracle 上限 = 完美选择器（每样本选对的那个模态）
    oracle = (both + main_only + aux_only) / n
    print(f"\n==== main vs {aux_name}（{n} 样本）====", flush=True)
    print(f"main acc={main_ok.mean():.4f} | {aux_name} acc={aux_ok.mean():.4f}", flush=True)
    print(f"both={both} | main_only={main_only} | {aux_name}_only={aux_only} | neither={neither}", flush=True)
    print(f"★ 互补纠正率（{aux_name} 救 main 的错，简单加权用）= {rescue_rate:.4f}", flush=True)
    print(f"oracle 完美选择上限 = {oracle:.4f}（vs main 单独 {main_ok.mean():.4f}）", flush=True)
    if per_class:
        print(f"\n==== per-class：main vs {aux_name}（判据：{aux_name} 胜类 = 选择性监督候选） ====",
              flush=True)
        for c in sorted(set(labels.tolist())):
            sel = labels == c
            n = int(sel.sum())
            if n < 4:
                continue
            mo = int((main_ok & sel).sum())
            ao = int((aux_ok & sel).sum())
            me = int((~main_ok & sel).sum())
            res = int((~main_ok & aux_ok & sel).sum())
            diff = (ao - mo) / n
            tag = "★ 辅助胜可比肩" if diff >= 0.10 else ("· 辅助远弱" if diff <= -0.10 else "")
            print(f"  {c:2d} n={n:3d} main={mo/n:.3f} {aux_name}={ao/n:.3f} "
                  f"rescue={res}/{me} diff={diff:+.3f} {tag}", flush=True)
        print("↑ 若存在多个 辅助胜 类(diff≥+0.10, n≥10) → 选择性蒸馏（仅这几类用骨架当 teacher）有空间；"
              "无 → 骨架整体弱于 main，任何监督都会拖累，路径关闭", flush=True)
    if rescue_rate > 0.3:
        print(f"结论：{aux_name} 简单加权集成值得试（纠正率高）", flush=True)
    elif oracle > main_ok.mean() + 0.05:
        print(f"结论：简单加权大概率拖累，但 oracle 高 → 可试样本级动态选择（main 低置信才听 {aux_name}）", flush=True)
    else:
        print(f"结论：{aux_name} 与 main 错误高度重叠，无集成价值", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--main_ckpt", default="outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth")
    ap.add_argument("--skeleton_ckpt", default="outputs/skeleton_mb/motionbert_fold0.pth")
    ap.add_argument("--thermal_ckpt", default="outputs/thermal_34/r2plus1d34_thermal_fold0.pth")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per_class", action="store_true",
                    help="输出 per-class main vs 辅助模态对比（选择性监督判据）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    main_clips = build_train_index(root)
    folds = split_by_subject(main_clips, 3)
    _, val_idx = folds[args.fold]
    va_main = [main_clips[i] for i in val_idx]
    labels = np.array([c.action_id for c in va_main], dtype=np.int64)

    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))

    # main（4ch 无帧差）
    ds = DepthIRVideoDataset(va_main, 16, 128, False, main_crop, use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    main_model = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=16).to(device)
    main_model.load_state_dict(load_sd(args.main_ckpt))
    main_pred = predict(main_model, loader, device)
    print(f"main acc = {(main_pred == labels).mean():.4f}", flush=True)

    # skeleton（MotionBERT）
    skel_clips = [type("C", (), {"action_id": c.action_id, "subject": c.subject, "sample": c.sample,
                                 "pred_dir": root / "Skeleton" / c.depth_dir.parent.parent.name
                                 / c.subject / c.sample / "predictions"})() for c in va_main]
    ds_s = MotionBertSkeletonDataset(skel_clips, 16, False)
    loader_s = DataLoader(ds_s, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    skel_model = load_mb(args.skeleton_ckpt, device)
    skel_pred = predict(skel_model, loader_s, device, is_skel=True)
    analyze(labels, main_pred, skel_pred, "skeleton", args.per_class)

    # thermal（可选）
    th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                 root / "Thermal" / c.depth_dir.parent.parent.name
                                 / c.subject / c.sample) for c in va_main]
    ds_t = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
    loader_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    th_model = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=16).to(device)
    th_model.load_state_dict(load_sd(args.thermal_ckpt))
    th_pred = predict(th_model, loader_t, device)
    analyze(labels, main_pred, th_pred, "thermal", args.per_class)


if __name__ == "__main__":
    main()
