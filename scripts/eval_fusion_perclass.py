#!/usr/bin/env python3
"""CUHK-X —— main + thermal(+skeleton) per-class 门控融合对拍（fold0 内留一被试验证，防自证）

动机（bundle 48378）：
- thermal 在 Drink_water(main .321 vs th .893)/Pour/Phone/Selfie 等手类碾压 → 均权 0.73 埋没了它
- skeleton 整体弱(.327) 但在 Write/Phone/Listen/Wipe_windows 4 类是"胜类" → 全量融合拖累，门控可捞回
- 均权(prob_avg) 与 Noisy-OR 都无增量 → 需要"类级"而非"样本级全体"

方法（防自证的留一被试验证）：
- 只推理一次 fold0 val 的三模态 logits（约 929 样本 / 6 个 val 被试）
- leave-one-subject-out：对每个被试 S，用「其余 5 个被试」的 per-class acc 学融合权重 w[k,m]，
  在 S 上测融合 acc。weight 与测试样本 **完全 disjoint subject** → 泛化证据。
- 3 条规则：uniform(基线) / soft_gate(T=1) / hard_gate(每类取最强模态)

回答两件事：
  A. per-class 门控是否在留一被试上有泛化增益（对照 uniform）
  B. 骨架是否被 gate 稳定选为某类"winner"（若 6 折中 ≥4 折且训练样本≥10 → 骨架的有效用法 = 仅这些类提供软证据；
     若从未被选 → 骨架在 val 上确认无门控价值，彻底收官）

规则实现（fc 每类）：
  fused_prob[k] = Σ_m w[k,m] * p_m[k]，w 来自 val(训练被试) 每类各模态 acc：
    soft_gate:  w[k,:] = softmax(acc[k,:]/T)
    hard_gate:  w[k,:] = onehot(argmax acc[k,:])
  训练被试中类样本 <5 → 该回退 uniform。

用法:
    python scripts/eval_fusion_perclass.py \
        --main_ckpt outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth \
        --thermal_ckpt outputs/thermal_34/r2plus1d34_thermal_fold0.pth \
        [--skeleton_ckpt outputs/skeleton_aug/augmented_fold0.pth] [--T 1.0]
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex, ThermalVideoDataset,
                         build_train_index)
from src.model import build_model
from src.motionbert.dstformer import DSTformer
from src.motionbert.action_net import ActionNet
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


def softmax(l):
    p = l - l.max(1, keepdims=True)
    p = np.exp(p)
    return p / p.sum(1, keepdims=True)


def predict(model, loader, device, is_skel=False):
    model.eval()
    logits = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            if is_skel:
                x = x.unsqueeze(1)
            logits.append(model(x).cpu().float().numpy())
    return np.concatenate(logits)


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


def per_class_acc(pred, labels, cls):
    """cls 类的准确率；无样本返回 nan。"""
    sel = labels == cls
    if sel.sum() == 0:
        return float("nan")
    return float((pred[sel] == cls).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--main_ckpt", default="outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth")
    ap.add_argument("--thermal_ckpt", default="outputs/thermal_34/r2plus1d34_thermal_fold0.pth")
    ap.add_argument("--skeleton_ckpt", default="outputs/skeleton_aug/augmented_fold0.pth", help="空串=不参与")
    ap.add_argument("--main_crop", default="bbox_train.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal.json")
    ap.add_argument("--T", type=float, default=1.0, help="soft_gate 温度：越大越接近 uniform，越小越接近 hard")
    ap.add_argument("--min_cls_n", type=int, default=5, help="类样本 <n 时回退 uniform")
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
    subjs = np.array([c.subject for c in va_main])
    n = len(va_main)
    print(f"fold{args.fold} val: {n} 样本 / {len(set(subjs))} 被试", flush=True)

    main_crop = json.loads(Path(args.main_crop).expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path(args.thermal_crop).expanduser().read_text(encoding="utf-8"))

    probs = {}

    ds = DepthIRVideoDataset(va_main, 16, 128, False, main_crop, use_frame_diff=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    m = build_model("r2plus1d34", num_classes=40, in_channels=4, n_segment=16).to(device)
    m.load_state_dict(load_sd(args.main_ckpt))
    probs["main"] = softmax(predict(m, loader, device))
    print(f"main acc = {(probs['main'].argmax(1) == labels).mean():.4f}", flush=True)

    th_clips = [ThermalClipIndex(c.action_id, c.subject, c.sample,
                                 root / "Thermal" / c.depth_dir.parent.parent.name
                                 / c.subject / c.sample) for c in va_main]
    ds_t = ThermalVideoDataset(th_clips, 16, 128, False, th_crop, use_frame_diff=False)
    loader_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    mt = build_model("r2plus1d34", num_classes=40, in_channels=3, n_segment=16).to(device)
    mt.load_state_dict(load_sd(args.thermal_ckpt))
    probs["thermal"] = softmax(predict(mt, loader_t, device))
    print(f"thermal acc = {(probs['thermal'].argmax(1) == labels).mean():.4f}", flush=True)

    mods = ["main", "thermal"]
    if args.skeleton_ckpt:
        sk = Path(args.skeleton_ckpt).expanduser()
        if sk.is_file():
            skel_clips = [type("C", (), {"action_id": c.action_id, "subject": c.subject,
                                         "sample": c.sample,
                                         "pred_dir": root / "Skeleton"
                                         / c.depth_dir.parent.parent.name / c.subject
                                         / c.sample / "predictions"})() for c in va_main]
            ds_s = MotionBertSkeletonDataset(skel_clips, 16, False)
            loader_s = DataLoader(ds_s, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
            ms = load_mb(sk, device)
            probs["skeleton"] = softmax(predict(ms, loader_s, device, is_skel=True))
            mods.append("skeleton")
            print(f"skeleton acc = {(probs['skeleton'].argmax(1) == labels).mean():.4f}", flush=True)
        else:
            print(f"⚠️ 跳过 skeleton（不存在 {sk}）", flush=True)

    mm = len(mods)
    sk_idx = mods.index("skeleton") if "skeleton" in mods else -1
    # ---------------- leave-one-subject-out ----------------
    val_subjs = sorted(set(subjs.tolist()))
    print(f"\n==== per-class 门控融合: 留一被试验证（{mm} 模态: {mods}）====", flush=True)
    rule_acc = {r: [] for r in ("uniform", "soft_gate", "hard_gate")}
    skeleton_win = defaultdict(int)   # 类 -> skeleton 被 hard_gate 选中次数（6 折）
    skeleton_win_n = {}
    for si, s in enumerate(val_subjs):
        test_idx = np.where(subjs == s)[0]
        tr_idx = np.where(subjs != s)[0]
        tr_lab = labels[tr_idx]
        # 每类每模态训练折 acc → 权重
        w = np.full((40, mm), 1.0 / mm, dtype=np.float32)  # uniform 初始
        for c in range(40):
            acc = np.array([per_class_acc(probs[mo][tr_idx].argmax(1), tr_lab, c)
                            for mo in mods], dtype=np.float32)
            if np.isfinite(acc).sum() < 2:
                continue
            acc = np.nan_to_num(acc, nan=1e-6)
            cnt_c = int((tr_lab == c).sum())
            if cnt_c < args.min_cls_n:
                continue
            soft = np.exp((acc - acc.max()) / max(args.T, 1e-6))
            w[c] = soft / soft.sum()
            if sk_idx >= 0 and int(np.argmax(acc)) == sk_idx:
                skeleton_win[c] += 1
                skeleton_win_n[c] = cnt_c
        # 类样本少(<min_cls_n)的类保持 uniform（已跳过）
        for rule in ("soft_gate", "hard_gate"):
            if rule == "hard_gate":
                wh = np.zeros_like(w)
                wh[np.arange(40), w.argmax(1)] = 1.0
                wf = wh
            else:
                wf = w
            fused = np.stack([probs[mo] for mo in mods], -1)  # (N, 40, mm)
            pf = (wf[None, :, :] * fused).sum(-1)             # (N, 40)
            rule_acc[rule].append(float((pf.argmax(1)[test_idx] == labels[test_idx]).mean()))
        # uniform
        pf = np.stack([probs[mo] for mo in mods], -1).mean(-1)
        rule_acc["uniform"].append(float((pf.argmax(1)[test_idx] == labels[test_idx]).mean()))
        print(f"[split {s}] test_n={len(test_idx)} "
              + " | ".join(f"{r}={v[-1]:.4f}" for r, v in rule_acc.items()), flush=True)

    print("\n==== 汇总（平均 acc ± std over 6 个测试被试） ====", flush=True)
    for r, v in rule_acc.items():
        a = np.mean(v)
        print(f"  {r:<10} = {a:.4f} ± {np.std(v):.4f}", flush=True)

    if "skeleton" in mods:
        wins = [(c, n_sel, skeleton_win_n.get(c, 0)) for c, n_sel in skeleton_win.items()
                if n_sel >= 4 and skeleton_win_n.get(c, 0) >= 10]
        print(f"\n==== 骨架被 hard_gate 稳定选中为 winner 的类（≥4/6 折且训练样本≥10）====", flush=True)
        if wins:
            for c, n_sel, cnt in sorted(wins, key=lambda x: -x[1]):
                print(f"  类 {c}: 选中 {n_sel}/6 折, 训练样本 {cnt} → 骨架有效用法 = 仅这些类提供软证据", flush=True)
            print("→ 在 0.73 测试线上做'类级骨架软证据门控'（只加这些类）值得试", flush=True)
        else:
            print("无稳定骨架 winner 类 → 骨架在 per-class 门控下无增益，骨架线彻底收官", flush=True)

    print("\n==== 判读 ====", flush=True)
    g_soft = np.mean(rule_acc["soft_gate"]); g_hard = np.mean(rule_acc["hard_gate"])
    u = np.mean(rule_acc["uniform"])
    print(f"soft_gate - uniform = {g_soft - u:+.4f} | hard_gate - uniform = {g_hard - u:+.4f}",
          flush=True)
    print("≥ +0.01 且非单折偶然 → per-class 门控有泛化增益 → 上 0.73/0.706 测试线 A/B", flush=True)


if __name__ == "__main__":
    main()
