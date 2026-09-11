#!/usr/bin/env python3
"""双塔同步难对重权 fold0 验证（P0 修复版, 2026-09-11）。

⚠️ 协议声明（修复 double-softmax + 撤销绝对增益声称）:
  - 原版 softmax_logits() 先 sf 一次, eval_fusion 又 sf 一次 → double-softmax。
  - 实测 double-softmax 对 argmax 影响 ~0.1pt（单调变换），但不符合提交协议。
  - 正确协议: 所有 logits **softmax 恰好一次**再融合（与 prob_avg 一致）。
  - 本文件仅作 fold 诊断; 因 OOF 模型是 baseline_aug2 (≠锚 s42 全数据),
    任何数字**不构成"锚有效 AB"**; 锚级判断只能跑 full(seed42) + int5 + flip 同协议。

用法: python scripts/hardpair_dual_fold.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalClipIndex, ThermalVideoDataset,
                         build_thermal_index, build_train_index)
from src.model import build_model
from src.split import split_by_subject

H11 = {(0, 1), (6, 37), (7, 37), (8, 9), (8, 10), (8, 15),
       (8, 18), (11, 14), (17, 18), (18, 20), (38, 39)}
OOF = Path("outputs/oof")


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def softmax_logits(d):
    return {k: sf(np.asarray(v, dtype=np.float64)) for k, v in d.items()}


@torch.no_grad()
def infer(model, loader, device):
    out = np.zeros((len(loader.dataset), 40), np.float32)
    s = 0
    for x, _, _ in loader:
        b = x.shape[0]
        o = model(x.to(device))
        # 与基线 OOF 同口径(--flip): o + model(flip(x))，水平翻转 TTA
        o = o + model(torch.flip(x, dims=(-1,)))
        out[s:s + b] = o.float().cpu().numpy()
        s += b
    return out


def eval_fusion(pm, pt, keys, labels):
    """融合评估: pm/pt 已是 softmax 概率(恰一次), sum 后 argmax (与 prob_avg 同协议)。
    ⚠️ 修复: 原版这里又 sf() → double-softmax; 现在要求调用方传入 exactly-once softmax。"""
    tot = corr = 0
    hp_e = nh_e = 0
    for k, g in zip(keys, labels):
        fu = pm[k] + pt[k]
        p = int(fu.argmax())
        tot += 1
        corr += (p == g)
        if any(g in q for q in H11):
            hp_e += (p != g)
        else:
            nh_e += (p != g)
    return corr / tot, corr, hp_e, nh_e


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("~/Multimodal/data/Training/HAR").expanduser()
    clips = build_train_index(root)
    _, va_idx = split_by_subject(clips, n_folds=3)[0]
    va_clips = [clips[i] for i in va_idx]
    print(f"fold0 val {len(va_clips)} clips", flush=True)

    # 基线 OOF fold0
    bm = softmax_logits(pickle.load(open(OOF / "main_oof.pkl", "rb"))[0])
    bt = softmax_logits(pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[0])
    # 统一评估域 = main∩th 共同 keys（避免 thermal/main 未对齐 clip 造成 KeyError）
    ckeys = [k for k in bm if k in bt]
    labels = [int(k.split("/")[0]) for k in ckeys]
    print(f"eval 域(交集) = {len(ckeys)} clips", flush=True)
    acc, c, he, ne = eval_fusion(bm, bt, ckeys, labels)
    print(f"[R 基线双塔] acc={acc:.4f} ({c}/{len(ckeys)}) 难对区错={he} 非难对错={ne}", flush=True)

    # 新难对双塔 fold0
    import json
    main_crop = json.loads(Path("bbox_train.json").expanduser().read_text(encoding="utf-8"))
    th_crop = json.loads(Path("bbox_thermal_train.json").expanduser().read_text(encoding="utf-8"))

    m_ck = Path("outputs/dual_hp_f0/main/r2plus1d34_depthir_fold0.pth")
    t_ck = Path("outputs/dual_hp_f0/th/r2plus1d34_thermal_fold0.pth")
    if not (m_ck.exists() and t_ck.exists()):
        print("❌ 缺难对双塔 ckpt", m_ck.exists(), t_ck.exists())
        return

    m_model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                          n_segment=16).to(device).eval()
    m_model.load_state_dict(torch.load(m_ck, map_location=device))
    mds = DepthIRVideoDataset(va_clips, 16, 128, False, main_crop)
    ml = DataLoader(mds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    pm = dict(zip([f"{c.action_id}/{c.subject}/{c.sample}" for c in va_clips],
                  infer(m_model, ml, device)))

    t_model = build_model("r2plus1d34", num_classes=40, in_channels=3,
                          n_segment=16).to(device).eval()
    t_model.load_state_dict(torch.load(t_ck, map_location=device))
    # th 塔用 ThermalClipIndex（build_thermal_index）+ 相同 split_by_subject
    th_clips = build_thermal_index(root)
    _, th_va_idx = split_by_subject(th_clips, n_folds=3)[0]
    th_va = [th_clips[i] for i in th_va_idx]
    print(f"th fold0 val {len(th_va)} clips", flush=True)
    tds = ThermalVideoDataset(th_va, 16, 128, False, th_crop)
    tl = DataLoader(tds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    pt = dict(zip([f"{c.action_id}/{c.subject}/{c.sample}" for c in th_va],
                  infer(t_model, tl, device)))

    hm = softmax_logits(pm)
    ht = softmax_logits(pt)
    # 同一交集评估域
    ckeys2 = [k for k in ckeys if k in hm and k in ht]
    labels2 = [int(k.split("/")[0]) for k in ckeys2]
    acc2, c2, he2, ne2 = eval_fusion(hm, ht, ckeys2, labels2)
    print(f"[A 难对双塔] acc={acc2:.4f} ({c2}/{len(ckeys2)}) 难对区错={he2} 非难对错={ne2}", flush=True)
    print(f"\nΔacc={100*(acc2-acc):+.2f}pt (域 {len(ckeys2)} vs {len(ckeys)})  难对区错 {he}→{he2} (Δ{he2-he:+.0f}) "
          f"非难对错 {ne}→{ne2} (Δ{ne2-ne:+.0f})", flush=True)

    # 也看单模型 acc（健康检查, 难对模型不应大幅掉总acc）
    bm2 = softmax_logits(bm); bt2 = softmax_logits(bt)
    def single_acc(p):
        corr = 0
        for k, g in zip(ckeys2, labels2):
            if k in p:
                corr += (int(p[k].argmax()) == g)
        return corr
    n2 = len(ckeys2)
    print(f"单模型(域{n2}): main base={single_acc(bm2)/n2:.4f} hp={single_acc(hm)/n2:.4f} | "
          f"th base={single_acc(bt2)/n2:.4f} hp={single_acc(ht)/n2:.4f}", flush=True)


if __name__ == "__main__":
    main()
