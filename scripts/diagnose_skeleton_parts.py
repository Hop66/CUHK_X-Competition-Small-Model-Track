#!/usr/bin/env python3
"""CUHK-X —— 骨架部位信息量诊断：数据/标签是否撑得起"骨架双专家"（BHaRNet 落地前提）

回答 3 个问题（决策骨架双专家之前必须先量化）：
  ① 训练规模：每类样本数 → 长尾实况、最小类样本量（决定"专属专家"会不会欠拟合）
  ② 数据质量：每类坏 clip 率（缺失/全零/抖动/肩宽异常）→ 部位专家输入信噪比
  ③ 现状可分性：用现成 fold0 checkpoint 在 fold0 val 上做 per-class acc，
     按"手精细 / 臂主导 / 下肢全身 / 静坐"四组聚合 → 判断 17 关节对手类到底有多少信息

判读（BHaRNet 的 N-UCLA 教训：手类少/不可分时 hand 专家会反过来拖累）：
  - 若 手精细组 平均 val_acc 显著低于 下肢组（如 <0.2 而 >0.4）→ 17 关节无指尖+无物体
    = 这些类信息不可恢复 → 双专家（上肢专家）只是在学噪声 → 收益天花板极低。
  - 若个别臂派生类（Take_selfie/Phone_call/Wipe_hands/点头）acc 尚可 → 骨架能捕获区域运动，
    分体（上肢专责手类）可能缓解"全身头平均化拖累"，值得一搏。

用法（服务器，1 个 A100 短 job）:
    python scripts/diagnose_skeleton_parts.py \
        --train_root ~/Multimodal/data/Training/HAR \
        --ckpt outputs/skeleton_aug/augmented_fold0.pth [--norm shoulder|torso] [--clean]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
import torch.nn as nn

from src.skeleton_dataset import (CONF_THRESH, SHOULDER_L, SHOULDER_R,
                                  MotionBertSkeletonDataset, build_skeleton_index)
from src.motionbert.action_net import ActionNet
from src.motionbert.dstformer import DSTformer
from src.split import split_by_subject

# ------------- 40 类部位分组（H3.6M-17：0骨盆 1-6腿 7-10躯干头 11-16臂） -------------
HAND_FINE = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22,
             23, 24, 26, 27, 37, 38, 39]      # 手-嘴 / 手-物 / 精细手指
ARM_DOM   = [3, 5, 15]                          # 臂主导大幅运动（穿衣脱衣/擦窗）
LOCO      = [12, 13, 28, 29, 30, 31, 32, 33, 34, 35, 36]  # 下肢 / 全身大幅度
STATIC    = [25]                                # 静坐（Watch_TV）

GROUP_NAME = {"HAND_FINE": "手精细(25)", "ARM_DOM": "臂主导(3)",
              "LOCO": "下肢全身(11)", "STATIC": "静坐(1)"}
GROUPS = [("HAND_FINE", HAND_FINE), ("ARM_DOM", ARM_DOM), ("LOCO", LOCO), ("STATIC", STATIC)]

CLASS_NAMES = ["Wash_face","Brush_teeth","Comb_hair","Take_off_clothes","Wipe_hands",
               "Put_on_clothes","Drink_water","Eat_food","Take_and_use_tableware",
               "Pour_drinks","Stir_drinks","Peel_fruits","Sweep_floor","Mop_floor",
               "Wipe_bowls","Wipe_windows","Fold_clothes","Tap_keyboard","Write",
               "Phone_call","Check_time","Read","Turn_pages","Listen_music","Use_mobile",
               "Watch_TV","Play_games","Take_selfie","Jog_in_place","Do_squats",
               "Do_jumping_jacks","Do_stretching","Stand_up","Lie_down","Sit_down",
               "Do_lunges","Walk","Take_medicine","Massage","Take_temp"]


def load_clip_kps(pred_dir: Path):
    kps = []
    if not pred_dir.is_dir():
        return np.zeros((0, 17, 3), np.float32)
    files = sorted(pred_dir.glob("*.json"),
                   key=lambda f: int(re.search(r"(\d{8})", f.name).group(1)) if re.search(r"(\d{8})", f.name) else 0)
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for fr in (data if isinstance(data, list) else [data]):
            if isinstance(fr, dict) and "keypoints" in fr:
                kps.append(np.asarray(fr["keypoints"], dtype=np.float32).reshape(17, 3))
    return np.stack(kps, 0) if kps else np.zeros((0, 17, 3), np.float32)


def clip_bad(kps: np.ndarray) -> bool:
    """坏 clip 判定（与 diag_skeleton_quality.py 一致的核心指标）。"""
    T = kps.shape[0]
    if T == 0:
        return True
    allzero = int((np.abs(kps).sum(axis=(1, 2)) == 0).sum())
    vel = np.linalg.norm(np.diff(kps, axis=0), axis=-1) if T >= 2 else np.zeros((0, 17), np.float32)
    if vel.size == 0:
        return True
    sw = np.linalg.norm(kps[:, SHOULDER_L] - kps[:, SHOULDER_R], axis=-1)
    return (T < 8 or allzero > max(1, T * 0.3)
            or (float(sw.std()) > 0.03 and float(sw.mean()) > 0)
            or float(np.percentile(vel, 99)) > 0.15
            or float(np.isfinite(kps).sum()) < kps.size
            or float(sw.mean()) < 0.15 or float(sw.mean()) > 0.5)


def load_actionnet(ckpt, device):
    from src.motionbert.action_net import ActionNet
    backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                         num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = ActionNet(backbone=backbone, dim_rep=512, num_classes=40,
                      dropout_ratio=0.5, version="class", hidden_dim=512, num_joints=17).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--ckpt", default="outputs/skeleton_aug/augmented_fold0.pth",
                    help="现成 fold0 骨架 checkpoint（默认 0.5447 基线）")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--norm", type=str, default="shoulder", choices=["shoulder", "torso"])
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_infer", action="store_true", help="只统计数据级（①/②），跳过 per-class acc")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_skeleton_index(root)
    print(f"训练 clip 总数: {len(clips)} | 被试数: {len(set(c.subject for c in clips))}", flush=True)

    # ① 每类样本数 + ② 每类坏 clip 率
    cnt = defaultdict(int); bad = defaultdict(int); badn = defaultdict(int)
    for c in clips:
        cnt[c.action_id] += 1
        kps = load_clip_kps(c.pred_dir)
        if clip_bad(kps):
            bad[c.action_id] += 1
        else:
            badn[c.action_id] += 1
    classes = sorted(cnt.keys())
    arr = np.array([cnt[c] for c in classes])
    print(f"① 每类样本: min={arr.min()} max={arr.max()} mean={arr.mean():.1f} "
          f"长尾比(max/min)={arr.max()/max(arr.min(),1):.1f}x", flush=True)

    print("\n==== 分部位组汇总（样本数 / 坏clip率） ====", flush=True)
    for gname, gcls in GROUPS:
        n = sum(cnt[c] for c in gcls if c in cnt)
        b = sum(bad[c] for c in gcls if c in cnt)
        print(f"  {GROUP_NAME[gname]:>10}: 样本 {n:4d}  坏clip {b:3d} ({b/max(n,1):5.1%})", flush=True)

    if not args.no_infer:
        folds = split_by_subject(clips, n_folds=args.folds)
        tr_idx, va_idx = folds[args.fold]
        va_clips = [clips[i] for i in va_idx]
        ds = MotionBertSkeletonDataset(va_clips, args.num_frames, False, input3d=True,
                                       clean=args.clean, norm=args.norm)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        model = load_actionnet(Path(args.ckpt).expanduser(), device)
        model.eval()
        per = defaultdict(lambda: [0, 0])   # class -> [correct, total]
        conf = np.zeros((40, 40), dtype=int)
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device).unsqueeze(1)
                pred = model(x).argmax(-1).cpu().numpy()
                y = y.numpy()
                for p, t in zip(pred, y):
                    per[t][0] += int(p == t); per[t][1] += 1
                    conf[t, p] += 1
        print(f"\n③ per-class val_acc（fold{args.fold}, 数据集='{args.ckpt}', "
              f"norm={args.norm} clean={args.clean}）——总体 acc="
              f"{sum(v[0] for v in per.values())/max(sum(v[1] for v in per.values()),1):.4f}", flush=True)

        # 分部位组聚合
        print("\n==== 分部位组 val_acc（关键判读） ====", flush=True)
        for gname, gcls in GROUPS:
            ok = [per[c][0] for c in gcls if per[c][1] > 0]
            tot = [per[c][1] for c in gcls if per[c][1] > 0]
            nnn = [cnt[c] for c in gcls if c in cnt]
            if tot:
                a = f"{sum(ok)/max(sum(tot),1):.3f}"
            else:
                a = "  -"
            print(f"  {GROUP_NAME[gname]:>10}: val_acc={a:>6}  训练样本/类="
                  f"{int(np.mean(nnn)) if nnn else '-':>3}", flush=True)

        # 每类明细（只打样本≥5 的类，按 val_acc 升序）
        print("\n==== 每类明细（样本>=5，按 val_acc 升序） ====", flush=True)
        rows = [(c, cnt[c], bad[c], per[c][0]/max(per[c][1],1))
                for c in classes if cnt[c] >= 5 and per[c][1] > 0]
        for c, n, b, a in sorted(rows, key=lambda r: r[3]):
            g = next((GROUP_NAME[g] for g, gl in GROUPS if c in gl), "?")
            print(f"  {c:2d} {CLASS_NAMES[c]:<22} [{g:>10}] n={n:3d} bad={b:3d} "
                  f"val_acc={a:.3f}", flush=True)

    print("\n==== 判读 ====", flush=True)
    print("HAND_FINE 组 val_acc << LOCO 组 → 17 关节无指尖/无物体，手类信息不可恢复 → "
          "骨架双专家收益天花板极低", flush=True)
    print("HAND_FINE 组个别类 ≥0.4 且高于其它手类 → 存在区域运动可分信号 → 分体(上肢专责手类)值得试", flush=True)


if __name__ == "__main__":
    main()
