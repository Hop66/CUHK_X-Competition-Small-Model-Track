#!/usr/bin/env python3
"""SM dual 推理链路自验（零 LB 定位）

背景：SM dual 提交 0.55223，但训练 fold0 SM val=0.6652。
问题：是推理链路 bug？还是测试集 OOD？—— 不需要交 LB，用训练 fold0 的
      val split（有 ground truth）直接复演「与推理完全一致」的 dual 前向。

做法：
  1. 用推理同一套 build_main_dual + load_state 加载 SM ckpt
  2. 在 fold0 val clips 上：
     - 视频 = DepthIRVideoDataset（crop=bbox_train.json，同推理）
     - 运动 = **从 Skeleton/predictions 重新 extract**（同推理），并打印 energy 分布
       vs motion_cache 对照（还可 --use_cache 直接用 motion_cache 交叉验证特征链路）
  3. 三种融合档各算一次 fold0 val acc：
       static(α=1.0) / learned(α ckpt) / motion-only(α=0.0)
  4. 对照训练记录 0.6652：
       acc_sm  ≈ 0.6652  → 推理链路 OK → 0.55223 是测试 OOD（motion 域/帧率）→ 对症
       acc_sm  ≪ 0.6652  → 推理链路有 bug → 就地修
       acc_static ≈ 0.66、acc_motion 低 → motion 推理侧差

用法：
  python scripts/selftest_main_dual_infer.py \
    --ckpt outputs/main_dual_full/main_SM_full_seed42.pth \
    [--use_cache] [--flip_tta]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DepthIRVideoDataset, build_train_index
from src.skeleton_dataset import build_skeleton_index
from src.split import split_by_subject
from src.skeleton_motion import extract_motion_features, load_skeleton, resample_to_T
from scripts.ensemble_inference import build_main_dual, load_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--crop", default="bbox_train.json")
    ap.add_argument("--motion_cache", default="outputs/motion_cache.pkl")
    ap.add_argument("--use_cache", action="store_true",
                    help="用离线 motion_cache（交叉验证特征提取链路）；默认重提取（同推理）")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.train_root).expanduser()
    clips = build_train_index(root)
    tr_idx, va_idx = split_by_subject(clips, n_folds=3)[args.fold]
    va = [clips[i] for i in va_idx]
    print(f"[self] fold{args.fold} val clips = {len(va)}", flush=True)

    crop_cache = {}
    if args.crop:
        _p = Path(args.crop).expanduser()
        if _p.exists():
            crop_cache = json.loads(_p.read_text(encoding="utf-8"))
            print(f"[self] crop entries = {len(crop_cache)}", flush=True)

    mc = {}
    if args.use_cache:
        import pickle
        with open(Path(args.motion_cache).expanduser(), "rb") as f:
            mc = pickle.load(f)
        print(f"[self] motion_cache clips = {len(mc)}", flush=True)

    # ------------- 运动特征（两种来源） -------------
    face2 = {f"{c.action_id}/{c.subject}/{c.sample}": c.pred_dir
             for c in build_skeleton_index(root)}
    motion_src = np.zeros((len(va), args.num_frames, 29), np.float32)
    energ = np.zeros(len(va), np.float32)
    for i, c in enumerate(va):
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        if args.use_cache:
            a = mc.get(key)
            arr = np.asarray(a, np.float32) if a is not None else np.zeros((0, 29), np.float32)
            if arr.shape[0]:
                motion_src[i] = resample_to_T(arr, args.num_frames)
        else:
            pred_dir = face2.get(key)
            kp, _ = load_skeleton(pred_dir) if pred_dir is not None else (np.zeros((0, 17, 3), np.float32), None)
            motion_src[i] = extract_motion_features(kp, T=args.num_frames)
        energ[i] = float(np.abs(motion_src[i]).sum())
    print(f"[self] motion 来源 = {'motion_cache' if args.use_cache else '重提取(predictions,via build_skeleton_index)'} | "
          f"energy mean={energ.mean():.3f} median={np.median(energ):.3f} "
          f"std={energ.std():.3f} 零={int((energ < 1e-4).sum())}/{len(va)}", flush=True)

    model = build_main_dual(device).eval()
    model.load_state_dict(load_state(args.ckpt, False, device))
    with torch.no_grad():
        a0 = float(torch.sigmoid(model.a).item())
    print(f"[self] learned alpha = {a0:.3f}", flush=True)

    ds = DepthIRVideoDataset(va, args.num_frames, args.size, False, crop_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    y_all = np.zeros(len(va), np.int64)
    for cfg_name, alpha in [("static(α=1.0)", 1.0), ("learned(α=%.3f)" % a0, -1.0),
                            ("motion(α=0.0)", 0.0)]:
        a = alpha if alpha >= 0 else a0
        logit = np.zeros((len(va), 40), np.float32)
        with torch.no_grad():
            s = 0
            for x, lab, _ in loader:
                x = x.to(device)
                mb = torch.from_numpy(motion_src[s:s + len(x)]).to(device)
                pm = torch.softmax(model.motion(mb), -1)
                ps = torch.softmax(model.static(x), -1)
                p = a * ps + (1 - a) * pm
                o = torch.log(p.clamp_min(1e-12))
                if args.flip_tta:
                    psf = torch.softmax(model.static(torch.flip(x, dims=(-1,))), -1)
                    pf = a * psf + (1 - a) * pm
                    o = o + torch.log(pf.clamp_min(1e-12))
                    o = o / 2
                logit[s:s + len(x)] = o.float().cpu().numpy()
                y_all[s:s + len(x)] = lab.numpy()
                s += len(x)
        acc = (logit.argmax(1) == y_all).mean()
        print(f"[self] {cfg_name:22s} fold{args.fold} val acc = {acc:.4f}  "
              f"(对照训练 SM=0.6652 / S-only=0.6598 / motion-only≈?)", flush=True)

    print("\n[自验判读]")
    print("  acc_learned ≈ 0.6652     → 推理链路 OK，0.55223=测试 OOD（motion 域/帧率），对症处理")
    print("  acc_learned ≪ 0.6652     → 推理链路有 bug，就地修")
    print("  acc_static≈0.66, acc_motion 低 → motion 推理侧差 → 推理端用高 α / 重训")


if __name__ == "__main__":
    main()
