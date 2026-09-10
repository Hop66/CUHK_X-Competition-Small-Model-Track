#!/usr/bin/env python3
"""OOF(折外)概率提取 —— 供多模态概率 Stacking 元融合使用。

同协议: 每个流拿 subject-3 折 fold 模型, 各自在【未训练过的】val 折上出 logits
(即真 OOF)。产出:
  out/oof/<stream>_oof.pkl = { "fold": { "<action>/<subj>/<sample>": logits[40] } }
支持 main / thermal。IMU/Radar/骨架后续核加。
用法:
  python scripts/extract_oof_logits.py --main --folds 0 1 2 \
      --fold_ckpts outputs/main_baseline/baseline_aug2_fold0.pth ... \
      --out outputs/oof/main_oof.pkl
  python scripts/extract_oof_logits.py --thermal --folds 0 1 2 \
      --fold_ckpts outputs/ab_th_fold0/base/r2plus1d34_thermal_fold0.pth ... \
      --out outputs/oof/thermal_oof.pkl
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (DepthIRVideoDataset, ThermalVideoDataset, ClipIndex,
                         build_test_index, build_train_index, build_thermal_index)
from src.split import split_by_subject
from src.model import build_model
from scripts.ensemble_inference import load_state, load_crop


def main():
    ap = argparse.ArgumentParser()
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument("--main", action="store_true")
    gp.add_argument("--thermal", action="store_true")
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--test_root", default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--fold_ckpts", nargs="+", default=[], help="fold 模型(f0 f1 f2); test 模式给任一推理模型")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop", default="bbox_train.json")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--save_feats", type=str, default="",
                    help="同时保存 encoder 512d 特征 {key: feats[512]}（eg outputs/oof/main_src_feats.pkl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["oof", "test"], default="oof")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import json

    def build_stream():
        if args.thermal:
            model = build_model("r2plus1d34", num_classes=40, in_channels=3,
                                n_segment=args.num_frames).to(device).eval()
        else:
            model = build_model("r2plus1d34", num_classes=40, in_channels=4,
                                n_segment=args.num_frames).to(device).eval()
        return model

    if args.mode == "oof":
        root = Path(args.train_root).expanduser()
        crop_cache = json.loads(Path(args.crop).read_text(encoding="utf-8"))
        if args.thermal:
            clips = build_thermal_index(root)
        else:
            clips = build_train_index(root)
        model = build_stream()
        folds = split_by_subject(clips, n_folds=3)
        assert len(args.fold_ckpts) == 3, "oof 模式需要 3 个 fold 模型"
        out = {}
        out_feats = {f: {} for f in range(3)} if args.save_feats else None
        for f in range(3):
            va = [clips[i] for i in folds[f][1]]
            if args.thermal:
                ds = ThermalVideoDataset(va, args.num_frames, args.size, False,
                                         crop_cache, use_frame_diff=False)
            else:
                ds = DepthIRVideoDataset(va, args.num_frames, args.size, False,
                                         crop_cache, use_frame_diff=False)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            model.load_state_dict(load_state(args.fold_ckpts[f], False, device))
            model.eval()
            logits = np.zeros((len(va), 40), np.float32)
            feats = np.zeros((len(va), 512), np.float32) if args.save_feats else None
            s = 0
            hook = None
            if args.save_feats:
                # 抓 encoder 输出 (fc.Identity 前的 512d)
                def _make_hook():
                    buf = {}
                    def _h(m, i, o):
                        buf["f"] = o.detach().float().cpu().numpy()
                    return buf, _h
                hbuf, hfn = _make_hook()
                hook = model.encoder.register_forward_hook(hfn)
            with torch.no_grad():
                for b in loader:
                    x = b[0].to(device)
                    o = model(x)
                    if args.flip:
                        o = o + model(torch.flip(x, dims=(-1,)))
                    logits[s:s + len(o)] = o.float().cpu().numpy()
                    if args.save_feats:
                        feats[s:s + len(o)] = hbuf["f"][:len(o)]
                    s += len(o)
            if hook is not None:
                hook.remove()
            keys = [f"{c.action_id}/{c.subject}/{c.sample}" for c in va]
            out[f] = dict(zip(keys, logits.astype(np.float32)))
            if args.save_feats:
                out_feats[f] = dict(zip(keys, feats.astype(np.float32)))
                print(f"  [ooof] fold{f} feats shape {feats.shape}", flush=True)
            print(f"[oof] fold{f} val={len(va)} clips done", flush=True)
    else:  # test
        test_root = Path(args.test_root).expanduser()
        raw = build_test_index(test_root)
        crop_p = "bbox_test.json" if args.main else "bbox_thermal_test.json"
        cmap = load_crop(crop_p, raw)
        model = build_stream()
        assert len(args.fold_ckpts) == 1, "test 模式给 1 份推理模型"
        if args.thermal:
            clips = [type("C", (), {"subject": cid, "sample": cid, "action_id": -1,
                                    "thermal_dir": ddir.parent / "Thermal"})()
                     for (cid, ddir, _) in raw]
            ds = ThermalVideoDataset(clips, args.num_frames, args.size, False,
                                     cmap, use_frame_diff=False)
        else:
            clips = [ClipIndex(-1, cid, cid, ddir, idir) for (cid, ddir, idir) in raw]
            ds = DepthIRVideoDataset(clips, args.num_frames, args.size, False,
                                     cmap, use_frame_diff=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        model.load_state_dict(load_state(args.fold_ckpts[0], False, device))
        model.eval()
        logits = np.zeros((len(raw), 40), np.float32)
        feats = np.zeros((len(raw), 512), np.float32) if args.save_feats else None
        hbuf = {"f": np.zeros((args.batch_size, 512), np.float32)}
        def _h(m, i, o):
            hbuf["f"] = o.detach().float().cpu().numpy()
        hook = model.encoder.register_forward_hook(_h) if args.save_feats else None
        s = 0
        with torch.no_grad():
            for b in loader:
                x = b[0].to(device)
                o = model(x)
                if args.flip:
                    o = o + model(torch.flip(x, dims=(-1,)))
                logits[s:s + len(o)] = o.float().cpu().numpy()
                if args.save_feats:
                    feats[s:s + len(o)] = hbuf["f"][:len(o)]
                s += len(o)
        if hook is not None:
            hook.remove()
        out = {cid: logits[i].astype(np.float32) for i, (cid, _, _) in enumerate(raw)}
        if args.save_feats:
            out_feats = {cid: feats[i].astype(np.float32) for i, (cid, _, _) in enumerate(raw)}
        print(f"[test] {len(out)} clips logits done", flush=True)

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"[save] {args.out} (mode={args.mode})", flush=True)

    if args.save_feats:
        # 单独存 feats 文件
        fpath = Path(args.save_feats); fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "wb") as fh:
            pickle.dump(out_feats, fh, protocol=4)
        print(f"[save feats] {fpath} (mode={args.mode})", flush=True)


if __name__ == "__main__":
    main()

