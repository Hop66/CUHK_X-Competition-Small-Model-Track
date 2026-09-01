#!/usr/bin/env python3
"""CUHK-X —— 伪标签融合：原始 0.73（int5）+ selftrain_full（fp32）prob 平均 → submission

用法:
    python scripts/fuse_selftrain.py \
        --main_orig outputs/pack/main_s42_fold0_int5.pth \
        --thermal_orig outputs/pack/thermal_s42_fold0_int5.pth \
        --main_st outputs/selftrain_full/main_selftrain_full.pth \
        --thermal_st outputs/selftrain_full/thermal_selftrain_full.pth \
        --w_orig 1.0 --w_st 1.0 \
        --flip_tta --output submission_fusion.csv
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from scripts.ensemble_inference import infer_main, infer_thermal


def _softmax(l):
    p = l - l.max(axis=1, keepdims=True)
    p = np.exp(p)
    return p / p.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_orig", nargs="+",
                    default=["outputs/pack/main_s42_fold0_int5.pth"])
    ap.add_argument("--thermal_orig", nargs="+",
                    default=["outputs/pack/thermal_s42_fold0_int5.pth"])
    ap.add_argument("--main_st", nargs="+",
                    default=["outputs/selftrain_full/main_selftrain_full.pth"])
    ap.add_argument("--thermal_st", nargs="+",
                    default=["outputs/selftrain_full/thermal_selftrain_full.pth"])
    ap.add_argument("--w_orig", type=float, default=1.0, help="原始 0.73 权重")
    ap.add_argument("--w_st", type=float, default=1.0, help="selftrain 权重")
    ap.add_argument("--main_backbone", default="r2plus1d34")
    ap.add_argument("--thermal_backbone", default="r2plus1d34")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--test_root", default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--test_csv", default="~/Multimodal/data/Testing/test_file/test.csv",
                    help="test.csv 路径（提交文件，含 path 列；与 ensemble_inference 对齐）")
    ap.add_argument("--main_crop", default="bbox_test.json")
    ap.add_argument("--thermal_crop", default="bbox_thermal_test.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default="submission_fusion.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    base = dict(main_backbone=args.main_backbone, thermal_backbone=args.thermal_backbone,
                num_frames=args.num_frames, size=args.size, test_root=args.test_root,
                frame_diff=False, time_tta=1, flip_tta=args.flip_tta,
                batch_size=args.batch_size, workers=args.workers)
    o = argparse.Namespace(**base, main=args.main_orig, thermal=args.thermal_orig,
                           quantize=True,  # 原始 int5 打包 → 反量化
                           main_crop=args.main_crop, thermal_crop=args.thermal_crop)
    s = argparse.Namespace(**base, main=args.main_st, thermal=args.thermal_st,
                           quantize=False,  # selftrain fp32
                           main_crop=args.main_crop, thermal_crop=args.thermal_crop)

    print("==== 原始 0.73（int5 反量化）推理 ====", flush=True)
    lo_m = infer_main(o, device)
    lo_t = infer_thermal(o, device)
    print(f"  orig: main/thermal logits {lo_m.shape} {lo_t.shape}", flush=True)
    print("==== selftrain_full（fp32）推理 ====", flush=True)
    ls_m = infer_main(s, device)
    ls_t = infer_thermal(s, device)
    print(f"  selftrain: main/thermal logits {ls_m.shape} {ls_t.shape}", flush=True)

    # prob 平均：orig 内部 main+thermal 平均，selftrain 内部 main+thermal 平均，再按权重融合
    p_orig = (_softmax(lo_m) + _softmax(lo_t)) / 2.0
    p_st = (_softmax(ls_m) + _softmax(ls_t)) / 2.0
    fused = args.w_orig * p_orig + args.w_st * p_st
    preds = fused.argmax(axis=1).astype(int)

    # 复用 ensemble_inference 的 clip_id 映射逻辑
    test_root = Path(args.test_root).expanduser()
    clip_ids = [d.name for d in sorted(test_root.iterdir())
                if d.is_dir() and d.name.startswith("SM_test_")]
    test_df = pd.read_csv(Path(args.test_csv).expanduser())
    assert len(test_df) == len(clip_ids), f"{len(test_df)} vs {len(clip_ids)}"
    pred_map = dict(zip(clip_ids, preds))

    def _clip_of_path(p):
        m = re.search(r"(SM_test_\d+)", str(p))
        return m.group(1) if m else str(p)

    order = test_df["path"].astype(str).map(_clip_of_path)
    test_df["prediction"] = [pred_map.get(k, 0) for k in order]
    out = Path(args.output).expanduser()
    test_df[["path", "prediction"]].to_csv(out, index=False)
    n_zero = int((test_df["prediction"] == 0).sum())
    print(f"submission saved: {out} ({len(test_df)} rows) "
          f"类0={n_zero} 去重类={test_df['prediction'].nunique()}", flush=True)
    assert test_df["prediction"].between(0, 39).all() and len(test_df) == 405
    print("判读: 提交此 CSV，LB > 0.73 → 伪标签有效（同分布数据大杠杆）", flush=True)


if __name__ == "__main__":
    main()
