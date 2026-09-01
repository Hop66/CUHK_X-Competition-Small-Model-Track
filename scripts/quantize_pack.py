#!/usr/bin/env python3
"""
CUHK-X —— int8 量化打包（100MB 约束）

把训练好的 fp32 checkpoint 量化为 int8，报告总大小，检查是否 <100MB。

用法:
    # 单模态打包（多折可 --average 合并成一份）
    python scripts/quantize_pack.py --name main --checkpoints \
        outputs/main_fd/r2plus1d_depthir_fold0.pth \
        outputs/main_fd/r2plus1d_depthir_fold1.pth \
        outputs/main_fd/r2plus1d_depthir_fold2.pth --average

    # 三模态一起打包（检查总大小）
    python scripts/quantize_pack.py \
        --name main     --checkpoints outputs/main_fd/*.pth --average \
        --name thermal  --checkpoints outputs/thermal_*/*.pth --average \
        --name skeleton --checkpoints outputs/skeleton_mb/*.pth --average \
        --out_dir outputs/pack --max_mb 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.quantize import (average_state_dicts, dequantize_state_dict,
                          estimate_size_bytes, quantize_state_dict)


def pack_one(name, checkpoints, out_dir, average, bits):
    paths = [Path(c).expanduser() for c in checkpoints]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {p}")
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if average and len(paths) > 1:
        print(f"[{name}] 平均 {len(paths)} 折权重（SWA）...", flush=True)
        sd = average_state_dicts(paths)
        q_sd, scales = quantize_state_dict(sd, bits=bits)
        size_mb = estimate_size_bytes(q_sd, scales) / 1e6
        out_path = out / f"{name}_int{bits}.pth"
        torch.save({"q_sd": q_sd, "scales": scales, "bits": bits}, out_path)
        print(f"[{name}] 量化后 {size_mb:.1f} MB -> {out_path}", flush=True)
        return [size_mb]

    # 不平均：每折单独量化打包（推理时 logits 平均，避免 SWA 权重平均导致预测塌缩）
    sizes = []
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if "model" in sd:  # MotionBERT ActionNet checkpoint
            sd = sd["model"]
        q_sd, scales = quantize_state_dict(sd, bits=bits)
        size_mb = estimate_size_bytes(q_sd, scales) / 1e6
        out_path = out / f"{name}_fold{i}_int{bits}.pth"
        torch.save({"q_sd": q_sd, "scales": scales, "bits": bits}, out_path)
        sizes.append(size_mb)
        print(f"[{name}] fold{i} 单独量化 {size_mb:.1f} MB -> {out_path}", flush=True)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", default=[], help="模态名（与 --checkpoints 配对，可多次）")
    ap.add_argument("--checkpoints", action="append", nargs="+", default=[], help="checkpoint 列表（可多次）")
    ap.add_argument("--average", action="store_true", help="多折权重平均成一份（推荐，省体积+泛化）")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--out_dir", type=str, default="outputs/pack")
    ap.add_argument("--max_mb", type=float, default=100.0)
    args = ap.parse_args()

    if len(args.name) != len(args.checkpoints):
        print("错误: --name 与 --checkpoints 数量必须一致（每个模态一组）")
        sys.exit(1)

    total = 0.0
    for name, ckpts in zip(args.name, args.checkpoints):
        sizes = pack_one(name, ckpts, args.out_dir, args.average, args.bits)
        total += sum(sizes)

    print(f"\n==== 总大小: {total:.1f} MB （上限 {args.max_mb} MB）====")
    if total > args.max_mb:
        print(f"❌ 超限 {total - args.max_mb:.1f} MB。建议：")
        print("   1) 用 --average 把多折合并成一份")
        print("   2) 减少集成模态数（去掉最弱的）")
        print("   3) 降低 --bits（注意 int8 是 torch 原生下限，更低需 bit-packing）")
    else:
        print(f"✅ 合规，剩余 {args.max_mb - total:.1f} MB")


if __name__ == "__main__":
    main()
