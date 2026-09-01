#!/usr/bin/env python3
"""检查 VideoMAE-S 权重文件结构（打印 key + 形状，供接入 key 映射）。

用法（权重传到服务器后）:
    python scripts/inspect_videomae_ckpt.py --ckpt weights/videomae_s_k400_pretrain.pth
"""

import argparse
from pathlib import Path

import torch


def _find_state_dict(obj, depth=0):
    """递归查找"所有值都是 tensor"的字典（即 state_dict），兼容各种包装。

    VideoMAE 官方权重可能是 {'model': ...} / {'module': ...} / {'state_dict': ...} /
    直接 state_dict 等结构。
    """
    if isinstance(obj, dict):
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for k in ("state_dict", "model", "module", "backbone", "encoder"):
            if k in obj and isinstance(obj[k], dict):
                inner = _find_state_dict(obj[k], depth + 1)
                if inner is not None:
                    return inner
        for v in obj.values():
            if isinstance(v, dict):
                inner = _find_state_dict(v, depth + 1)
                if inner is not None:
                    return inner
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="weights/videomae_s_k400_pretrain.pth")
    ap.add_argument("--max_keys", type=int, default=60)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        print(f"❌ 权重不存在: {ckpt}")
        return
    raw = torch.load(ckpt, map_location="cpu")
    print(f"顶层类型: {type(raw).__name__}", flush=True)
    if isinstance(raw, dict):
        print(f"顶层 key: {list(raw.keys())[:10]}", flush=True)
    sd = _find_state_dict(raw)
    if sd is None:
        print("❌ 未找到 tensor state_dict（检查权重结构）", flush=True)
        return
    print(f"state_dict 参数数: {len(sd)}", flush=True)
    keys = list(sd.keys())
    for k in keys[:args.max_keys]:
        print(f"  {k}: {tuple(sd[k].shape)}", flush=True)
    if len(keys) > args.max_keys:
        print(f"  ... 共 {len(keys)} 个 key", flush=True)
    # 探测关键结构
    for probe in ["encoder", "cls_token", "pos_embed", "blocks", "norm", "decoder"]:
        hit = [k for k in keys if probe in k]
        print(f"  含 '{probe}': {len(hit)} 个（例: {hit[:2]}）", flush=True)
    # 推断 embed_dim（cls_token 或 pos_embed）
    for probe in ["cls_token", "pos_embed"]:
        hit = [k for k in keys if probe in k]
        if hit:
            print(f"  {probe} shape: {tuple(sd[hit[0]].shape)}（推断 embed_dim）", flush=True)


if __name__ == "__main__":
    main()
