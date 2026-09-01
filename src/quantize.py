"""权重量化工具（per-output-channel，反量化推理）。

100MB 约束下的打包方案：
  - bits>=8: 权重 per-output-channel 量化到 torch.int8（1 byte/参数）
  - bits<8 : 权重量化后 **bit-packing**（int5=5bit/int6=6bit，比 int8 更小）
  - bias / BN / 小张量保持 fp32（数量少，占比小）
  - 推理时反量化回 fp32 计算

用法见 scripts/quantize_pack.py。
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# bit-packing（int5/int6 位压缩）
# ---------------------------------------------------------------------------
def _pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """把 uint 数组（0..2^bits-1）向量化打包成 uint8 字节流。"""
    values = values.astype(np.uint32).reshape(-1)
    bit_matrix = ((values[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    flat = bit_matrix.reshape(-1)
    pad = (-len(flat)) % 8
    if pad:
        flat = np.pad(flat, (0, pad))
    return np.packbits(flat)


def _unpack_bits(packed: np.ndarray, bits: int, n_values: int) -> np.ndarray:
    """从 uint8 字节流 unpack 回 uint 数组。"""
    flat = np.unpackbits(packed.astype(np.uint8))[:n_values * bits]
    bit_matrix = flat.reshape(n_values, bits)
    return (bit_matrix * (1 << np.arange(bits, dtype=np.uint32))).sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# 量化 / 反量化
# ---------------------------------------------------------------------------
def quantize_state_dict(sd, bits: int = 8):
    """per-output-channel 量化 state_dict。

    返回 (q_sd, scales)：
      - bits>=8: q_sd[k]=torch.int8, scales[k]=torch.Tensor [out_ch]
      - bits<8 : q_sd[k]=torch.uint8(打包字节), scales[k]=dict(scale/shape/bits/qmin)
    """
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q_sd, scales = {}, {}
    for k, v in sd.items():
        # 只量化 float32 的大权重张量（Conv/Linear weight）
        if v.dtype not in (torch.float32, torch.float16) or v.ndim < 2 or v.numel() < 4096:
            q_sd[k] = v.to(torch.float32) if v.dtype == torch.float16 else v
            continue
        v = v.float()
        flat = v.reshape(v.shape[0], -1)
        amax = flat.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
        scale = amax / qmax
        q = (flat / scale).round().clamp(qmin, qmax)
        if bits >= 8:
            q_sd[k] = q.reshape(v.shape).to(torch.int8)
            scales[k] = scale.squeeze(1).to(torch.float32)
        else:
            q_unsigned = (q - qmin).cpu().numpy().astype(np.uint32).reshape(-1)
            packed = _pack_bits(q_unsigned, bits)
            q_sd[k] = torch.from_numpy(packed).to(torch.uint8)
            scales[k] = {"scale": scale.squeeze(1).to(torch.float32),
                         "shape": tuple(v.shape), "bits": bits, "qmin": qmin}
    return q_sd, scales


def dequantize_state_dict(q_sd, scales, device=None):
    """把量化后的 state_dict 反量化回 fp32（用于推理）。"""
    sd = {}
    for k, v in q_sd.items():
        if k not in scales:
            sd[k] = v
            continue
        meta = scales[k]
        if isinstance(meta, dict):
            # bit-packed
            shape = meta["shape"]
            n_values = int(np.prod(shape))
            q_unsigned = _unpack_bits(v.cpu().numpy(), meta["bits"], n_values)
            q = torch.from_numpy((q_unsigned + meta["qmin"]).astype(np.float32))
            q = q.reshape(shape)
            s = meta["scale"].view([-1] + [1] * (len(shape) - 1))
            if device is not None:
                q, s = q.to(device), s.to(device)
            sd[k] = q * s
        else:
            # int8
            s = meta.view([-1] + [1] * (v.dim() - 1))
            if device is not None:
                s = s.to(device)
            sd[k] = v.float().to(s.device) * s
    return sd


def load_quantized(path, device=None):
    """加载量化包（torch.save 的 {q_sd, scales}），返回 fp32 state_dict。"""
    pkg = torch.load(path, map_location=device)
    return dequantize_state_dict(pkg["q_sd"], pkg.get("scales", {}), device)


def estimate_size_bytes(q_sd, scales) -> int:
    """估算量化后的字节数（近似 torch.save 后的大小）。"""
    total = 0
    for v in q_sd.values():
        n = v.numel()
        total += n * (1 if v.dtype in (torch.int8, torch.uint8) else 4)
    for s in scales.values():
        if isinstance(s, dict):
            total += s["scale"].numel() * 4
        else:
            total += s.numel() * 4
    return total


def average_state_dicts(paths):
    """多折 checkpoint 权重平均（SWA）。先解包 model/state_dict 包装。"""
    sds = []
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sds.append(sd)
    keys = sds[0].keys()
    avg = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in sds])
        avg[k] = stacked.mean(dim=0)
    return avg
