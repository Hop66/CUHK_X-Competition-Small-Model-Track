#!/usr/bin/env python3
"""
CUHK-X —— 预训练权重一次性下载（全部走镜像/本地缓存）

- HuggingFace 下载走镜像 hf-mirror.com（HF_ENDPOINT）
- torchvision 权重（R2Plus1D-18 Kinetics / ResNet18 ImageNet）缓存到 ~/.cache/torch
- YOLO11n 权重：优先 ultralytics 默认源，失败则走 HF 镜像并保存为本地 yolo11n.pt

用法（登录节点即可，一次性）:
    python scripts/download_weights.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 1) HuggingFace 走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_torchvision():
    import torchvision.models as tvm
    print("[torchvision] 下载 ResNet18 (ImageNet) ...", flush=True)
    try:
        tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        print("  resnet18 ok", flush=True)
    except Exception as e:
        print(f"  resnet18 FAILED: {e}", flush=True)
    print("[torchvision] 下载 R2Plus1D-18 (Kinetics400) ...", flush=True)
    try:
        tvm.video.r2plus1d_18(weights=tvm.video.R2Plus1D_18_Weights.KINETICS400_V1)
        print("  r2plus1d_18 ok", flush=True)
    except Exception as e:
        print(f"  r2plus1d_18 FAILED: {e}", flush=True)


def download_yolo():
    print("[ultralytics] 下载 YOLO11n（huggingface_hub 走镜像）...", flush=True)
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from pathlib import Path as P
    local = P("yolo11n.pt")
    if local.is_file():
        print("  yolo11n.pt 已存在", flush=True)
        return
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="Ultralytics/YOLO11", filename="yolo11n.pt",
                        local_dir=".", local_dir_use_symlinks=False)
        print(f"  yolo11n.pt ok -> {local}", flush=True)
    except Exception as e:
        print(f"  yolo11n.pt FAILED: {e}", flush=True)


def download_ig65m():
    print("[ig65m] R2+1D-34 权重 (IG-65M+Kinetics, ~250MB)：", flush=True)
    url = ("https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/"
           "r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth")
    out = Path("ig65m_r2plus1d34.pth")
    if out.is_file():
        print("  已存在", flush=True)
        return
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(out))
        print(f"  ok -> {out}", flush=True)
    except Exception as e:
        print(f"  GitHub 下载失败: {e}", flush=True)
        print(f"  请在本地（能上 GitHub 的机器）下载后 scp 到服务器：", flush=True)
        print(f"    {url}", flush=True)
        print(f"    scp ig65m_r2plus1d34.pth <server>:~/Multimodal/", flush=True)


if __name__ == "__main__":
    print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"), flush=True)
    download_torchvision()
    download_yolo()
    download_ig65m()
    print("\nDONE. 权重已缓存，训练时不再联网。", flush=True)
