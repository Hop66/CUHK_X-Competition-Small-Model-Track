#!/usr/bin/env python3
"""下载 VideoMAE 权重（走 HF 镜像 / 直链），先验证可获取性再接入。

VideoMAE-S（ViT-Small ~22M）自监督权重主要源：
  1. HF 镜像 transformers 格式（config.json + pytorch_model.bin）
  2. 官方 GitHub（MCG-NJU/VideoMAE）的 Google Drive 权重（服务器可能访问不了）
本脚本先探测哪些源可达，下载成功打印格式。

用法（服务器）:
    python scripts/download_videomae.py --probe          # 只探测可达性
    python scripts/download_videomae.py --which small    # 下载 small（默认）
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
# 候选 repo（transformers 格式；small/base 视 HF 上是否存在）
CANDIDATES = [
    "MCG-NJU/videomae-small",
    "MCG-NJU/videomae-base",
    "MCG-NJU/videomae-base-finetuned-kinetics",
]


def probe(repo: str) -> str:
    """探测镜像上 repo 的可达性，返回 'config'/'bin'/'none'。"""
    base = f"{MIRROR}/{repo}"
    for f in ("config.json", "pytorch_model.bin"):
        url = f"{base}/resolve/main/{f}"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status == 200:
                    return f
        except Exception:
            continue
    return "none"


def download(repo: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{MIRROR}/{repo}"
    for f in ("config.json", "pytorch_model.bin"):
        url = f"{base}/resolve/main/{f}"
        dst = out_dir / f"{repo.split('/')[-1]}_{f}"
        try:
            print(f"  下载 {url} → {dst}", flush=True)
            urllib.request.urlretrieve(url, dst)
            print(f"    OK {dst} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"    FAIL {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只探测可达性不下载")
    ap.add_argument("--which", type=str, default="small", choices=["small", "base"])
    ap.add_argument("--out", type=str, default="weights/videomae")
    args = ap.parse_args()

    print(f"==== 探测 HF 镜像 VideoMAE 权重（{MIRROR}）====", flush=True)
    found = []
    for repo in CANDIDATES:
        status = probe(repo)
        print(f"  {repo}: {status}", flush=True)
        if status != "none":
            found.append((repo, status))

    if args.probe or not found:
        print(f"\n可下载 repo: {[r for r, _ in found]}", flush=True)
        print("若全 none：镜像无此权重，需换源（官方 GitHub / 手动上传）", flush=True)
        return

    # 下载第一个可用 repo（优先含 small 的）
    repo, _ = found[0]
    for r, s in found:
        if "small" in r:
            repo = r
            break
    print(f"\n下载 {repo} → {args.out}", flush=True)
    download(repo, Path(args.out).expanduser())
    print("\n==== 下载完成，检查 pytorch_model.bin 格式后写接入 ====", flush=True)


if __name__ == "__main__":
    main()
