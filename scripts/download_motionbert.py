#!/usr/bin/env python3
"""
CUHK-X —— 下载 MotionBERT 权重（H3.6M-17 拓扑匹配）

来源: walterzhu/MotionBERT (HF 镜像)
- lite: checkpoint/pretrain/MB_lite/latest_epoch.bin (61MB, dim_feat=256, H36M/AMASS 预训练)
- ntu : checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin (162MB, dim_feat=512,
        NTU60 x-sub 动作识别微调 97.2% —— **论文官方骨架路线**)

用法:
    python scripts/download_motionbert.py --which lite
    python scripts/download_motionbert.py --which ntu
"""

import argparse
import os
import time
from pathlib import Path

PRESETS = {
    "lite": ("checkpoint/pretrain/MB_lite/latest_epoch.bin", "weights/mb_lite_latest_epoch.bin"),
    "ntu": ("checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin",
            "weights/mb_ntu_xsub_best_epoch.bin"),
}


def download_direct(url: str, out: Path, retries: int = 5):
    """直链下载（hf-mirror resolve URL），绕开 huggingface_hub 的 CDN DNS 问题。"""
    import urllib.request
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, str(out))
            return
        except Exception as e:
            print(f"  直链下载失败（第 {i+1}/{retries} 次）: {type(e).__name__}: {e}", flush=True)
            if i < retries - 1:
                time.sleep(15)
    raise RuntimeError(f"直链下载失败: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="lite", choices=list(PRESETS.keys()),
                    help="lite=MB_lite 预训练(61MB)；ntu=NTU动作权重(162MB,论文官方)")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--file", type=str, default="")
    ap.add_argument("--repo", type=str, default="walterzhu/MotionBERT")
    ap.add_argument("--direct", action="store_true", help="直接用 hf-mirror 直链（不用 huggingface_hub）")
    args = ap.parse_args()

    if args.file:
        file, out = args.file, args.out or "weights/ckpt.bin"
    else:
        file, out = PRESETS[args.which]
        if args.out:
            out = args.out

    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

    if args.direct:
        # 直链：https://hf-mirror.com/{repo}/resolve/main/{file}
        url = f"{mirror}/{args.repo}/resolve/main/{file}"
        print(f"downloading (直链) {url} ...", flush=True)
        download_direct(url, out)
        print(f"saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    # 优先 huggingface_hub（走镜像），失败后 fallback 直链
    print(f"downloading {args.repo} :: {file} ...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        path = None
        for attempt in range(3):
            try:
                path = hf_hub_download(repo_id=args.repo, filename=file,
                                       local_dir=str(out.parent))
                break
            except Exception as e:
                print(f"  hub 下载失败（第 {attempt+1}/3 次）: {type(e).__name__}: {e}", flush=True)
                if attempt < 2:
                    time.sleep(10)
        if path is not None and Path(path) != out:
            out.write_bytes(Path(path).read_bytes())
    except Exception as e:
        print(f"huggingface_hub 失败: {e}，改用直链", flush=True)
        url = f"{mirror}/{args.repo}/resolve/main/{file}"
        download_direct(url, out)
    print(f"saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
