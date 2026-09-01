#!/usr/bin/env python3
"""CUHK-X —— 骨架数据全量质量诊断（先量化噪声，再针对性清洗）

背景：骨架 0.45-0.54 一直被当"弱模态"放弃，但从未诊断过它为什么弱。
本地样例（58帧）质量极高（肩宽 std ±1.4%）→ 噪声不是均匀的，是部分坏 clip。
本脚本全量统计：缺失率、抖动、肩宽/骨骼稳定性、越界、全零帧，定位坏帧分布。

用法:
    python scripts/diag_skeleton_quality.py \
        --train_root ~/Multimodal/data/Training/HAR [--limit N] [--sample_dir data/Training/data]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.skeleton_dataset import build_skeleton_index, frame_num_of

# H3.6M-17 关节
PELVIS, NECK, HEAD, SHL, SHR = 0, 8, 9, 11, 14


def stat_clip(kps: np.ndarray):
    """kps [T,17,3] → 该 clip 的质量指标 dict。"""
    T = kps.shape[0]
    if T == 0:
        return {"frames": 0, "allzero": 1}
    allzero = int((np.abs(kps).sum(axis=(1, 2)) == 0).sum())          # 全零帧
    finite = int(np.isfinite(kps).sum())
    vel = np.linalg.norm(np.diff(kps, axis=0), axis=-1) if T >= 2 \
        else np.zeros((0, 17), np.float32)                            # [T-1,17]；T<2 时空

    def _safe(f, default=0.0):
        return float(f()) if vel.size else default

    sw = np.linalg.norm(kps[:, SHL] - kps[:, SHR], axis=-1)           # 肩宽 [T]
    torso = np.linalg.norm(kps[:, PELVIS] - kps[:, NECK], axis=-1)    # 躯干 [T]
    z = kps[:, :, 2]
    return {
        "frames": T,
        "allzero": allzero,
        "nonfinite": int((~np.isfinite(kps)).sum()),
        "vel_mean": _safe(vel.mean), "vel_p95": _safe(lambda: np.percentile(vel, 95)),
        "vel_p99": _safe(lambda: np.percentile(vel, 99)), "vel_max": _safe(vel.max),
        "shoulder_mean": float(sw.mean()), "shoulder_std": float(sw.std()),
        "shoulder_min": float(sw.min()), "shoulder_max": float(sw.max()),
        "torso_std": float(torso.std()),
        "z_out": int((z > 2.5).sum()), "z_neg": int((z < -0.5).sum()),
        "finite": finite,
    }


def load_clip_kps(pred_dir: Path):
    kps = []
    if pred_dir.is_dir():
        files = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    else:
        files = sorted(Path(pred_dir).glob("*.json"))
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for fr in (data if isinstance(data, list) else [data]):
            if isinstance(fr, dict) and "keypoints" in fr:
                kps.append(np.asarray(fr["keypoints"], dtype=np.float32).reshape(17, 3))
    return np.stack(kps, 0) if kps else np.zeros((0, 17, 3), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--sample_dir", default="", help="本地平铺样例目录 data/Training/data（含 Skeleton/）")
    ap.add_argument("--limit", type=int, default=0, help="只统计前 N 个 clip")
    args = ap.parse_args()

    clips = []
    if args.sample_dir:
        sd = Path(args.sample_dir).expanduser()
        clips = [(Path(x).parent, Path(x).stem) for x in
                 sorted(glob.glob(str(sd / "Skeleton" / "*.json")))]
        # 每帧一个 json，按 parent 分组
        by_dir = {}
        for f in sorted(glob.glob(str(sd / "Skeleton" / "*.json"))):
            by_dir.setdefault(Path(f).parent, []).append(Path(f))
        clip_iter = [(d, c) for d, c in by_dir.items()]
    else:
        root = Path(args.train_root).expanduser()
        idx = build_skeleton_index(root)
        clip_iter = [(c.pred_dir, c) for c in idx]
    if args.limit:
        clip_iter = clip_iter[: args.limit]

    print(f"clips to scan: {len(clip_iter)}", flush=True)
    agg = {"frames": 0, "bad_clips": 0, "n_clips": 0}
    worst = []  # (score, clip)
    for pred_dir, c in clip_iter:
        kps = load_clip_kps(pred_dir)
        s = stat_clip(kps)
        agg["n_clips"] += 1
        agg["frames"] += s["frames"]
        # 坏 clip 判据：帧太少 / 大量全零 / 肩宽异常 / 速度 p99 过大
        bad = (s["frames"] < 8 or s["allzero"] > max(1, s["frames"] * 0.3)
               or (s["shoulder_std"] > 0.03 and s["shoulder_mean"] > 0)
               or s["vel_p99"] > 0.15 or s["nonfinite"] > 0
               or s["shoulder_mean"] < 0.15 or s["shoulder_mean"] > 0.5)
        if bad:
            agg["bad_clips"] += 1
        score = s["vel_p99"] + 10 * s["shoulder_std"] + s["allzero"]
        worst.append((score, str(pred_dir), s))
    worst.sort(reverse=True, key=lambda x: x[0])

    print("\n==== 全量骨架质量汇总 ====")
    print(f"clips={agg['n_clips']} 总帧={agg['frames']} 坏clip={agg['bad_clips']} "
          f"({agg['bad_clips']/max(agg['n_clips'],1)*100:.1f}%)")
    if agg["frames"]:
        print(f"平均每clip帧数={agg['frames']/agg['n_clips']:.1f}")
    print("\n==== 最差 15 个 clip（按 抖动+肩宽漂移+全零 评分）====  ")
    print(f"{'clip':<40} {'帧':>3} {'零帧':>3} {'v_p99':>7} {'肩宽std':>8} {'肩宽m':>6} {'z_out':>5}")
    for score, name, s in worst[:15]:
        print(f"{name[-40:]:<40} {s['frames']:>3} {s['allzero']:>3} "
              f"{s['vel_p99']:>7.4f} {s['shoulder_std']:>8.4f} {s['shoulder_mean']:>6.3f} {s['z_out']:>5}")
    print("\n==== 指标分布（全部分位） ====")
    if agg["n_clips"]:
        vs = [w[2]["vel_p99"] for w in worst]
        sws = [w[2]["shoulder_std"] for w in worst if w[2]["shoulder_mean"] > 0]
        print(f"vel_p99: p50={np.percentile(vs,50):.4f} p90={np.percentile(vs,90):.4f} p99={np.percentile(vs,99):.4f}")
        print(f"shoulder_std: p50={np.percentile(sws,50):.4f} p90={np.percentile(sws,90):.4f} "
              f"p99={np.percentile(sws,99):.4f}")
    print("\n判读：坏 clip 比例 / vel_p99 / shoulder_std 是清洗强度依据。")


if __name__ == "__main__":
    main()
