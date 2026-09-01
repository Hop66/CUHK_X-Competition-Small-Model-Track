#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUHK-X 小模型赛道 —— 服务器数据/环境核查脚本（Step 0）

用途：在服务器上运行，收集本地无法确认的全部信息，输出 JSON + Markdown 报告，
     把报告下载回本地供架构师决策。

用法:
    python inspect_server.py \
        --train_root /path/to/HAR/data \
        --test_root  /path/to/small_model_track_test \
        --out ./report

说明:
    - 训练集预期结构: <train_root>/<Modality>/<Action>/<Subject>/[<sample>]/<files>
    - 测试集预期结构: <test_root>/SM_test_XXXX/<Modality>/<files>
    - 脚本全部 try/except，任何一步失败不影响其余步骤。
    - 只读、不写数据、不上传任何东西。
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------- 通用 ----------
MODALITIES = ["Thermal", "Depth_Color", "IR", "Skeleton", "IMU", "Radar"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
IMU_SENSOR_ORDER = ["WTLL", "WTRL", "WTC", "WTLA", "WTRA"]  # 左腿/右腿/躯干/左臂/右臂

report = {"meta": {}, "train": {}, "test": {}, "modalities": {}, "environment": {}}


def log(msg):
    print(msg, flush=True)


def safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa
        return f"ERROR: {e}"


# ---------- 环境 ----------
def check_environment():
    env = report["environment"]
    env["platform"] = sys.platform
    env["python"] = sys.version.split()[0]
    env["cwd"] = os.getcwd()
    try:
        import numpy as np
        env["numpy"] = np.__version__
    except Exception as e:
        env["numpy"] = f"missing: {e}"
    try:
        import pandas as pd
        env["pandas"] = pd.__version__
    except Exception as e:
        env["pandas"] = f"missing: {e}"
    try:
        import cv2
        env["opencv"] = cv2.__version__
    except Exception as e:
        env["opencv"] = f"missing: {e}"
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                free, total = torch.cuda.mem_get_info(0)
                env["gpu_mem_free_gb"] = round(free / 1e9, 1)
                env["gpu_mem_total_gb"] = round(total / 1e9, 1)
            except Exception:
                pass
    except Exception as e:
        env["torch"] = f"missing: {e}"
        env["cuda_available"] = False
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except Exception as e:
        env["ultralytics"] = f"missing: {e}"
    # 磁盘剩余
    for p in [".", str(Path.home())]:
        try:
            s = os.statvfs(p)
            env[f"disk_free_gb({p})"] = round(s.f_bavail * s.f_frsize / 1e9, 1)
        except Exception:
            pass
    # 网络到 HuggingFace（预训练权重来源）
    env["hf_reachable"] = safe(
        lambda: _probe_url("https://huggingface.co", timeout=5), "unreachable")


def _probe_url(url, timeout=5):
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


# ---------- 图像物理含义检查 ----------
def image_stats(path):
    import cv2
    import numpy as np
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"error": "unreadable"}
    out = {"shape": list(img.shape), "dtype": str(img.dtype)}
    if img.ndim == 3 and img.shape[2] == 3:
        b = img[..., 0].astype(np.float32).ravel()
        g = img[..., 1].astype(np.float32).ravel()
        r = img[..., 2].astype(np.float32).ravel()
        out["mean_BGR"] = [round(float(b.mean()), 1), round(float(g.mean()), 1), round(float(r.mean()), 1)]
        out["std_BGR"] = [round(float(b.std()), 1), round(float(g.std()), 1), round(float(r.std()), 1)]
        out["corr_B_R"] = round(float(np.corrcoef(b, r)[0, 1]), 3)
        out["corr_G_R"] = round(float(np.corrcoef(g, r)[0, 1]), 3)
        # 判读: corr_B_R 明显为负 => 伪彩色(深度jet/热力ironbow)；自然RGB应为正
        out["looks_pseudocolor"] = bool(out["corr_B_R"] < -0.3)
    else:
        out["mean"] = round(float(img.mean()), 2)
        out["std"] = round(float(img.std()), 2)
    return out


# ---------- 训练集扫描 ----------
def scan_train(root: Path):
    tr = report["train"]
    tr["root"] = str(root)
    if not root.exists():
        tr["error"] = "train_root not found"
        return

    # 1) 发现模态目录（兼容 <root>/<Modality> 与 <root>/data/<Modality> 两种布局）
    present_mods = [m for m in MODALITIES if (root / m).is_dir()]
    if not present_mods and (root / "data").is_dir():
        root = root / "data"
        tr["root"] = str(root)
        present_mods = [m for m in MODALITIES if (root / m).is_dir()]
    tr["present_modalities"] = present_mods
    if not present_mods:
        tr["error"] = f"no modality dirs under {root}; 请确认 train_root 指向包含 Thermal/Depth_Color/... 的目录"
        return

    # 2) 用第一个存在的模态枚举 clip (action/subject/sample)
    disc = present_mods[0]
    tr["discovery_modality"] = disc
    clips = []  # (action_dir_name, subject_dir_name, sample_dir_name)
    action_subjects = defaultdict(set)
    class_counter = Counter()
    for action_dir in sorted((root / disc).iterdir()):
        if not action_dir.is_dir():
            continue
        try:
            action_id = int(action_dir.name.split("_")[0])
        except Exception:
            action_id = None
        for subj_dir in sorted(action_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            # subject 目录下可能是 sample 目录，也可能直接是文件
            entries = [e for e in subj_dir.iterdir()]
            has_sample_dirs = any(e.is_dir() for e in entries)
            if has_sample_dirs:
                for sample_dir in sorted(subj_dir.iterdir()):
                    if sample_dir.is_dir():
                        clips.append((action_dir.name, subj_dir.name, sample_dir.name))
                        action_subjects[action_dir.name].add(subj_dir.name)
                        if action_id is not None:
                            class_counter[action_id] += 1
            else:
                clips.append((action_dir.name, subj_dir.name, None))
                action_subjects[action_dir.name].add(subj_dir.name)
                if action_id is not None:
                    class_counter[action_id] += 1

    tr["total_clips"] = len(clips)
    tr["total_actions"] = len(action_subjects)
    subjects = {s for v in action_subjects.values() for s in v}
    tr["total_subjects"] = len(subjects)
    tr["subject_ids"] = sorted(subjects)
    tr["has_sample_level"] = any(c[2] is not None for c in clips)

    # 类分布
    if class_counter:
        counts = sorted(class_counter.values())
        tr["class_distribution"] = dict(sorted(class_counter.items()))
        tr["class_min"] = counts[0]
        tr["class_max"] = counts[-1]
        tr["class_mean"] = round(sum(counts) / len(counts), 1)
        tr["longtail_ratio"] = round(counts[-1] / max(counts[0], 1), 1)
    # 每个 subject 的 clip 数（用于设计 subject-fold）
    subj_clip_count = Counter()
    for c in clips:
        subj_clip_count[c[1]] += 1
    tr["subject_clip_count"] = dict(sorted(subj_clip_count.items()))

    # 3) 逐模态统计
    for mod in present_mods:
        scan_modality(root, mod, clips)


def scan_modality(root: Path, mod: str, clips):
    m = report["modalities"].setdefault(mod, {})
    mod_root = root / mod
    # 采样第一组 action/subject/sample
    sample_path, sample_files = None, []
    n_missing = 0
    frame_counts = []
    for (act, subj, sample) in clips:
        p = mod_root / act / subj / (sample if sample else "")
        # Skeleton 的 json 在 predictions/ 子目录下
        content = p / "predictions" if (mod == "Skeleton" and (p / "predictions").is_dir()) else p
        files = sorted([f for f in content.iterdir() if f.is_file()]) if content.is_dir() else []
        if not files:
            n_missing += 1
            continue
        frame_counts.append(len(files))
        if sample_path is None:
            sample_path = content
            sample_files = files
    m["missing_clips"] = n_missing
    m["missing_rate"] = round(100.0 * n_missing / max(len(clips), 1), 1)
    if frame_counts:
        m["frames_per_clip_min"] = min(frame_counts)
        m["frames_per_clip_max"] = max(frame_counts)
        m["frames_per_clip_mean"] = round(sum(frame_counts) / len(frame_counts), 1)

    if not sample_files:
        m["sample"] = "no files found"
        return
    f = sample_files[0]
    m["sample_dir"] = str(sample_path)
    m["sample_file"] = f.name
    ext = f.suffix.lower()

    if mod in ("Thermal", "Depth_Color", "IR"):
        if ext in IMG_EXTS:
            st = image_stats(f)
            m["image"] = st
            # 帧率估计: 从文件名时间戳差
            m["fps_estimate"] = estimate_fps(sample_files)
        else:
            m["image"] = {"error": f"unexpected ext {ext}"}

    elif mod == "Skeleton":
        m["skeleton"] = inspect_skeleton(sample_files)
        m["fps_estimate"] = estimate_fps(sample_files)

    elif mod == "IMU":
        m["imu"] = inspect_imu(sample_files)

    elif mod == "Radar":
        m["radar"] = inspect_radar(clips, mod_root)


def estimate_fps(files):
    """从文件名里的时间戳估算帧率（Depth/IR/Skeleton 文件名含毫秒时间戳）。"""
    import re
    ts = []
    for f in files[:200]:
        mt = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})", f.name)
        if mt:
            try:
                from datetime import datetime
                ts.append(datetime.strptime(mt.group(1), "%Y-%m-%d_%H-%M-%S.%f").timestamp())
            except Exception:
                pass
    if len(ts) >= 2:
        diffs = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if diffs:
            avg = sum(diffs) / len(diffs)
            return round(1.0 / avg, 2) if avg > 0 else None
    return None


def inspect_skeleton(files):
    import json
    out = {}
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                d0 = data[0]
                out["num_frames_in_file"] = len(data)
                out["first_frame_keys"] = list(d0.keys()) if isinstance(d0, dict) else "not dict"
                if isinstance(d0, dict) and "keypoints" in d0:
                    import numpy as np
                    kp = np.asarray(d0["keypoints"])
                    out["keypoints_shape"] = list(kp.shape)
                    if "keypoint_scores" in d0:
                        out["has_keypoint_scores"] = True
                        out["scores_shape"] = list(np.asarray(d0["keypoint_scores"]).shape)
                    else:
                        out["has_keypoint_scores"] = False
                break
        except Exception as e:
            out["error"] = f"{f.name}: {e}"
            break
    return out


def inspect_imu(files):
    out = {}
    for f in files:
        try:
            df = _read_imu_csv(f)
            out["file"] = f.name
            out["columns"] = list(df.columns)
            out["n_columns"] = len(df.columns)
            out["n_rows"] = len(df)
            # 设备名
            if "设备名称" in df.columns or "device" in str(df.columns).lower():
                devcol = "设备名称" if "设备名称" in df.columns else [c for c in df.columns if "device" in c.lower()][0]
                names = df[devcol].astype(str).str[:4]
                out["sensors_found"] = sorted(names.unique().tolist())
                out["sensor_order_ok"] = all(s in out["sensors_found"] for s in IMU_SENSOR_ORDER)
            # 时间戳是否乱序（全局）
            tcol = df.columns[0]
            tvals = df[tcol].astype(str)
            is_sorted = tvals.is_monotonic_increasing
            out["timestamp_global_sorted"] = bool(is_sorted)
            # 采样率估计
            try:
                from datetime import datetime
                ts = pd_to_datetime(tvals)
                diffs = ts.diff().dropna().dt.total_seconds()
                if len(diffs) > 0 and diffs.median() > 0:
                    out["hz_estimate_global"] = round(1.0 / float(diffs.median()), 1)
            except Exception:
                pass
            break
        except Exception as e:
            out["error"] = f"{f.name}: {e}"
            break
    return out


def pd_to_datetime(series):
    import pandas as pd
    try:
        return pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return pd.to_datetime(series)


def _read_imu_csv(path):
    import pandas as pd
    # 先尝试 gbk/gb2312，再 utf-8
    for enc in ("gbk", "gb2312", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def inspect_radar(clips, mod_root):
    out = {}
    total = valid = empty = 0
    sample = None
    for (act, subj, sample_dir) in clips:
        p = mod_root / act / subj / (sample_dir if sample_dir else "")
        files = [f for f in p.iterdir() if f.is_file()] if p.is_dir() else []
        if not files:
            continue
        total += 1
        f = files[0]
        if sample is None:
            sample = f
        size = f.stat().st_size
        if size < 60:  # 只有表头或空
            empty += 1
        else:
            valid += 1
    out["files_sampled"] = total
    out["empty_or_header_only"] = empty
    out["valid"] = valid
    out["valid_rate"] = round(100.0 * valid / max(total, 1), 1)
    if sample is not None:
        try:
            head = sample.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
            out["sample_header"] = head
        except Exception:
            pass
    return out


# ---------- 测试集扫描 ----------
def scan_test(root: Path):
    te = report["test"]
    te["root"] = str(root)
    if not root.exists():
        te["error"] = "test_root not found"
        return
    clip_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("SM_test_")])
    te["total_clips"] = len(clip_dirs)
    if not clip_dirs:
        te["error"] = "no SM_test_* dirs found"
        return
    first = clip_dirs[0]
    mods = sorted([d.name for d in first.iterdir() if d.is_dir()])
    te["modalities_per_clip"] = mods
    # 每模态帧数
    for m in mods:
        files = list((first / m).glob("*"))
        te.setdefault("frames_per_clip_sample", {})[m] = len([f for f in files if f.is_file()])
    # 抽样检查一张 Depth_Color 的物理含义
    dc = first / "Depth_Color"
    if dc.is_dir():
        imgs = sorted(dc.glob("*.png"))
        if imgs:
            te["Depth_Color_image"] = image_stats(imgs[len(imgs) // 2])


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str,
                    default=os.path.expanduser("~/Multimodal/data/Training/HAR"),
                    help="训练集根目录（含 Thermal/Depth_Color/... 模态子目录）")
    ap.add_argument("--test_root", type=str,
                    default=os.path.expanduser("~/Multimodal/data/Testing/data/small_model_track_test"),
                    help="测试集根目录（含 SM_test_XXXX 子目录）")
    ap.add_argument("--out", type=str, default="./report", help="输出目录")
    args = ap.parse_args()

    report["meta"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["meta"]["hostname"] = safe(lambda: __import__("socket").gethostname(), "unknown")

    log("=" * 60)
    log("[1/4] 检查环境 ...")
    check_environment()

    log("[2/4] 扫描训练集 ...")
    scan_train(Path(args.train_root))

    log("[3/4] 扫描测试集 ...")
    scan_test(Path(args.test_root))

    log("[4/4] 写报告 ...")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / "data_report.md"
    md_path.write_text(render_md(report), encoding="utf-8")

    log(f"\nDONE. JSON: {json_path}   MD: {md_path}")
    log("把 data_report.json 和 data_report.md 下载回本地即可。")
    # 控制台速览
    print(json.dumps(report, ensure_ascii=False, indent=2))


def render_md(r):
    lines = ["# CUHK-X 服务器数据核查报告", ""]
    lines += [f"- 生成时间: {r['meta'].get('generated_at')}",
              f"- 主机: {r['meta'].get('hostname')}", ""]
    env = r["environment"]
    lines += ["## 环境", ""]
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")
    tr = r["train"]
    lines += ["", "## 训练集", ""]
    for k, v in tr.items():
        if k in ("class_distribution", "subject_clip_count", "subject_ids"):
            continue
        lines.append(f"- **{k}**: {v}")
    if tr.get("class_distribution"):
        lines += ["", "### 类分布", "```"] + [f"{k}: {v}" for k, v in sorted(tr["class_distribution"].items())] + ["```"]
    if tr.get("subject_clip_count"):
        lines += ["", "### 每个被试的 clip 数（subject-fold 设计用）", "```"] + [
            f"{k}: {v}" for k, v in sorted(tr["subject_clip_count"].items())] + ["```"]
    lines += ["", "## 各模态", ""]
    for mod, v in r["modalities"].items():
        lines.append(f"### {mod}")
        for k, vv in v.items():
            lines.append(f"- **{k}**: {vv}")
    te = r["test"]
    lines += ["", "## 测试集", ""]
    for k, v in te.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
