"""
CUHK-X —— YOLO 人体检测 → 每 clip 一个固定窗口（保运动）

原则（不是逐帧裁剪）：
  多帧探测 IR → union 所有框 → 1.4x 放大 → 转正方形 → 全程固定。
  固定窗口保留"人在画面中移动"的全局运动；逐帧移动裁剪会抹掉位移运动。

用法（离线缓存 bbox，一次跑完）：
    python -m src.detect --test_root ... --out bbox_test.json
    python -m src.detect --train_root ... --out bbox_train.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

BOX = Tuple[float, float, float, float]  # 归一化 (x1,y1,x2,y2)


def load_yolo():
    """加载 YOLO11n：本地有则直接用，否则用 huggingface_hub 走 HF 镜像（不直连 hf-mirror.com）。"""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from ultralytics import YOLO
    local = Path("yolo11n.pt")
    if local.is_file():
        return YOLO(str(local))
    from huggingface_hub import hf_hub_download
    print("[detect] 从 HF 镜像下载 yolo11n.pt (huggingface_hub) ...", flush=True)
    p = hf_hub_download(repo_id="Ultralytics/YOLO11", filename="yolo11n.pt",
                        local_dir=".", local_dir_use_symlinks=False)
    return YOLO(str(p))


def pick_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    return np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()


def window_from_boxes(boxes: List[List[float]], margin: float = 1.4,
                      min_side: float = 0.35, w: int = 640, h: int = 480,
                      square: bool = True, median_center: bool = False) -> BOX:
    arr = np.asarray(boxes, dtype=np.float64)  # 已经是归一化 xyxy
    if median_center:
        # v9: median of per-box centres（对 stray 检测鲁棒）；边长仍用最大框×margin（保训练 scale）
        cx = float(np.median((arr[:, 0] + arr[:, 2]) / 2.0))
        cy = float(np.median((arr[:, 1] + arr[:, 3]) / 2.0))
    else:
        x0, y0 = arr[:, 0].min(), arr[:, 1].min()
        x1, y1 = arr[:, 2].max(), arr[:, 3].max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0, y0 = arr[:, 0].min(), arr[:, 1].min()
    x1, y1 = arr[:, 2].max(), arr[:, 3].max()
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    if square:
        # 训练窗口：正方形 + 大 margin（保位移、防切肢体），1.4x
        side = max(bw, bh) * margin
        side = max(side, min_side * max(w, h))
        hx, hy = side / w / 2.0, side / h / 2.0
    else:
        # 紧框：宽高各自放大 margin（不做正方形化），贴近真实人体（骨架投影锚定用）
        hx, hy = bw * margin / 2.0 / w, bh * margin / 2.0 / h
    return (max(cx - hx, 0.0), max(cy - hy, 0.0), min(cx + hx, 1.0), min(cy + hy, 1.0))


def detect_clip(model, frame_dir: Path, probes: int = 8, device: str = "cpu",
                margin: float = 1.4, square: bool = True,
                median_center: bool = False) -> Optional[BOX]:
    """在帧目录（IR 灰度 或 Thermal 伪彩色）上探测 person，返回固定窗口或 None。"""
    import cv2
    files = sorted(list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg"))) if frame_dir.is_dir() else []
    if not files:
        return None
    idxs = pick_indices(len(files), probes)
    boxes: List[List[float]] = []
    for i in idxs:
        img = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.dtype == np.uint16:
            img = (img / 65535.0 * 255.0).astype(np.uint8)
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, classes=[0], conf=0.25, verbose=False, device=device)
        if len(results[0].boxes):
            b = results[0].boxes.xyxy[results[0].boxes.conf.argmax()].tolist()
            H, W = results[0].orig_shape
            boxes.append([b[0] / W, b[1] / H, b[2] / W, b[3] / H])
    return window_from_boxes(boxes, margin=margin, square=square,
                             median_center=median_center) if boxes else None


def detect_clip_perframe(model, frame_dir: Path, device: str = "cpu",
                         margin: float = 1.1, square: bool = False) -> Optional[dict]:
    """逐帧 person bbox → {"names":[...], "boxes":[[xyxy]|None,...]}（归一化）。
    names 与帧文件一一对应，供 dataset 按文件名对齐（规避排序差异）。
    用途：每帧跟人裁剪 + 显式位移/尺度轨迹（用户设想：极致裁人物 + 位移回补）。
    """
    import cv2
    files = sorted(list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg"))) if frame_dir.is_dir() else []
    names, boxes = [], []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        box = None
        if img is not None:
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255.0).astype(np.uint8)
            if img.ndim == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, classes=[0], conf=0.25, verbose=False, device=device)
            if len(results[0].boxes):
                b = results[0].boxes.xyxy[results[0].boxes.conf.argmax()].tolist()
                H, W = results[0].orig_shape
                box = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
                if margin != 1.0:
                    hx = (box[2] - box[0]) * (margin - 1.0) / 2.0
                    hy = (box[3] - box[1]) * (margin - 1.0) / 2.0
                    box = [max(box[0] - hx, 0.0), max(box[1] - hy, 0.0),
                           min(box[2] + hx, 1.0), min(box[3] + hy, 1.0)]
        names.append(f.name)
        boxes.append(box)
    return {"names": names, "boxes": boxes} if files else None


def run_train(root: Path, out: Path, modality: str = "depthir",
              margin: float = 1.4, square: bool = True, per_frame: bool = False,
              median_center: bool = False):
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[detect] device={device} modality={modality} margin={margin} square={square} median={median_center}", flush=True)
    model = load_yolo()
    cache: Dict[str, List[float]] = {}
    n_found = n_fallback = 0
    if modality == "thermal":
        from src.dataset import build_thermal_index
        clips = build_thermal_index(root)
        frame_dir_of = lambda c: c.thermal_dir
    else:
        from src.dataset import build_train_index
        clips = build_train_index(root)
        frame_dir_of = lambda c: c.ir_dir
    print(f"[detect] {len(clips)} clips 待检测 ...", flush=True)
    for idx, c in enumerate(clips):
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        if per_frame:
            pf = detect_clip_perframe(model, frame_dir_of(c), device=device, margin=margin, square=square)
            if pf is not None and any(b is not None for b in pf["boxes"]):
                cache[key] = pf
                n_found += 1
            else:
                n_fallback += 1
        else:
            b = detect_clip(model, frame_dir_of(c), device=device, margin=margin, square=square,
                            median_center=median_center)
            if b is not None:
                cache[key] = list(b)
                n_found += 1
            else:
                n_fallback += 1
        if (idx + 1) % 200 == 0:
            print(f"[detect] {idx + 1}/{len(clips)} (found={n_found})", flush=True)
    out.write_text(json.dumps(cache), encoding="utf-8")
    print(f"train bbox cache [{modality}]: {len(clips)} clips, {n_found} detected, {n_fallback} fallback -> {out}")


def run_test(root: Path, out: Path, modality: str = "depthir",
             margin: float = 1.4, square: bool = True):
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[detect] device={device} modality={modality} margin={margin} square={square}", flush=True)
    model = load_yolo()
    cache: Dict[str, List[float]] = {}
    n_found = n_fallback = 0
    frame_subdir = "Thermal" if modality == "thermal" else "IR"
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("SM_test_"):
            continue
        b = detect_clip(model, d / frame_subdir, device=device, margin=margin, square=square)
        if b is not None:
            cache[d.name] = list(b)
            n_found += 1
        else:
            n_fallback += 1
    out.write_text(json.dumps(cache), encoding="utf-8")
    print(f"test bbox cache [{modality}]: {len(cache)} clips, {n_found} detected, {n_fallback} fallback -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--test_root", type=str, default="~/Multimodal/data/Testing/data/small_model_track_test")
    ap.add_argument("--out", type=str, default="bbox_cache.json")
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--modality", choices=["depthir", "thermal"], default="depthir",
                    help="depthir=在 IR 上检测（主线）；thermal=在 Thermal 上检测（Thermal 定位增强）")
    ap.add_argument("--margin", type=float, default=1.4,
                    help="窗口放大倍数（1.4=训练窗口；1.1=紧框，用于骨架投影锚定）")
    ap.add_argument("--no_square", action="store_true",
                    help="不做正方形化（紧框：宽高各自放大，贴近真实人体）")
    ap.add_argument("--per_frame", action="store_true",
                    help="逐帧 person bbox 输出（每帧跟人裁剪 + 位移/尺度轨迹用；紧框默认）")
    ap.add_argument("--median_center", action="store_true",
                    help="median-centre crop（yolo v9：对 stray 检测鲁棒；固定 clip 框用）")
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "train":
        run_train(Path(args.train_root).expanduser(), out, args.modality,
                  margin=args.margin, square=not args.no_square, per_frame=args.per_frame,
                  median_center=args.median_center)
    else:
        run_test(Path(args.test_root).expanduser(), out, args.modality,
                 margin=args.margin, square=not args.no_square)
