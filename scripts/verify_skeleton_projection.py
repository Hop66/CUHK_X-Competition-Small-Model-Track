#!/usr/bin/env python3
"""P1-1b 前置：验证骨架 3D→IR 自校准投影（画到 IR 图像上，肉眼检查）

方法：用 IR 的 YOLO bbox（bbox_train.json）+ 骨架 3D 尺寸自校准
  - 参考点 = bbox 中心（人体 2D 位置）
  - 尺度 = bbox 高度(像素) / 骨架身高(3D 米) → 像素/米
  - 每个关节 = bbox 中心 + 相对骨盆的 (dx, dy) × 尺度
  - 简化：先忽略深度差异（近大远小），看是否大致合理

用法（服务器，CPU 即可）:
  CUDA_VISIBLE_DEVICES="" python scripts/verify_skeleton_projection.py
输出: outputs/verify_skel_proj/*.jpg（骨架叠在 IR 上，检查关节是否落在人体上）
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

IMG_W, IMG_H = 640, 480
PELVIS, HEAD, HEAD_TOP = 0, 9, 10  # H3.6M-17：0=骨盆, 9=头, 10=头顶点


def frame_num_of(name: str):
    m = re.search(r"(\d{8})", name)
    return int(m.group(1)) if m else None


def calibrate_clip(jframes, bbox_norm, va, ha, pad_top: float = 0.05):
    """每 clip 一次标定（用中位数，防下蹲帧尺度爆炸、防个别坏帧）。

    返回投影参数：
      - scale = 框高 / 头踝垂直分量（中位数，不是 3D 范数——范数含深度会压扁）
      - 垂直锚定：头顶点(中位数) → 框上边 + 5% padding
      - 水平锚定：骨盆(中位数) → 框水平中心
    """
    x1, y1, x2, y2 = bbox_norm
    px0, py0 = x1 * IMG_W, y1 * IMG_H
    px1, py1 = x2 * IMG_W, y2 * IMG_H
    ht = np.median(np.stack([f[HEAD_TOP] for f in jframes]), 0)   # 头顶点中位数
    an = np.median(np.stack([(f[3] + f[6]) / 2.0 for f in jframes]), 0)  # 踝中点中位数
    pe = np.median(np.stack([f[PELVIS] for f in jframes]), 0)     # 骨盆中位数
    h_vert = float(abs(ht[va] - an[va])) + 1e-6
    h_pix = py1 - py0
    scale = h_pix / h_vert  # 像素/米
    up_sign = 1.0 if ht[va] >= an[va] else -1.0
    up_head = ht[va] * up_sign
    top_pix = py0 + pad_top * h_pix
    cx = (px0 + px1) / 2.0
    return dict(scale=scale, up_sign=up_sign, up_head=up_head, top_pix=top_pix,
                cx=cx, pe=pe, h_vert=h_vert)


def project_frame(joints3d, cal, va, ha, flip_u: bool = False, flip_v: bool = False):
    """逐帧投影：用 clip 级标定参数。flip_u=水平镜像，flip_v=垂直翻转。"""
    scale, us = cal["scale"], cal["up_sign"]
    uh, tp, cx, pe = cal["up_head"], cal["top_pix"], cal["cx"], cal["pe"]
    su = -1.0 if flip_u else 1.0
    sv = -1.0 if flip_v else 1.0
    pts = []
    for j in joints3d:
        up = j[va] * us * sv
        right = (j[ha] - pe[ha]) * su
        u = cx + right * scale
        v = tp - (up - uh * sv) * scale  # 头顶在顶边，向下走
        pts.append((u, v))
    return pts


# H3.6M-17 骨架连线（画图用）
EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
         (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
         (8, 14), (14, 15), (15, 16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--bbox", default="bbox_train.json",
                    help="投影锚定框。推荐先用 detect.py 生成紧框:\n"
                         "  python -m src.detect --train_root ... --out bbox_tight_train.json --margin 1.1 --no_square")
    ap.add_argument("--crop", default=None,
                    help="训练裁剪窗口（绿色框，用于对比；默认同 --bbox）")
    ap.add_argument("--out", default="outputs/verify_skel_proj")
    ap.add_argument("--n_clips", type=int, default=4, help="验证的 clip 数")
    ap.add_argument("--flip_u", action="store_true", help="水平镜像（若骨架左右反了）")
    ap.add_argument("--flip_v", action="store_true", help="垂直翻转（若骨架上下颠倒）")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    bbox = json.loads(Path(args.bbox).expanduser().read_text(encoding="utf-8"))
    crop_path = args.crop or args.bbox
    crop = json.loads(Path(crop_path).expanduser().read_text(encoding="utf-8"))
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[verify] anchor={args.bbox} ({len(bbox)} clips) crop={crop_path} "
          f"flip_u={args.flip_u} flip_v={args.flip_v}", flush=True)

    # ---- 第一遍：收集所有 sample 的骨架帧 + IR 映射 ----------------
    samples = []
    for action in sorted(root.glob("Skeleton/*"))[:args.n_clips]:
        if not action.is_dir():
            continue
        for subj in sorted(action.iterdir())[:1]:
            for sample in sorted(subj.iterdir()):
                if not sample.is_dir():
                    continue
                key = f"{action.name.split('_')[0]}/{subj.name}/{sample.name}"
                if key not in bbox:
                    continue
                pred = sample / "predictions"
                files = sorted(pred.glob("*.json"))
                if len(files) < 16:
                    continue
                ir_dir = root / "IR" / action.name / subj.name / sample.name
                ir_map = {}
                if ir_dir.is_dir():
                    for p in ir_dir.iterdir():
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            fno = frame_num_of(p.name)
                            if fno is not None:
                                ir_map[fno] = p
                idx = np.linspace(0, len(files) - 1, 6).round().astype(int)
                jframes = []
                for j in idx:
                    fr = json.loads(files[j].read_text(encoding="utf-8"))
                    fr = fr[0] if isinstance(fr, list) else fr
                    jframes.append(np.asarray(fr["keypoints"], np.float32)[:, :3])
                samples.append(dict(key=key, files=files, idx=idx, jframes=jframes,
                                    ir_map=ir_map, bbox=bbox[key],
                                    crop=crop.get(key, bbox[key])))
                break

    # ---- 全局轴：跨所有 sample 聚合（相机固定，全数据集同一列约定） ----
    spans, shws = [], []
    for s in samples:
        spans.append(np.abs(np.stack([f[HEAD_TOP] - (f[3] + f[6]) / 2.0
                                     for f in s["jframes"]])).mean(0))
        shws.append(np.abs(np.stack([f[11] - f[14] for f in s["jframes"]])).mean(0))
    spans = np.mean(spans, 0)
    shws = np.mean(shws, 0)
    va = int(np.argmax(spans))
    rem = [a for a in range(3) if a != va]
    ha = int(max(rem, key=lambda a: shws[a]))
    depth = [a for a in range(3) if a not in (va, ha)][0]
    print(f"[全局轴] 头踝跨度每列={np.round(spans,2)} | 肩宽每列={np.round(shws,2)} "
          f"→ 垂直列={va} 水平列={ha} 深度列={depth}", flush=True)

    # ---- 第二遍：每 clip 中位数标定 + 逐帧投影渲染 ----------------
    n_saved = 0
    for s in samples:
        cal = calibrate_clip(s["jframes"], s["bbox"], va, ha)
        print(f"  [标定] {s['key']} 尺度={cal['scale']:.0f}px/m 头踝垂跨={cal['h_vert']:.2f}m "
              f"up_sign={cal['up_sign']} 锚框={[round(v,3) for v in s['bbox']]}", flush=True)
        n_ir_hit = 0
        for t, joints in enumerate(s["jframes"]):
            fno = frame_num_of(s["files"][s["idx"][t]].name)
            p = s["ir_map"].get(fno)
            if p is None:
                continue
            ir_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if ir_img is None:
                continue
            n_ir_hit += 1
            if ir_img.ndim == 2:
                ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
            pts = project_frame(joints, cal, va, ha, args.flip_u, args.flip_v)
            # 画训练裁剪窗口（绿）
            x1, y1, x2, y2 = s["crop"]
            cv2.rectangle(ir_img, (int(x1 * IMG_W), int(y1 * IMG_H)),
                          (int(x2 * IMG_W), int(y2 * IMG_H)), (0, 255, 0), 2)
            # 画骨架
            for (a, b) in EDGES:
                pxa, pya = pts[a]
                pxb, pyb = pts[b]
                cv2.line(ir_img, (int(pxa), int(pya)), (int(pxb), int(pyb)), (0, 0, 255), 2)
            for (u, v) in pts:
                cv2.circle(ir_img, (int(u), int(v)), 3, (255, 0, 0), -1)
            # 画投影骨架自身外接框（青）对比绿框
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cv2.rectangle(ir_img, (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))), (255, 255, 0), 1)
            in_img = sum(0 <= u < IMG_W and 0 <= v < IMG_H for u, v in pts)
            fname = out_dir / f"{s['key'].replace('/', '_')}_t{t}_in{in_img}.jpg"
            cv2.imwrite(str(fname), ir_img)
            n_saved += 1
        print(f"saved {s['key']}（{len(s['files'])}帧, IR命中 {n_ir_hit}/6, "
              f"IR文件 {len(s['ir_map'])}）", flush=True)
    print(f"\n==== 保存 {n_saved} 张验证图到 {out_dir} ====", flush=True)
    print("判读:", flush=True)
    print("  绿框=训练裁剪窗口；青框=骨架外接框；红线/蓝点应贴紧人体", flush=True)
    print("  若左右反了 → 加 --flip_u；若上下颠倒 → 加 --flip_v", flush=True)


if __name__ == "__main__":
    main()
