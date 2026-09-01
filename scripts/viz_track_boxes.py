#!/usr/bin/env python3
"""CUHK-X —— 逐帧跟人 bbox 可视化 + 质量统计（服务器跑，真数据）

验证 track_crop 数据链在真实 clip 上是否成立：
  ① 覆盖：有 bbox 的帧占比（越高越好）
  ② 框住人体 + 大小：box 面积占画面比例（太大→背景多；太小→切肢体；目标~0.2-0.5）
  ③ 居中：box 中心与画面中心偏移（偏移大→人物偏，裁剪后人物可能出框）
  ④ 平滑：相邻帧 box 中心位移（抖动大→检测不稳，轨迹噪声）
  ⑤ 裁剪后人物比例：box crop+resize 后人物占多少（近似=1/box面积 → 裁剪即放大）

输出：
  - 统计表（上述指标）
  - clip_key 的处理图: 原图5帧+box、裁剪后人物图、轨迹折线（均用 cv2/np，无 matplotlib 依赖）
  - 判读文本

用法（服务器）:
  python scripts/viz_track_boxes.py --frame_dir ~/Multimodal/data/Training/HAR/Thermal/<Action>/<User>/<Trial> \
      --mode detect --out outputs/viz_track   # 现场 YOLO 检测
  # 或读已生成的逐帧 bbox：
  python scripts/viz_track_boxes.py --frame_dir <同上> --box bbox_thermal_perframe.json \
      --clip_key <action>/<subject>/<sample> --out outputs/viz_track
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2


def load_boxes(args):
    """返回 (names, boxes_normalized_list or None)。"""
    if args.mode == "detect":
        from src.detect import detect_clip_perframe
        model = None  # detect_clip_perframe 需要 model；下面单独处理
        # 为复用：走 src.detect.load_yolo
        from src.detect import load_yolo
        model = load_yolo()
        pf = detect_clip_perframe(model, Path(args.frame_dir).expanduser(),
                                  margin=args.margin, square=False)
        return pf["names"], pf["boxes"]
    raw = json.loads(Path(args.box).expanduser().read_text(encoding="utf-8"))
    v = raw[args.clip_key]
    return v["names"], v["boxes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", default="",
                    help="一个 clip 的图像目录（cache 模式可省：按 --clip_key 从索引自动反查）")
    ap.add_argument("--mode", choices=["detect", "cache"], default="cache")
    ap.add_argument("--box", default="bbox_thermal_perframe.json")
    ap.add_argument("--clip_key", default="", help="cache 模式：bbox json 里的 key")
    ap.add_argument("--margin", type=float, default=1.1)
    ap.add_argument("--out", default="outputs/viz_track")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # ---- 定位帧目录：cache 模式可不给 --frame_dir，按 clip_key 从索引自动反查 ----
    frame_dir = Path(args.frame_dir).expanduser() if args.frame_dir else None
    if args.mode == "cache":
        if frame_dir is None:
            from src.dataset import build_thermal_index, build_train_index
            root = Path("~/Multimodal/data/Training/HAR").expanduser()
            use_thermal = "thermal" in Path(args.box).name.lower()
            clips = build_thermal_index(root) if use_thermal else build_train_index(root)
            hit = next((c for c in clips
                        if f"{c.action_id}/{c.subject}/{c.sample}" == args.clip_key), None)
            if hit is None:
                print(f"❌ clip_key={args.clip_key!r} 不在 {len(clips)} 个 clip 里；示例 key: "
                      f"{[f'{c.action_id}/{c.subject}/{c.sample}' for c in clips[:5]]}")
                sys.exit(1)
            frame_dir = Path(hit.thermal_dir) if use_thermal else Path(hit.ir_dir)
            print(f"[viz] 自动定位 frame_dir = {frame_dir}", flush=True)
        if not frame_dir.is_dir():
            print(f"❌ frame_dir 不存在: {frame_dir}")
            sys.exit(1)
    else:
        if frame_dir is None:
            print("❌ detect 模式必须给 --frame_dir")
            sys.exit(1)
        args.clip_key = "detect"

    files = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
    if not files:
        print(f"❌ {frame_dir} 下没有 jpg/png 帧（目录错或扩展名不支持）")
        sys.exit(1)

    names, boxes = load_boxes(args)
    name2box = {n: b for n, b in zip(names, boxes)}

    # 1) 覆盖 & 几何统计
    # files 顺序就是 detect_perframe 的排序（jpg 后 png 同 detect）
    ord_f = sorted(files)
    used = [name2box.get(f.name) for f in ord_f]
    cov = sum(b is not None for b in used) / max(len(used), 1)
    bs = [b for b in used if b is not None]
    if not bs:
        print("❌ 该 clip 所有帧 bbox 都是 None（与 cache 不一致，检查 --box/clip_key 是否匹配）")
        sys.exit(1)
    areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in bs]
    cxs = [(b[0] + b[2]) / 2 for b in bs]
    cys = [(b[1] + b[3]) / 2 for b in bs]
    off = [np.hypot(cx - 0.5, cy - 0.5) for cx, cy in zip(cxs, cys)]
    disp = [np.hypot(cxs[i] - cxs[i - 1], cys[i] - cys[i - 1]) for i in range(1, len(cxs))]
    aw = [max(0.0, b[2] - b[0]) for b in bs]
    ah = [max(0.0, b[3] - b[1]) for b in bs]

    print("==== track bbox 质量统计 ====", flush=True)
    print(f"clip 帧数={len(ord_f)}  检测覆盖={cov:.3f} (要有框帧占比)", flush=True)
    print(f"box 面积占比: mean={np.mean(areas):.3f} min={np.min(areas):.3f} max={np.max(areas):.3f}", flush=True)
    print(f"box 宽占: mean={np.mean(aw):.3f} | 高占: mean={np.mean(ah):.3f}", flush=True)
    print(f"中心偏移(离画面中心): mean={np.mean(off):.3f} max={np.max(off):.3f}"
          f" (>0.2 说明人物严重偏心，裁剪易切)", flush=True)
    print(f"相邻帧中心位移: mean={np.mean(disp):.3f} max={np.max(disp):.3f}"
          f" (大=抖动/检测跳变，轨迹噪声)", flush=True)
    good = (cov > 0.9 and np.mean(areas) > 0.08 and np.mean(areas) < 0.6
            and np.mean(off) < 0.2)
    print(f"判读: {'✅ box 覆盖/大小/居中度可接受，可进入训练' if good else '⚠️ 需检查（见上指标）'}")

    # 2) 可视化：采样 6 帧原图叠框 + 裁剪后人物 + 轨迹
    import math
    sel = np.linspace(0, len(ord_f) - 1, 6).round().astype(int)
    tiles = []
    for i in sel:
        img = cv2.imread(str(ord_f[i]))
        b = name2box.get(ord_f[i].name)
        if img is not None:
            if img.shape[1] > 640:  # 缩放显示
                img = cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))
            H, W = img.shape[:2]
            if b is not None:
                x0, y0, x1, y1 = [int(v * W) if j % 2 == 0 else int(v * H) for j, v in enumerate(b)]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            tiles.append(img)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in tiles]
    grid = np.hstack(tiles)
    cv2.imwrite(str(out / "frames_boxes.jpg"), grid)
    print(f"图1 原图+框已存: {out / 'frames_boxes.jpg'}")

    # 轨迹折线（归一化坐标→画布）
    canvas = np.full((480, 640, 3), 30, np.uint8)
    if len(cxs) > 1:
        pts = [(int(cx * 640), int(cy * 480)) for cx, cy in zip(cxs, cys)]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 0), 1)
        cv2.circle(canvas, pts[0], 4, (0, 0, 255), -1)
        cv2.circle(canvas, pts[-1], 4, (255, 0, 0), -1)
    cv2.putText(canvas, f"trajectory (cov={cov:.2f})", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(out / "trajectory.jpg"), canvas)
    print(f"图2 轨迹已存: {out / 'trajectory.jpg'}")
    print("查看: view_image 打开 outputs/viz_track/frames_boxes.jpg / trajectory.jpg")


if __name__ == "__main__":
    main()
