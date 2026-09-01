#!/usr/bin/env python3
"""P1-1a：骨架 2D 投影可行性探索（决定"姿态热图早融合"能否做）

回答三个问题：
  1. 骨架 json 结构：keypoints 是几维？坐标量级（米制/像素/归一化）？是否含 2D/bbox/conf？
  2. 骨架与 IR 是否同帧号对齐（同相机）？IR 图像尺寸？
  3. 3D 米制坐标能否投影到 IR 图像（需要相机内参或近似方案）？

用法（服务器 CPU 即可）:
  CUDA_VISIBLE_DEVICES="" python scripts/explore_skeleton_2d.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def frame_num_of(name: str):
    m = re.search(r"(\d{8})", name)
    return int(m.group(1)) if m else None


def list_images(d: Path):
    """返回 {帧号: 路径}，兼容多种命名。"""
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".jpg", ".png"):
            m = re.search(r"(\d{8})", p.name)
            out[int(m.group(1))] = p if m else out.setdefault(hash(p.name), p)
    return out


def main():
    root = Path("~/Multimodal/data/Training/HAR").expanduser()
    skel = root / "Skeleton"
    ir = root / "IR"

    # 找 2 个 sample（不同动作）看结构
    samples = []
    for action in sorted(skel.iterdir())[:3]:
        if not action.is_dir():
            continue
        for subj in sorted(action.iterdir())[:2]:
            if not subj.is_dir():
                continue
            for s in sorted(subj.iterdir()):
                if s.is_dir():
                    samples.append((action, subj, s))
                    break
            break

    for action, subj, sample in samples:
        pred = sample / "predictions"
        files = sorted(pred.glob("*.json"))
        print(f"\n===== {action.name}/{subj.name}/{sample.name} =====", flush=True)
        print(f"骨架 json 数: {len(files)}", flush=True)
        if not files:
            print("  ❌ 无骨架 json，跳过", flush=True)
            continue

        # 1) json 结构
        data = json.loads(files[0].read_text(encoding="utf-8"))
        fr = data[0] if isinstance(data, list) else data
        print(f"json keys: {list(fr.keys())}", flush=True)
        kp = np.asarray(fr["keypoints"], dtype=np.float32)
        print(f"keypoints shape: {kp.shape}", flush=True)
        # 坐标统计（前 3 维）
        kp3 = kp[..., :3]
        print(f"  x 范围 [{kp3[:,0].min():.3f}, {kp3[:,0].max():.3f}] 均值 {kp3[:,0].mean():.3f}", flush=True)
        print(f"  y 范围 [{kp3[:,1].min():.3f}, {kp3[:,1].max():.3f}] 均值 {kp3[:,1].mean():.3f}", flush=True)
        print(f"  z 范围 [{kp3[:,2].min():.3f}, {kp3[:,2].max():.3f}] 均值 {kp3[:,2].mean():.3f}", flush=True)
        print(f"  量级判断: max|coord|={np.abs(kp3).max():.2f} → "
              f"{'米制(~1)' if np.abs(kp3).max() < 10 else '像素(100+)/其他'}", flush=True)
        if "keypoint_scores" in fr:
            ks = np.asarray(fr["keypoint_scores"], np.float32)
            print(f"  keypoint_scores: min={ks.min():.3f} max={ks.max():.3f} mean={ks.mean():.3f}", flush=True)
        # 额外字段
        for k, v in fr.items():
            if k not in ("keypoints", "keypoint_scores"):
                print(f"  额外字段 {k}: {type(v).__name__} "
                      f"{np.asarray(v).shape if hasattr(v, '__len__') else v}", flush=True)

        # 2) 与 IR 对齐
        ir_dir = ir / action.name / subj.name / sample.name
        ir_map = list_images(ir_dir) if ir_dir.is_dir() else {}
        skel_frames = sorted(frame_num_of(f.name) for f in files if frame_num_of(f.name))
        print(f"IR 图像数: {len(ir_map)}", flush=True)
        if ir_map:
            first_ir = next(iter(ir_map.values()))
            import cv2
            img = cv2.imread(str(first_ir), cv2.IMREAD_UNCHANGED)
            print(f"IR 图像尺寸: {img.shape if img is not None else '读取失败'}", flush=True)
        common = set(ir_map) & set(skel_frames)
        print(f"骨架帧范围: [{min(skel_frames) if skel_frames else 'NA'}, {max(skel_frames) if skel_frames else 'NA'}]", flush=True)
        print(f"IR 帧范围: [{min(ir_map) if ir_map else 'NA'}, {max(ir_map) if ir_map else 'NA'}]", flush=True)
        print(f"✅ 帧号交集 {len(common)}（>0 = 同相机同帧对齐，可做热图早融合）" if common
              else "❌ 帧号无交集（需检查命名）", flush=True)

    print("\n==== 探索完成 ====", flush=True)
    print("结论判读: 量级~1=米制3D(需投影); 若帧号交集大+有2D/bbox → 直接画热图", flush=True)


if __name__ == "__main__":
    main()
