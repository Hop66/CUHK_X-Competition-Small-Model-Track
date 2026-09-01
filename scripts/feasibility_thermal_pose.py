#!/usr/bin/env python3
"""CUHK-X —— 热像姿态可行性门禁测试（"双视角骨架桥"的第一步）

问题：OpenPose 类姿态估计算法在铁火伪彩热像（320x240）上能否检测到人+关键点？
这是"NYX 骨架 + 热像 2D 骨架 双视角输入"整条线的命门，必须先测。

方法：torchvision KeypointRCNN（COCO 17 关键点，姿态估计器）在若干热像 sample 上
检测 person，统计：人检测率 / 关键点≥10 的帧占比 / 平均置信度，并保存可视化。
判读：
  - 人检测率>90% 且关键点完整 → 门禁通过，值得建双视角骨架模型
  - 检测率低/关键点残缺 → 铁火伪彩对姿态估计是硬伤，整条线不投入

用法（服务器，GPU）:
  python scripts/feasibility_thermal_pose.py --root ~/Multimodal/data/Training/HAR
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

BONES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
         (11,12),(11,13),(13,15),(12,14),(14,16)]  # COCO 17 骨架


def load_frames(thermal_dir: Path):
    files = sorted(thermal_dir.glob("*.jpg")) + sorted(thermal_dir.glob("*.png"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--n_clips", type=int, default=6, help="测试多少个 sample（不同动作）")
    ap.add_argument("--out", default="outputs/thermal_pose_feas")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    import torchvision
    from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                              keypointrcnn_resnet50_fpn)
    print(f"[pose] torchvision {torchvision.__version__}，加载 KeypointRCNN（COCO 17 关键点）...",
          flush=True)
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval().to(args.device)
    # 归一化约定：<0.15 的 GeneralizedRCNNTransform 内部会再 /255，需喂 [0,255]；
    # >=0.15 期望 [0,1]（脚本默认 /255 正确）。旧版本自动切换。
    tv_major, tv_minor = (int(x) for x in torchvision.__version__.split(".")[:2])
    self_norm = (tv_major, tv_minor) >= (0, 15)
    print(f"[pose] 归一化模式: {'[0,1]（自归一化，>=0.15）' if self_norm else '[0,255]（旧版，内部归一化）'}",
          flush=True)

    # 取 n_clips 个不同动作的 sample
    clips = []
    for action in sorted(root.glob("Thermal/*"))[:args.n_clips]:
        if not action.is_dir():
            continue
        for subj in sorted(action.iterdir())[:1]:
            for sample in sorted(subj.iterdir()):
                if sample.is_dir():
                    clips.append((action.name, subj.name, sample.name, sample))
                    break
    print(f"[pose] 测试 {len(clips)} 个 sample", flush=True)

    det_rate_all, kp_rate_all, conf_all = [], [], []
    for ai, (action, subj, sample, sdir) in enumerate(clips):
        files = load_frames(sdir)
        n_frames = min(len(files), 10)
        det = kp_ok = 0
        confs = []
        for j in range(0, n_frames, 2):  # 每 2 帧测 1 帧，省时间
            img = cv2.imread(str(files[j]), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.dtype == np.uint16:  # 热像可能是 16bit
                img = (img / 65535.0 * 255.0).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 归一化：>=0.15 喂 [0,1]；旧版喂 [0,255]（内部再归一化）
            if self_norm:
                arr = img.astype(np.float32) / 255.0
            else:
                arr = img.astype(np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred = model(tensor)[0]
            scores = pred["scores"].cpu().numpy()
            kpts = pred["keypoints"].cpu().numpy()
            ksc = pred["keypoints_scores"].cpu().numpy()
            if len(scores) > 0 and scores[0] > 0.5:
                det += 1
                k = ksc[0]
                confs.append(float(k.max()))
                if (k > 0.5).sum() >= 10:
                    kp_ok += 1
                # 第一帧画图
                if j == 0:
                    vis = img.copy()
                    x, y = kpts[0][:, :2].astype(int).T
                    for a, b in BONES:
                        if (ksc[0][a] > 0.3 and ksc[0][b] > 0.3):
                            cv2.line(vis, tuple(kpts[0][a][:2].astype(int)),
                                     tuple(kpts[0][b][:2].astype(int)), (255, 0, 0), 2)
                    for xi, yi in zip(x, y):
                        cv2.circle(vis, (int(xi), int(yi)), 3, (0, 255, 0), -1)
                    cv2.imwrite(str(out / f"{action.replace(' ','_')}_{sample}_pose.jpg"), vis)
        n_t = max(n_frames // 2, 1)
        dr = det / n_t
        kr = kp_ok / max(det, 1)
        det_rate_all.append(dr)
        kp_rate_all.append(kr)
        if confs:
            conf_all.append(float(np.mean(confs)))
        print(f"  [{ai+1}/{len(clips)}] {action}/{sample}: 人检测率={dr:.2f} "
              f"骨架完整率={kr:.2f} 关键点均置信={np.mean(confs) if confs else 0:.2f}", flush=True)

    print("\n==== 热像姿态可行性汇总 ====", flush=True)
    print(f"  人检测率 mean = {np.mean(det_rate_all):.3f}", flush=True)
    print(f"  骨架完整(≥10kp)率 mean = {np.mean(kp_rate_all):.3f}", flush=True)
    print(f"  关键点置信度 mean = {np.mean(conf_all):.3f}", flush=True)
    print("判读:", flush=True)
    print("  人检测>0.9 且 骨架完整率>0.7 → 门禁通过，可建 NYX+热像 双视角骨架模型", flush=True)
    print("  否则 → 铁火伪彩对姿态估计是硬伤，双视角线不投入", flush=True)


if __name__ == "__main__":
    main()
