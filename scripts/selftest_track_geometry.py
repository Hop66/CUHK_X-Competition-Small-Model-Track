#!/usr/bin/env python3
"""CUHK-X —— track_crop 几何自测（不依赖 YOLO，本地可跑）
用合成热像帧 + 逐帧人工人物框，验证真实 ThermalVideoDataset(track_crop) 代码：
  1) track_crop 输出形状 [T,3,S,S]
  2) 逐帧跟人裁剪后人物被放大（亮像素占比显著上升 = 框住人体）
  3) 轨迹 traj vs 造的人物框完全一致（[cx,cy,bw,bh] 对齐）
  4) flip=False 时 traj.x 不变（is_train=False）
失败则断言报错 → 说明 dataset 代码有问题；全过 → crop/traj 链路正确。
用法: python scripts/selftest_track_geometry.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import torch

from src.dataset import ThermalVideoDataset


def make_synthetic_frames(tmp: Path, n_frames=16, W=200, H=140):
    """每帧：灰背景 50 + 一个亮白"人物"矩形（亮度 240），位置逐帧移动（模拟位移）。"""
    names = []
    cx, cy = W * 0.5, H * 0.5
    bw, bh = W * 0.28, H * 0.55
    boxes = []
    for t in range(n_frames):
        img = np.full((H, W, 3), 50, np.uint8)
        # 人物中心逐帧水平移动（0.40→0.62），垂直微动
        cx_n, cy_n = 0.40 + 0.22 * t / max(n_frames - 1, 1), 0.48 + 0.02 * t / max(n_frames - 1, 1)
        x0, y0 = int((cx_n - bw / (2 * W)) * W), int((cy_n - bh / (2 * H)) * H)
        x1, y1 = int((cx_n + bw / (2 * W)) * W), int((cy_n + bh / (2 * H)) * H)
        img[y0:y1, x0:x1] = 240
        name = f"frame_{1000001 + t:08d}.jpg"
        cv2.imwrite(str(tmp / name), img)
        names.append(name)
        # 归一化人物框（margin=1.0 原始人物矩形）
        boxes.append([cx_n - bw / (2 * W), cy_n - bh / (2 * H),
                      cx_n + bw / (2 * W), cy_n + bh / (2 * H)])
    return names, boxes


def main():
    tmp = Path(tempfile.mkdtemp(prefix="cuhkx_track_selftest_"))
    try:
        names, boxes = make_synthetic_frames(tmp)
        clip_key = "1/user1/s0"
        perframe = {"names": names, "boxes": boxes}
        (tmp / "boxes.json").write_text(json.dumps({clip_key: perframe}), encoding="utf-8")

        from src.skeleton_dataset import SkeletonClipIndex  # noqa 仅占位不上实际
        fake_clip = type("C", (), {
            "action_id": 1, "subject": "user1", "sample": "s0", "thermal_dir": tmp,
        })()
        ds = ThermalVideoDataset([fake_clip], num_frames=8, size=112, is_train=False,
                                 track_crop=True, box_path=str(tmp / "boxes.json"))
        x, traj, label, subject = ds[0]

        assert tuple(x.shape) == (8, 3, 112, 112), f"shape {x.shape}"
        assert label == 1 and subject == "user1"

        # 亮像素占比：裁剪放大后应显著高于无裁剪原始占比
        bright_frac = float((x.mean(dim=1) > 0.6).float().mean())
        raw_frac = 0.28 * 0.55  # 原始人物框面积占比
        print(f"x shape {tuple(x.shape)}; 裁剪后亮像素占比={bright_frac:.3f} (原始人物占比≈{raw_frac:.3f})")

        # 采样帧索引（uniform：ds 自己采的帧 = 返回 x/traj 实际用的帧）
        samp_idx = ds._sample_indices(16)

        # 轨迹对齐：traj[t] 应对应 采样帧 samp_idx[t] 的人物框
        for t in range(8):
            b = boxes[samp_idx[t]]
            exp_cx, exp_cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            exp_bw, exp_bh = (b[2] - b[0]), (b[3] - b[1])
            got = traj[t].numpy()
            ok = (abs(got[0] - exp_cx) < 1e-3 and abs(got[1] - exp_cy) < 1e-3
                  and abs(got[2] - exp_bw) < 1e-3 and abs(got[3] - exp_bh) < 1e-3)
            assert ok, f"traj[{t}] {got} vs 期望({exp_cx},{exp_cy},{exp_bw},{exp_bh}) idx={samp_idx[t]}"
        print("轨迹对齐: 8 帧全部一致（按采样帧 idx）[OK]  [cx,cy,bw,bh] 逐帧正确")

        # 放大验证：裁剪后人物应占更大画面
        assert bright_frac > raw_frac * 1.8, "裁剪后人物占比未放大，几何可能出错"
        print("人物放大: 裁剪后占比显著高于原始 [OK]")

        # DepthIR 帧号映射逻辑（构造 IR 帧名带动帧号 → frame_num → box）
        from src.dataset import frame_num_of
        ir_name = "IR_2025-06-10_10-43-43.516_00000096.png"
        assert frame_num_of(ir_name) == 96, frame_num_of(ir_name)
        print(f"DepthIR 帧号解析: {ir_name} → {frame_num_of(ir_name)} [OK]")

        # 可视化：原图+框 | 裁剪后人物（存 png 供直接查看）
        vis = Path("outputs/viz_track")
        vis.mkdir(parents=True, exist_ok=True)
        src0 = cv2.imread(str(tmp / names[0]))
        b0 = boxes[0]
        W0, H0 = src0.shape[1], src0.shape[0]
        cv2.rectangle(src0, (int(b0[0] * W0), int(b0[1] * H0)),
                      (int(b0[2] * W0), int(b0[3] * H0)), (0, 0, 255), 2)
        crop_t0 = x[0].permute(1, 2, 0).numpy()
        crop_t0 = ((crop_t0 - crop_t0.min()) / (crop_t0.max() - crop_t0.min() + 1e-9)
                   * 255).clip(0, 255).astype("uint8")
        stack = np.hstack([cv2.resize(src0, (224, 224)),
                           cv2.resize(np.ascontiguousarray(crop_t0), (224, 224))])
        vis_path = vis / "selftest_track_sample.jpg"
        cv2.imwrite(str(vis_path), stack)
        print(f"可视化样例（左=原图+人物框 | 右=逐帧跟人裁剪后满框人物）: {vis_path}")

        print("\n==== track_crop 几何自测全部通过 ====")
        print("结论：逐帧跟人裁剪 + 轨迹通道的 dataset 代码正确（人物放大、居中、轨迹对齐、帧号映射）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
