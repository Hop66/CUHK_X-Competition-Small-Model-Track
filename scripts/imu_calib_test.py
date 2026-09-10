#!/usr/bin/env python3
"""IMU 推理侧幅值校准验证: 根因=test 角速度幅值 = train×1.63.
把 test 输入 gyro 通道乘 scale=(train_幅/train测 幅)=15.68/25.64≈0.61 再推, 对比校准前后:
  - IMU test softmax 置信(近随机=0.03-0.1? 之前 0.1-0.4)
  - IMU 与强 teacher(main+th) 的一致率(回升=根因打中)
"""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.imu_dataset import IMUDataset, IMUClipIndex
from train_imu import IMUCNN


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    dev = "cpu"
    troot = Path("data/Testing/data/small_model_track_test")
    clip_ids = [d.name for d in sorted(troot.iterdir()) if d.is_dir() and d.name.startswith("SM_test_")]
    clips = [IMUClipIndex(-1, c, c, troot / c / "IMU") for c in clip_ids]
    ds = IMUDataset(clips, 128, False)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    raw = pickle.load(open("outputs/test_teacher_avg_probs.pkl", "rb"))
    teacher = {re.sub(r"^test:", "", k): np.asarray(v, np.float32) for k, v in raw.items()}
    tgt = np.stack([teacher[c] for c in clip_ids])

    model = IMUCNN().to(dev).eval()
    acc = np.zeros((len(clip_ids), 40), np.float32)
    for ck in ("imu_fold0.pth", "imu_fold1.pth", "imu_fold2.pth"):
        sd = torch.load(f"outputs/imu/{ck}", map_location=dev)
        model.load_state_dict(sd["model"])
        lg = []
        for x, _, _ in loader:
            lg.append(model(x.to(dev)).detach().cpu().numpy())
        lg = np.concatenate(lg)
        acc += sf(lg) / 3

    conf = acc.max(1).mean()
    agree = (acc.argmax(1) == tgt.argmax(1)).mean()
    print(f"校准前: test IMU max置信={conf:.3f}  ↔teacher一致率={agree:.3f}")

    # 校准: gyro 通道(每设备 3::6)乘 scale
    SCALE = 15.68 / 25.64  # ≈0.612(以 train 幅值为目标)
    acc2 = np.zeros((len(clip_ids), 40), np.float32)
    for ck in ("imu_fold0.pth", "imu_fold1.pth", "imu_fold2.pth"):
        sd = torch.load(f"outputs/imu/{ck}", map_location=dev)
        model.load_state_dict(sd["model"])
        lg = []
        s = 0
        for x, _, _ in loader:
            xc = x.clone()
            xc[:, :, 3::6] = xc[:, :, 3::6] * SCALE  # gyro 通道缩放
            lg.append(model(xc.to(dev)).detach().cpu().numpy())
        acc2 += sf(np.concatenate(lg)) / 3
    conf2 = acc2.max(1).mean()
    agree2 = (acc2.argmax(1) == tgt.argmax(1)).mean()
    print(f"校准后: test IMU max置信={conf2:.3f}  ↔teacher一致率={agree2:.3f}")


if __name__ == "__main__":
    main()
