"""CUHK-X —— NTU RGB+D 骨架数据集（外部补充，25→17 关节映射 + MotionBERT 对齐预处理）。

数据：data/external/ntu/ntu60/*.skeleton（NTU60，A001-A040 单人日常，37,920 样本）
格式：token 流多帧序列（见 scripts/ntu_skeleton_verify.py 解析验证），
      每帧 body 信息 11 token + 每关节 12 token（x,y,z 米制，Kinect 坐标系）
输出：x[T,17,3]（H3.6M-17 拓扑，中心化 + 肩宽归一化）+ ntu_class(0-39) + subject
预处理与 src/skeleton_dataset.py 对齐：中心化(减骨盆) + 肩宽归一化 + CG 几何增强。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# H3.6M-17 index → NTU-25 index（Kinect v2）
# 丢弃 NTU 手/脚/指尖等 8 个冗余关节：7,11,15,19,21,22,23,24
H36M_TO_NTU25 = {
    0: 0,    # Hip ← SpineBase
    1: 16,   # RHip ← HipRight
    2: 17,   # RKnee ← KneeRight
    3: 18,   # RAnkle ← AnkleRight
    4: 12,   # LHip ← HipLeft
    5: 13,   # LKnee ← KneeLeft
    6: 14,   # LAnkle ← AnkleLeft
    7: 1,    # Spine ← SpineMid
    8: 20,   # Thorax ← SpineShoulder
    9: 2,    # Neck ← Neck
    10: 3,   # Head ← Head
    11: 4,   # LShoulder ← ShoulderLeft
    12: 5,   # LElbow ← ElbowLeft
    13: 6,   # LWrist ← WristLeft
    14: 8,   # RShoulder ← ShoulderRight
    15: 9,   # RElbow ← ElbowRight
    16: 10,  # RWrist ← WristRight
}
SHOULDER_L, SHOULDER_R = 11, 14  # H3.6M 左右肩
# H3.6M-17 左右对称关节对（增强镜像用）
MIRROR_PAIRS = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3,
                11: 14, 12: 15, 13: 16, 14: 11, 15: 12, 16: 13}
FNAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")

# NTU 动作(1-based) → 我们 action_id（严格语义一致，清洗分类用）
NTU_TO_OURS = {
    1: 6,   # drink water → Drink_water
    2: 7,   # eat meal/snack → Eat_food
    3: 1,   # brushing teeth → Brush_teeth
    4: 2,   # brushing hair → Comb_hair
    8: 34,  # sitting down → Sit_down
    9: 32,  # standing up → Stand_up
    11: 21, # reading → Read_documents
    12: 18, # writing → Write
    28: 19, # make phone call → Make_a_phone_call
    29: 24, # playing phone → Use_a_mobile_phone
    30: 17, # typing → Tap_the_keyboard
    32: 27, # selfie → Take_a_selfie
    33: 20, # check time → Check_the_time
}


def parse_ntu_skeleton(path: Path) -> List[np.ndarray]:
    """解析 NTU .skeleton → 每帧第一个 body 的 25 关节 x,y,z → list[np[25,3]]。"""
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    frames = []
    i = 0

    def take(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    nframes = int(take(1)[0])
    for _ in range(nframes):
        nbody = int(take(1)[0])
        for b in range(nbody):
            info = take(11)
            nj = int(info[-1])
            joints = take(nj * 12)
            if b == 0:  # 取第一个 body（单人动作）
                arr = np.array(joints, np.float32).reshape(-1, 12)
                seq = np.zeros((25, 3), np.float32)
                n = min(25, arr.shape[0])
                seq[:n] = arr[:n, :3]
                frames.append(seq)
    return frames


class NTUSkeletonDataset(Dataset):
    """NTU 骨架 → [T,17,3]（H3.6M-17，与 MotionBertSkeletonDataset 对齐）。"""

    def __init__(self, root: str, num_frames: int = 16, is_train: bool = True,
                 seed: int = 0, max_action: int = 40, files=None, label_ours: bool = False):
        # 只保留单人日常类 A001-A040（A041+ 是双人交互类，标签空间不同）
        if files is not None:
            self.files = list(files)
        else:
            self.files = []
            for f in sorted(Path(root).glob("*.skeleton")):
                m = FNAME_RE.search(f.name)
                if m and int(m.group(5)) <= max_action:
                    self.files.append(f)
        # 清洗分类：label_ours=True → 只保留能映射到我们 40 类的 NTU 样本，标签=映射后我们类
        self.label_ours = label_ours
        if label_ours:
            self.files = [f for f in self.files
                          if int(FNAME_RE.search(f.name).group(5)) in NTU_TO_OURS]
        self.num_frames = num_frames
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def _sample(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(self.num_frames, dtype=int)
        return np.linspace(0, n - 1, self.num_frames).round().astype(int)

    def _augment(self, feat: np.ndarray) -> np.ndarray:
        """CG 几何增强（与 src/skeleton_dataset.py 对齐）：镜像 + 旋转 + 缩放 + 平移 + 抖动。"""
        r = self.rng
        if r.random() < 0.5:
            mirrored = feat.copy()
            for a, b in MIRROR_PAIRS.items():
                mirrored[:, a, :] = feat[:, b, :]
            mirrored[:, :, 0] = -mirrored[:, :, 0]
            feat = mirrored
        ang = float(r.uniform(-15.0, 15.0) * math.pi / 180.0)
        c, s = math.cos(ang), math.sin(ang)
        xy = feat[:, :, :2]
        feat[:, :, 0] = xy[:, :, 0] * c - xy[:, :, 1] * s
        feat[:, :, 1] = xy[:, :, 0] * s + xy[:, :, 1] * c
        sc = float(r.uniform(0.85, 1.15))
        feat[:, :, :2] *= sc
        feat[:, :, 2] *= sc
        feat[:, :, 0] += float(r.uniform(-0.08, 0.08))
        feat[:, :, 1] += float(r.uniform(-0.08, 0.08))
        jit = r.uniform(-0.03, 0.03, size=(feat.shape[0], feat.shape[1], 2)).astype(feat.dtype)
        feat[:, :, :2] += jit
        return feat

    def __getitem__(self, i: int):
        fname = self.files[i]
        frames = parse_ntu_skeleton(fname)
        F = len(frames)
        x = torch.zeros(self.num_frames, 17, 3, dtype=torch.float32)
        if F > 0:
            idx = self._sample(F)
            seq = np.stack([frames[t] for t in idx], 0)     # [T,25,3]
            h = np.zeros((self.num_frames, 17, 3), np.float32)
            for h_i, n_i in H36M_TO_NTU25.items():
                h[:, h_i] = seq[:, n_i]
            # 中心化（减骨盆 idx0）+ 肩宽归一化
            h = h - h[:, 0:1, :]
            shoulder = np.linalg.norm(h[:, SHOULDER_L] - h[:, SHOULDER_R], axis=-1)
            h = h / (shoulder[:, None, None] + 1e-8)
            if self.is_train:
                h = self._augment(h)
            x = torch.from_numpy(h)
        m = FNAME_RE.search(fname.name)
        ntu_action = int(m.group(5)) if m else 0
        if self.label_ours:
            label = NTU_TO_OURS[ntu_action]     # 映射到我们类（已过滤，必在表内）
        else:
            label = ntu_action - 1              # A001-A040 → 0-39
        return x, label, fname.name[:16]
