"""CUHK-X —— 3D 骨架辅助监督（视频模型回归对齐的原生 3D 骨架，multi-task "3D 加强 2D"）

机制：视频模型除动作 CE 外，另加小头回归同一相机逐帧对齐的原生 3D 骨架（MPJPE）
  → 模型被迫编码显式人体几何，把"动作"从"外观/身份"解耦 → 提升 cross-subject。
前提（本数据集天然满足）：Skeleton 与 Depth/IR 同相机、帧号一致 → 逐帧监督免费。
测试时摘掉位姿头 → 推理零开销、模型大小不变。

用法（配合 scripts/train_auxpose.py）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import DepthIRVideoDataset, ThermalVideoDataset, list_images
from src.skeleton_dataset import SkeletonClipIndex, frame_num_of

SHOULDER_L, SHOULDER_R = 11, 14  # H3.6M-17 左右肩
# H3.6M-17 左右交换（镜像用）：(1,4)(2,5)(3,6) 腿，(11,14)(12,15)(13,16) 臂；与 MIRROR_PAIRS 一致
H36M_FLIP_IDX = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def load_skeleton_map(pred_dir: Path) -> Dict[int, np.ndarray]:
    """{帧号: [17,3] 原始坐标}（与视频同帧号对齐）。"""
    out: Dict[int, np.ndarray] = {}
    if not pred_dir.is_dir():
        return out
    for f in pred_dir.glob("*.json"):
        fn = frame_num_of(f.name)
        if fn is None:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        fr = data[0] if isinstance(data, list) else data
        if not isinstance(fr, dict) or "keypoints" not in fr:
            continue
        out[fn] = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
    return out


def normalize3d(kp: np.ndarray) -> np.ndarray:
    """[T,17,3] 原始[水平,深度,垂直] → [水平,垂直,深度]=[0,2,1]，中心化 + 肩宽归一化。

    与 MotionBertSkeletonDataset input3d=True 一致，保证骨架分支/监督同坐标系。
    """
    xyz = kp[:, :, [0, 2, 1]].astype(np.float32)
    xyz = xyz - xyz[:, 0:1, :]  # 中心化（减骨盆）
    sh = np.linalg.norm(xyz[:, SHOULDER_L] - xyz[:, SHOULDER_R], axis=-1)
    return xyz / (sh[:, None, None] + 1e-8)


class PoseAlignedVideoDataset(Dataset):
    """视频 + 对齐骨架配对：[T,4,H,W]（与 DepthIRVideoDataset 同款加载/增强）
    + [T,17,3] 原生 3D 骨架（中心化归一化，与视频采样帧对齐）+ [T] 缺失掩码。

    返回 (x, skel, mask, action_id, subject)：
      - skel[t]=该视频帧对应的原生 3D 骨架（缺失帧为 0，靠 mask 排除）
      - mask[t]=1 表示该帧有骨架真值（避免零目标污染梯度）
    """

    def __init__(self, clips, skel_clips, num_frames: int = 16, size: int = 128,
                 is_train: bool = True, crop_cache=None, aug_strength: int = 2,
                 seed: int = 0):
        self.clips = clips
        self.skel_clips = skel_clips
        self.num_frames = num_frames
        self.size = size
        self.is_train = is_train
        self.crop_cache = crop_cache or {}
        self.rng = np.random.default_rng(seed)
        # 视频加载/增强复用 DepthIRVideoDataset 的 _load / _uniform_indices / aug 表
        self.video = DepthIRVideoDataset(clips, num_frames, size, is_train, crop_cache,
                                         use_ir_mask=False, use_frame_diff=False,
                                         aug_strength=aug_strength, seed=seed)
        mean3 = (0.43216, 0.394666, 0.37645)
        std3 = (0.22803, 0.22145, 0.216989)
        self.mean = torch.tensor((*mean3, sum(mean3) / 3.0)).view(1, 4, 1, 1)
        self.std = torch.tensor((*std3, sum(std3) / 3.0)).view(1, 4, 1, 1)

    def __len__(self):
        return len(self.clips)

    @staticmethod
    def _clip_key(clip):
        return f"{clip.action_id}/{clip.subject}/{clip.sample}"

    def __getitem__(self, i: int):
        if self.is_train:
            # DataLoader fork 复制同一 self.rng → 各 worker 同序列；worker 种源+样本序号重派生
            self.rng = np.random.default_rng(int((torch.initial_seed() + i) & 0x7FFFFFFF))
        clip = self.clips[i]
        skel_dir = self.skel_clips[i].pred_dir
        depth_map = list_images(clip.depth_dir)
        ir_map = list_images(clip.ir_dir)
        common = sorted(set(depth_map) & set(ir_map))
        if not common:
            common = sorted(depth_map.keys()) or sorted(ir_map.keys())
        idx = self.video._uniform_indices(len(common))
        picked = [common[j] for j in idx]

        crop = self.crop_cache.get(self._clip_key(clip))
        do_flip = False
        bright, contrast = 1.0, 1.0
        if self.is_train:
            a = self.video
            do_flip = self.rng.random() < a.aug_flip
            bright = float(self.rng.uniform(*a.aug_bright))
            contrast = float(self.rng.uniform(*a.aug_contrast))
            if crop is not None:
                s = float(self.rng.uniform(*a.aug_scale))
                dx = float(self.rng.uniform(-a.aug_shift, a.aug_shift))
                dy = float(self.rng.uniform(-a.aug_shift, a.aug_shift))
                x1, y1, x2, y2 = crop
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nw, nh = w * s, h * s
                crop = (max(cx - nw / 2.0 + dx * w, 0.0), max(cy - nh / 2.0 + dy * h, 0.0),
                        min(cx + nw / 2.0 + dx * w, 1.0), min(cy + nh / 2.0 + dy * h, 1.0))

        frames = np.zeros((self.num_frames, 4, self.size, self.size), np.float32)
        for t, fn in enumerate(picked):
            d = self.video._load(depth_map.get(fn), 3, crop)
            r = self.video._load(ir_map.get(fn), 1, crop)
            if d is None:
                d = np.zeros((self.size, self.size, 3), np.float32)
            if r is None:
                r = np.zeros((self.size, self.size, 1), np.float32)
            frames[t] = np.concatenate([d, r], axis=2).transpose(2, 0, 1) / 255.0
        x = torch.from_numpy(frames)
        if self.is_train:
            x = x * bright
            x = (x - 0.5) * contrast + 0.5
            x = torch.clamp(x, 0.0, 1.0)
        if do_flip:
            x = torch.flip(x, dims=(3,))
        x = (x - self.mean) / self.std

        # 骨架目标：按 picked 帧对齐 → 原生 3D（含缺失掩码，避免零目标污染）
        skel_map = load_skeleton_map(skel_dir)
        kps, mask = [], []
        for fn in picked:
            k = skel_map.get(fn)
            if k is not None:
                kps.append(k)
                mask.append(1.0)
            else:
                kps.append(np.zeros((17, 3), np.float32))
                mask.append(0.0)
        kp = normalize3d(np.stack(kps, 0))                     # [T,17,3]
        if do_flip:
            # 修复：图像已水平翻转 → 骨架监督同步镜像（x 取反 + 左右关节交换），否则 ~50% 样本监督矛盾
            kp[..., 0] *= -1.0
            kp = kp[:, H36M_FLIP_IDX, :]
        skel = torch.from_numpy(kp)                            # [T,17,3]
        mask_t = torch.from_numpy(np.asarray(mask, np.float32))  # [T]
        return x, skel, mask_t, clip.action_id, clip.subject


class ThermalPoseAlignedVideoDataset(ThermalVideoDataset):
    """Thermal 视频 + 时间比例对齐的 3D 骨架监督（跨相机 auxpose）。

    Thermal（Hikvision ~25fps）与 Skeleton（NYX ~10fps）是**不同相机**、帧号不同基准，
    但 global time 同步 → 按帧比例对齐：thermal 第 j 帧 ≈ skeleton 序列第 round(j·(n_s-1)/(n_t-1)) 帧。
    视频复用父类 ThermalVideoDataset（同款加载/增强/归一化）；骨架缺失帧 mask=0 防污染。

    返回 (x [T,3,H,W], skel [T,17,3], mask [T], action_id, subject)。
    """

    def __init__(self, clips, skel_clips, num_frames: int = 16, size: int = 128,
                 is_train: bool = True, crop_cache=None, aug_strength: int = 2,
                 seed: int = 0):
        super().__init__(clips, num_frames, size, is_train, crop_cache,
                         use_frame_diff=False, aug_strength=aug_strength, seed=seed)
        self.skel_clips = skel_clips

    def __getitem__(self, i: int):
        clip = self.clips[i]
        skel_dir = self.skel_clips[i].pred_dir
        x, _, _ = super().__getitem__(i)  # [T,3,H,W] 视频（增强+归一化）
        files = sorted(clip.thermal_dir.glob("*.jpg")) + sorted(clip.thermal_dir.glob("*.png"))
        n_t = len(files)
        idx = self._sample_indices(n_t)
        skel_map = load_skeleton_map(skel_dir)
        skel_frames = sorted(skel_map.keys())
        n_s = len(skel_frames)
        kps, mask = [], []
        for j in idx:
            if n_s == 0:
                kps.append(np.zeros((17, 3), np.float32))
                mask.append(0.0)
            else:
                pos = int(round(j * (n_s - 1) / max(n_t - 1, 1)))
                fn = skel_frames[min(max(pos, 0), n_s - 1)]
                kps.append(skel_map[fn])
                mask.append(1.0)
        if kps:
            kp = normalize3d(np.stack(kps, 0))                 # [T,17,3] 原生 3D 归一化
        else:
            kp = np.zeros((self.num_frames, 17, 3), np.float32)
        if getattr(self, "_last_do_flip", False):
            # 父类已翻转视频 → 骨架监督同步镜像（x 取反 + 左右关节交换），否则 ~50% 样本监督矛盾
            kp[..., 0] *= -1.0
            kp = kp[:, H36M_FLIP_IDX, :]
        skel = torch.from_numpy(kp)
        mask_t = torch.from_numpy(np.asarray(mask, np.float32))
        return x, skel, mask_t, clip.action_id, clip.subject
