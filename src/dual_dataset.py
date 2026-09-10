"""DualSkeletonDataset —— 双视角骨架数据集（热像 2D [x,y,conf] + NYX 3D [x,y,z]）。

对齐：同一 clip（action_id/subject/sample）两源天然同步（七模态同步 + global time 对齐）。
  * 3D: <root>/Skeleton/<Action>/<Subject>/<sample>/predictions/*.json（H36M-17 顺序，米制）
  * 2D: <thermal_out>/<action_id>/<subject>/<sample>.npz
        含 kp[17,2]（H36M 顺序，root-relative + 肩宽归一化）+ conf[17]（KeypointRCNN 原始分数）
过滤：两源都齐全的 clip 才保留（提取 skip 的 18 个 clip 会被滤掉，避免污染训练）。

conf 归一化（顶会结论：非退化则 [x,y,conf] + 可用；统计 std=4.11 非退化）：
  * sigmoid（默认）：KeypointRCNN 原始分数（logit 尺度，-6.87~25）→ (0,1) 概率域，
    贴近 MotionBERT 2D 预训练输入（[x,y,confidence]）。
  * minmax：全局统计后线性映射到 [0,1]。
  * none：恒 1.0（对照，退化等价）。

增强（顶会 cross-view 标准，补相机视点差）：
  * 3D 分支：左右镜像 + 绕竖直轴(y)随机旋转（关键！补两相机方位角差）+ 缩放 + 平移 + 抖动
  * 2D 分支：左右镜像 + xy 平面旋转 ±15° + 缩放 + 平移 + 抖动（conf 不动）

输出: (x_2d [T,17,3], x_3d [T,17,3], action_id, subject)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.skeleton_dataset import (MIRROR_PAIRS, SHOULDER_L, SHOULDER_R,
                                  SkeletonClipIndex, build_skeleton_index,
                                  frame_num_of)

# H3.6M-17 顺序: 0骨盆 1-6腿 7脊柱 8颈 9鼻 10头 11-13左臂 14-16右臂


def _conf_norm_sigmoid(c: np.ndarray) -> np.ndarray:
    """原始分数 → sigmoid → (0,1)，单调且无参。"""
    return 1.0 / (1.0 + np.exp(-c.astype(np.float32)))


class DualSkeletonDataset(Dataset):
    def __init__(self, clips: List[SkeletonClipIndex], thermal_out: Path,
                 num_frames: int = 16, is_train: bool = True, seed: int = 0,
                 conf_norm: str = "sigmoid", rot_angle: float = 30.0,
                 conf_global: Optional[Tuple[float, float]] = None):
        """clips: build_skeleton_index 结果（已滤掉热像 npz 缺失的 clip，见 build_dual_pairs）。

        conf_global: conf_norm=minmax 时的全局 (min,max)（build_dual_pairs 返回）。
        """
        self.clips = clips
        self.thermal_out = Path(thermal_out)
        self.num_frames = num_frames
        self.is_train = is_train
        self.conf_norm = conf_norm
        self.rot_angle = rot_angle
        self._conf_global = conf_global or (0.0, 1.0)
        self.rng = np.random.default_rng(seed)
        # 每个 clip 的热像 npz 路径（与 3D 用同一 action_id/subject/sample 对齐）
        self.npz_paths = [
            self.thermal_out / f"{c.action_id}/{c.subject}/{c.sample}.npz" for c in clips]

    # ------------------------------------------------------------------ loaders
    def _load_3d(self, pred_dir: Path) -> np.ndarray:
        """读 Skeleton predictions -> [N,17,3] 原始 [水平, 深度, 垂直]。"""
        if not pred_dir.is_dir():
            return np.zeros((0, 17, 3), np.float32)
        files = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
        kps = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            frames = data if isinstance(data, list) else [data]
            for fr in frames:
                if not isinstance(fr, dict) or "keypoints" not in fr:
                    continue
                kps.append(np.asarray(fr["keypoints"], dtype=np.float32).reshape(17, 3))
        if not kps:
            return np.zeros((0, 17, 3), np.float32)
        return np.stack(kps, 0)

    def _load_2d(self, npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """读热像 npz -> (kp [N,17,2] 已归一化, conf [N,17] 原始分数)。"""
        if not npz_path.is_file():
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        z = np.load(npz_path)
        return z["kp"].astype(np.float32), z["conf"].astype(np.float32)

    # ------------------------------------------------------------------ sample
    def _uniform(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(self.num_frames, dtype=int)
        if n <= self.num_frames:
            return np.linspace(0, max(n - 1, 0), self.num_frames).round().astype(int)
        return np.linspace(0, n - 1, self.num_frames).round().astype(int)

    # ------------------------------------------------------------------ aug
    @staticmethod
    def _mirror(feat: np.ndarray) -> np.ndarray:
        """左右镜像（x 取反 + 交换左右关节对），feat 最后一维为坐标（conf 在输入前已分离）。"""
        out = feat.copy()
        for a, b in MIRROR_PAIRS.items():
            out[:, a, :] = feat[:, b, :]
        out[:, :, 0] = -out[:, :, 0]
        return out

    def _aug_3d(self, feat: np.ndarray) -> np.ndarray:
        """3D 分支增强：镜像 + 绕竖直轴(y)旋转（补相机视点差）+ 缩放 + 平移 + 抖动。"""
        r = self.rng
        if r.random() < 0.5:
            feat = self._mirror(feat)
        # 绕竖直轴(y)随机旋转 ±rot_angle（x,z 平面旋转）—— NTU cross-view 标准做法
        ang = float(r.uniform(-self.rot_angle, self.rot_angle) * math.pi / 180.0)
        c, s = math.cos(ang), math.sin(ang)
        x = feat[..., 0].copy()
        z = feat[..., 2].copy()
        feat[..., 0] = x * c + z * s
        feat[..., 2] = -x * s + z * c
        # 随机缩放（3D 三通道同尺度，模拟体型/距离）
        sc = float(r.uniform(0.85, 1.15))
        feat *= sc
        # 随机平移（xy）
        feat[..., 0] += float(r.uniform(-0.08, 0.08))
        feat[..., 1] += float(r.uniform(-0.08, 0.08))
        # 帧级抖动（xy）
        jit = r.uniform(-0.03, 0.03, size=(feat.shape[0], feat.shape[1], 2)).astype(feat.dtype)
        feat[..., :2] += jit
        return feat

    def _aug_2d(self, feat: np.ndarray) -> np.ndarray:
        """2D 分支增强：镜像 + xy 平面旋转 + 缩放 + 平移 + 抖动（conf 通道不动）。"""
        r = self.rng
        if r.random() < 0.5:
            feat = self._mirror(feat)
        ang = float(r.uniform(-15.0, 15.0) * math.pi / 180.0)
        c, s = math.cos(ang), math.sin(ang)
        xy = feat[..., :2].copy()
        feat[..., 0] = xy[..., 0] * c - xy[..., 1] * s
        feat[..., 1] = xy[..., 0] * s + xy[..., 1] * c
        sc = float(r.uniform(0.85, 1.15))
        feat[..., :2] *= sc
        feat[..., 0] += float(r.uniform(-0.08, 0.08))
        feat[..., 1] += float(r.uniform(-0.08, 0.08))
        jit = r.uniform(-0.03, 0.03, size=(feat.shape[0], feat.shape[1], 2)).astype(feat.dtype)
        feat[..., :2] += jit
        return feat

    # ------------------------------------------------------------------ item
    def __getitem__(self, i: int):
        clip = self.clips[i]
        npz = self.npz_paths[i]

        # ---- 3D 分支 [x,y,z]（匹配 model_pos 3D 预训练：原始 [水平,深度,垂直] → [水平,垂直,深度]）
        kp3 = self._load_3d(clip.pred_dir)
        if kp3.shape[0] == 0:
            x3 = torch.zeros(self.num_frames, 17, 3, dtype=torch.float32)
        else:
            idx = self._uniform(kp3.shape[0])
            kp3 = kp3[idx]
            xyz = kp3[:, :, [0, 2, 1]].astype(np.float32)      # [水平,垂直,深度]
            xyz = xyz - xyz[:, 0:1, :]                          # root-relative（减骨盆）
            sh = np.linalg.norm(xyz[:, SHOULDER_L] - xyz[:, SHOULDER_R], axis=-1)
            xyz = xyz / (sh[:, None, None] + 1e-8)              # 肩宽归一化
            x3 = xyz
            if self.is_train:
                x3 = self._aug_3d(x3)
            x3 = torch.from_numpy(x3)

        # ---- 2D 分支 [x,y,conf]（kp 已归一化；conf 原始分数 → (0,1)）
        kp2, conf = self._load_2d(npz)
        if kp2.shape[0] == 0:
            x2 = torch.zeros(self.num_frames, 17, 3, dtype=torch.float32)
        else:
            idx = self._uniform(kp2.shape[0])
            kp2 = kp2[idx]
            conf = conf[idx]
            # 硬 mask 必须在归一化前判定：原始 conf==0（失效关节/占位帧）；
            # 归一化后判会被 sigmoid(0)=0.5 吞掉，mask 永不触发
            _z = conf <= 1e-6
            if self.conf_norm == "sigmoid":
                conf = _conf_norm_sigmoid(conf)
            elif self.conf_norm == "none":
                conf = np.ones_like(conf)
            elif self.conf_norm == "minmax":
                lo, hi = self._conf_global[0], self._conf_global[1]
                conf = np.clip((conf - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
            else:
                raise ValueError(f"unknown conf_norm={self.conf_norm}")
            conf[_z] = 0.0
            kp2[_z] = 0.0
            x2 = np.concatenate([kp2, conf[..., None]], axis=-1).astype(np.float32)
            if self.is_train:
                x2 = self._aug_2d(x2)
            x2 = torch.from_numpy(x2)

        return x2, x3, clip.action_id, clip.subject

    def __len__(self):
        return len(self.clips)


class Thermal2DSkeletonDataset(DualSkeletonDataset):
    """热像 2D 单分支数据集（诊断用：确定热像 2D 骨架的动作识别天花板）。

    继承 DualSkeletonDataset 复用 _load_2d/_uniform/_aug_2d/_mirror（与双分支 2D 分支同协议同增强）。
    读 npz kp[17,2]+conf[17]（H36M 顺序，已 root-relative+肩宽归一化）→ [T,17,3] [x,y,conf]。
    输出: (x_2d [T,17,3], action_id, subject)
    """

    def __init__(self, clips: List[SkeletonClipIndex], thermal_out: Path,
                 num_frames: int = 16, is_train: bool = True, seed: int = 0,
                 conf_norm: str = "sigmoid"):
        super().__init__(clips, thermal_out, num_frames, is_train, seed,
                         conf_norm=conf_norm, rot_angle=30.0)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        kp2, conf = self._load_2d(self.npz_paths[i])
        if kp2.shape[0] == 0:
            x2 = torch.zeros(self.num_frames, 17, 3, dtype=torch.float32)
            return x2, clip.action_id, clip.subject
        idx = self._uniform(kp2.shape[0])
        kp2 = kp2[idx]
        conf = conf[idx]
        # 硬 mask 必须在归一化前判定（原始 conf==0）；归一化后判会被 sigmoid(0)=0.5 吞掉
        _z = conf <= 1e-6
        if self.conf_norm == "sigmoid":
            conf = _conf_norm_sigmoid(conf)
        elif self.conf_norm == "none":
            conf = np.ones_like(conf)
        else:
            raise ValueError(f"unknown conf_norm={self.conf_norm}")
        conf[_z] = 0.0
        kp2[_z] = 0.0
        x2 = np.concatenate([kp2, conf[..., None]], axis=-1).astype(np.float32)
        if self.is_train:
            x2 = self._aug_2d(x2)
        return torch.from_numpy(x2), clip.action_id, clip.subject

    def __len__(self):
        return len(self.clips)


def build_dual_pairs(root: Path, thermal_out: Path,
                     minmax: bool = False) -> Tuple[List[SkeletonClipIndex], Optional[Tuple[float, float]]]:
    """build_skeleton_index 后过滤"3D+2D 都齐全"的 clip。

    返回 (pairs, conf_global)：conf_global 仅 conf_norm=minmax 时用（全局 min/max）。
    """
    clips = build_skeleton_index(root)
    pairs = []
    conf_lo, conf_hi = np.inf, -np.inf
    for c in clips:
        if not c.pred_dir.is_dir():
            continue
        if not any(c.pred_dir.glob("*.json")):
            continue
        npz = Path(thermal_out) / f"{c.action_id}/{c.subject}/{c.sample}.npz"
        if not npz.is_file():
            continue
        pairs.append(c)
        if minmax:
            try:
                conf = np.load(npz)["conf"].astype(np.float32)
                if conf.size:
                    conf_lo = min(conf_lo, float(conf.min()))
                    conf_hi = max(conf_hi, float(conf.max()))
            except Exception:
                pass
    if minmax and np.isfinite(conf_lo) and np.isfinite(conf_hi):
        return pairs, (conf_lo, conf_hi)
    return pairs, None
