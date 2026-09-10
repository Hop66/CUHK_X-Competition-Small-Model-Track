"""
CUHK-X —— Skeleton 数据加载（Step 3）

结构: <root>/Skeleton/<Action>/<Subject>/<sample>/predictions/*.json
- 每个 json = 1 帧（list 长度 1），含 keypoints[17,3] + keypoint_scores[17]
- 拓扑 = Human3.6M-17（0骨盆 1-6双腿 7-10躯干头 11-16双臂）
- 预处理: 置信度掩码 + 肩宽(11-14)尺度归一化 + uniform 采样 + random moving 增强
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

CONF_THRESH = 0.2
SHOULDER_L, SHOULDER_R = 11, 14  # H3.6M 左右肩
# H3.6M-17 左右对称关节对（右腿 1-3 ↔ 左腿 4-6；左臂 11-13 ↔ 右臂 14-16）
MIRROR_PAIRS = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3,
                11: 14, 12: 15, 13: 16, 14: 11, 15: 12, 16: 13}


def frame_num_of(name: str) -> Optional[int]:
    m = re.search(r"(\d{8})", name)
    return int(m.group(1)) if m else None


class SkeletonClipIndex:
    def __init__(self, action_id: int, subject: str, sample: str, pred_dir: Path):
        self.action_id = action_id
        self.subject = subject
        self.sample = sample
        self.pred_dir = pred_dir


def build_skeleton_index(train_root: Path) -> List[SkeletonClipIndex]:
    """discovery 用 Depth_Color（或 Thermal），骨架目录按同路径换到 Skeleton/.../predictions。"""
    clips = []
    disc = Path(train_root) / "Depth_Color"
    if not disc.is_dir():
        disc = Path(train_root) / "Thermal"
    for action_dir in sorted(disc.iterdir()):
        if not action_dir.is_dir():
            continue
        try:
            action_id = int(action_dir.name.split("_")[0])
        except ValueError:
            continue
        for subj_dir in sorted(action_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            for sample_dir in sorted(subj_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                pred = (Path(train_root) / "Skeleton" / action_dir.name /
                        subj_dir.name / sample_dir.name / "predictions")
                clips.append(SkeletonClipIndex(action_id, subj_dir.name, sample_dir.name, pred))
    return clips


def random_moving(skel: torch.Tensor, angle: float = 10.0, translate: float = 0.05,
                  scale: float = 0.1) -> torch.Tensor:
    """首末帧各采样一组旋转/平移/缩放，中间帧线性插值（模拟相机平滑移动，ST-GCN §3.6）。"""
    if skel.shape[1] < 2:
        return skel
    B, T, V, C = skel.shape
    dev = skel.device
    coords = skel[..., :3]

    def params():
        rot = (torch.rand(B, device=dev) * 2 - 1) * (angle * math.pi / 180.0)
        tx = (torch.rand(B, device=dev) * 2 - 1) * translate
        ty = (torch.rand(B, device=dev) * 2 - 1) * translate
        sc = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * scale
        return rot, tx, ty, sc

    r0, x0, y0, s0 = params()
    r1, x1, y1, s1 = params()
    out = skel.clone()
    for t in range(T):
        a = t / max(T - 1, 1)
        rot = (1 - a) * r0 + a * r1
        tx = (1 - a) * x0 + a * x1
        ty = (1 - a) * y0 + a * y1
        sc = (1 - a) * s0 + a * s1
        c, s = torch.cos(rot), torch.sin(rot)
        xx = coords[:, t, :, 0]
        yy = coords[:, t, :, 1]
        out[:, t, :, 0] = (xx * c[:, None] - yy * s[:, None]) * sc[:, None] + tx[:, None]
        out[:, t, :, 1] = (xx * s[:, None] + yy * c[:, None]) * sc[:, None] + ty[:, None]
    return out


class SkeletonVideoDataset(Dataset):
    """Skeleton -> [T, 17, 4]（x,y,z,conf）。"""

    def __init__(self, clips: List[SkeletonClipIndex], num_frames: int = 16,
                 is_train: bool = True, seed: int = 0):
        self.clips = clips
        self.num_frames = num_frames
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.clips)

    def _load_frames(self, pred_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (kp [N,17,3], conf [N,17])。"""
        if not pred_dir.is_dir():
            return np.zeros((0, 17, 3), np.float32), np.zeros((0, 17), np.float32)
        files = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
        kps, confs = [], []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            frames = data if isinstance(data, list) else [data]
            for fr in frames:
                if not isinstance(fr, dict) or "keypoints" not in fr:
                    continue
                kp = np.asarray(fr["keypoints"], dtype=np.float32).reshape(17, 3)
                cf = np.asarray(fr.get("keypoint_scores", [1.0] * 17), dtype=np.float32).reshape(17)
                kps.append(kp)
                confs.append(cf)
        if not kps:
            return np.zeros((0, 17, 3), np.float32), np.zeros((0, 17), np.float32)
        return np.stack(kps, 0), np.stack(confs, 0)

    def _uniform(self, n: int) -> np.ndarray:
        if n <= self.num_frames:
            return np.linspace(0, max(n - 1, 0), self.num_frames).round().astype(int)
        return np.linspace(0, n - 1, self.num_frames).round().astype(int)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        kp, _ = self._load_frames(clip.pred_dir)  # [N,17,3]；conf 已验证恒 1.0，弃用
        if kp.shape[0] == 0:
            x = torch.zeros(self.num_frames, 17, 6, dtype=torch.float32)
            return x, clip.action_id, clip.subject

        idx = self._uniform(kp.shape[0])
        kp = kp[idx]

        # 坐标语义（实测）: 原始 [左右, 前后, 高度] → [左右, 高度, 前后]
        # （z 是到地面高度 0~2.27m，非图像垂直；random_moving 旋转在图像平面）
        kp = kp[:, :, [0, 2, 1]]

        # 中心化: 减去骨盆(关节0)，消除绝对位置（跨被试/跨房间泄漏）
        kp = kp - kp[:, 0:1, :]

        # 尺度归一化：除以肩宽（H3.6M 肩 11-14）
        shoulder = np.linalg.norm(kp[:, SHOULDER_L] - kp[:, SHOULDER_R], axis=-1)  # [T]
        kp = kp / (shoulder[:, None, None] + 1e-8)

        x = torch.from_numpy(kp.astype(np.float32))  # [T,17,3] = x,y,z（去 conf）
        if self.is_train:
            x = random_moving(x.unsqueeze(0)).squeeze(0)
        # 关节速度通道（帧差）：在增强后的位置上计算，保证坐标一致性
        vel = torch.zeros_like(x)
        vel[1:] = x[1:] - x[:-1]
        x = torch.cat([x, vel], dim=-1)  # [T,17,6] = x,y,z, vx,vy,vz
        return x, clip.action_id, clip.subject


class MotionBertSkeletonDataset(Dataset):
    """Skeleton -> [T, 17, 3]（MotionBERT 动作识别输入）。

    input3d=True（默认）：原生 3D [水平, 垂直, 深度] = [0,2,1]，中心化 + 肩宽归一化（3D）。
      —— 匹配 MotionBERT model_pos（H36M/AMASS 3D 预训练）输入语义 + 论文"17 3D joints"，不丢深度。
    input3d=False：旧 [x, y, conf]（2D 图像平面 + 恒 1.0 置信度，丢深度）。
    """

    def __init__(self, clips: List[SkeletonClipIndex], num_frames: int = 16,
                 is_train: bool = True, seed: int = 0, input3d: bool = True,
                 clean: bool = False, norm: str = "shoulder"):
        """clean=True: 对原始 3D 坐标做时间平滑/坏帧修复/零帧插值（中心化前）。
        norm: 'shoulder'=肩宽缩放（旧）；'torso'=骨盆0-颈8 长度缩放（0.711 notebook 建议，
          躯干比肩宽更稳——肩宽受 2 个关节噪声影响，躯干用一个长尺度）。"""
        self.clips = clips
        self.num_frames = num_frames
        self.is_train = is_train
        self.input3d = input3d
        self.clean = clean
        self.norm = norm
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.clips)

    def _load_frames(self, pred_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not pred_dir.is_dir():
            return np.zeros((0, 17, 3), np.float32), np.zeros((0, 17), np.float32)
        files = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
        kps, confs = [], []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            frames = data if isinstance(data, list) else [data]
            for fr in frames:
                if not isinstance(fr, dict) or "keypoints" not in fr:
                    continue
                kp = np.asarray(fr["keypoints"], dtype=np.float32).reshape(17, 3)
                cf = np.asarray(fr.get("keypoint_scores", [1.0] * 17), dtype=np.float32).reshape(17)
                kps.append(kp)
                confs.append(cf)
        if not kps:
            return np.zeros((0, 17, 3), np.float32), np.zeros((0, 17), np.float32)
        return np.stack(kps, 0), np.stack(confs, 0)

    def _uniform(self, n: int) -> np.ndarray:
        if n <= self.num_frames:
            return np.linspace(0, max(n - 1, 0), self.num_frames).round().astype(int)
        return np.linspace(0, n - 1, self.num_frames).round().astype(int)

    def _augment(self, feat: np.ndarray) -> np.ndarray:
        """CG 几何增强（[T,17,3]=x,y,conf，肩宽归一化后 [-1,1] 量级）。
        左右镜像（交换关节对）+ 随机旋转 + 随机缩放 + 随机平移 + 帧级抖动。"""
        r = self.rng
        # 1) 左右镜像（x 取反 + 交换左右关节对）
        if r.random() < 0.5:
            mirrored = feat.copy()
            for a, b in MIRROR_PAIRS.items():
                mirrored[:, a, :] = feat[:, b, :]
            mirrored[:, :, 0] = -mirrored[:, :, 0]
            feat = mirrored
        # 2) 随机旋转（xy 平面 ±15°）
        ang = float(r.uniform(-15.0, 15.0) * math.pi / 180.0)
        c, s = math.cos(ang), math.sin(ang)
        xy = feat[:, :, :2]
        feat[:, :, 0] = xy[:, :, 0] * c - xy[:, :, 1] * s
        feat[:, :, 1] = xy[:, :, 0] * s + xy[:, :, 1] * c
        # 3) 随机缩放（0.85-1.15，模拟体型/距离差异；3D 时深度也按同尺度）
        sc = float(r.uniform(0.85, 1.15))
        feat[:, :, :2] *= sc
        if self.input3d:
            feat[:, :, 2] *= sc
        # 4) 随机平移（xy ±0.08，肩宽量级，模拟站立位置漂移）
        feat[:, :, 0] += float(r.uniform(-0.08, 0.08))
        feat[:, :, 1] += float(r.uniform(-0.08, 0.08))
        # 5) 帧级抖动（每帧每关节独立 ±0.03，只动 xy，模拟关键点估计噪声）
        jit = r.uniform(-0.03, 0.03, size=(feat.shape[0], feat.shape[1], 2)).astype(feat.dtype)
        feat[:, :, :2] += jit
        return feat

    def __getitem__(self, i: int):
        if self.is_train:
            # worker 种源+样本序号重派生（只按 initial_seed → 同 worker 增广参数全同）
            self.rng = np.random.default_rng(int((torch.initial_seed() + i) & 0x7FFFFFFF))
        clip = self.clips[i]
        kp, conf = self._load_frames(clip.pred_dir)  # [N,17,3], [N,17]
        if kp.shape[0] == 0:
            x = torch.zeros(self.num_frames, 17, 3, dtype=torch.float32)
            return x, clip.action_id, clip.subject

        idx = self._uniform(kp.shape[0])
        kp = kp[idx]            # [T,17,3] 原始 [水平(0), 深度(1), 垂直(2)]
        conf = conf[idx]

        if self.clean and kp.shape[0] >= 3:
            from src.skeleton_clean import clean_skeleton
            kp = clean_skeleton(kp)   # 原始米制坐标上做物理一致性修复（中心化前）

        if self.input3d:
            # 原生 3D：原始 [水平, 深度, 垂直] → [水平, 垂直, 深度] = [0,2,1]
            # 与 MotionBERT model_pos（3D 预训练）输入语义一致，不丢深度
            xyz = kp[:, :, [0, 2, 1]].astype(np.float32)
            xyz = xyz - xyz[:, 0:1, :]                       # 中心化（减骨盆）
            if self.norm == "torso":
                scale = np.linalg.norm(xyz[:, 0] - xyz[:, 8], axis=-1)   # 骨盆-颈 躯干长
            else:
                scale = np.linalg.norm(xyz[:, SHOULDER_L] - xyz[:, SHOULDER_R], axis=-1)
            xyz = xyz / (scale[:, None, None] + 1e-8)        # 归一化（3D）
            feat = xyz                                       # [T,17,3] = x,y,z
        else:
            # 旧：2D 图像平面 + conf（conf 恒 1.0，丢深度）
            xy = kp[:, :, [0, 2]].astype(np.float32)
            xy = xy - xy[:, 0:1, :]
            shoulder = np.linalg.norm(xy[:, SHOULDER_L] - xy[:, SHOULDER_R], axis=-1)
            xy = xy / (shoulder[:, None, None] + 1e-8)
            feat = np.concatenate([xy, conf[..., None]], axis=-1).astype(np.float32)
        if self.is_train:
            feat = self._augment(feat)
        x = torch.from_numpy(feat)
        return x, clip.action_id, clip.subject
