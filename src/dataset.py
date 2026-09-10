"""
CUHK-X 小模型赛道 —— 数据加载（Step 1）

主线：Depth_Color + IR 同相机帧对齐 → 4 通道视频输入。
训练集结构: <Modality>/<Action>/<Subject>/<sample>/<files>
测试集结构: SM_test_XXXX/<Modality>/<files>

关键实现：
- 按文件名里的帧号把 Depth_Color 与 IR 配对（同相机帧号一致）
- endpoint-uniform 采样 T 帧
- 可选 YOLO 固定窗口裁剪（bbox 由 detect.py 预缓存）
- 缺帧补零（缺失率 Depth/IR 3.6%）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITY_DIRS = {"depth": "Depth_Color", "ir": "IR"}

_FRAME_NUM = re.compile(r"(\d{8})")  # 8 位全局帧号


def natural_key(name: str):
    parts = re.split(r"(\d+)", name.lower())
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts if p)


def frame_num_of(name: str) -> Optional[int]:
    m = _FRAME_NUM.search(name)
    return int(m.group(1)) if m else None


def list_images(d: Path) -> Dict[int, Path]:
    """列出目录下所有图像，按帧号建索引。"""
    out: Dict[int, Path] = {}
    if not d.is_dir():
        return out
    for f in d.iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        n = frame_num_of(f.name)
        if n is not None:
            out[n] = f
    return out


def segment_indices(n: int, num_frames: int, is_train: bool = False,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """全球覆盖分段采样（fork 自 14th-place thermal baseline 思想）。
    clip 均分成 num_frames 段，每段取 1 帧 —— train 段内随机，eval 段中=确定性。
    对比 endpoint-uniform：强调"段内代表性 + 全程覆盖"（呼应时间 TTA 连续窗口翻车教训）。
    """
    if n <= 0:
        return np.array([], dtype=int)
    if n <= num_frames:
        return np.linspace(0, n - 1, num_frames).round().astype(int)
    edges = np.linspace(0, n, num_frames + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = min(int(a), n - 1), max(min(int(b) - 1, n - 1), int(a))
        if is_train and rng is not None and hi > lo:
            out.append(int(rng.integers(lo, hi + 1)))
        else:
            out.append((lo + hi) // 2)
    return np.array(out, dtype=int)


class ClipIndex:
    """单个训练 clip 的元信息。"""

    def __init__(self, action_id: int, subject: str, sample: str,
                 depth_dir: Path, ir_dir: Path):
        self.action_id = action_id
        self.subject = subject
        self.sample = sample
        self.depth_dir = depth_dir
        self.ir_dir = ir_dir


def build_train_index(train_root: Path, use_dirs=(("depth", "Depth_Color"), ("ir", "IR")),
                      include_ir_only: bool = False) -> List[ClipIndex]:
    """遍历训练集，产出 clip 索引（main：Depth+IR）。默认 discovery 只用 Depth_Color 树（保持 2931 基线）；
    include_ir_only=True 时补扫 IR 树并集去重（修复仅 IR、无 Depth_Color 的 ~105 clip 被静默丢弃）。"""
    clips: List[ClipIndex] = []
    t_root = Path(train_root)
    disc = t_root / "Depth_Color"
    if not disc.is_dir():
        disc = t_root / "Thermal"
    trees = [disc]
    if include_ir_only and (t_root / "IR").is_dir() and disc != (t_root / "IR"):
        trees.append(t_root / "IR")
    seen = set()
    for tree in trees:
        for action_dir in sorted(tree.iterdir()):
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
                    key = (action_id, subj_dir.name, sample_dir.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    depth_dir = t_root / "Depth_Color" / action_dir.name / subj_dir.name / sample_dir.name
                    ir_dir = t_root / "IR" / action_dir.name / subj_dir.name / sample_dir.name
                    clips.append(ClipIndex(action_id, subj_dir.name, sample_dir.name, depth_dir, ir_dir))
    return clips


def build_test_index(test_root: Path) -> List[Tuple[str, Path, Path]]:
    """测试集：返回 (clip_id, depth_dir, ir_dir)。"""
    out = []
    root = Path(test_root)
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("SM_test_"):
            out.append((d.name, d / "Depth_Color", d / "IR"))
    return out


def _skeleton_heatmap(pred_dir, T: int = 16, S: int = 128, sigma: float = 3.0, box=None):
    """M1(3D 保留版): 骨架 3D 三正交视图 heatmap stack [T,3,S,S].
    正=[x,z], 侧=[y,z], 俯=[x,y] -- 每 view 独立 minmax 归一化到 box(与 main bbox 窗对齐)。"""
    import json as _json
    fs = sorted(Path(pred_dir).glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    if len(fs) < 2:
        return None
    idx = np.linspace(0, len(fs) - 1, T).round().astype(int)
    hm = np.zeros((T, 3, S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    var = float(sigma * sigma)
    if box is None:
        bx0, by0, bx1, by1 = 0.15, 0.05, 0.85, 0.95
    else:
        bx0, by0, bx1, by1 = [float(v) for v in box]
    ox, oy, w, h = bx0 * S, by0 * S, (bx1 - bx0) * S, (by1 - by0) * S
    views = ((0, 2), (1, 2), (0, 1))   # 正面/侧面/俯视
    for t, fi in enumerate(idx):
        try:
            o = _json.loads(fs[fi].read_text("utf-8"))
            fr = o if isinstance(o, dict) else o[0]
            kp = np.asarray(fr["keypoints"], np.float32).reshape(17, 3)
        except Exception:
            continue
        for v, (ai, bi) in enumerate(views):
            a, b = kp[:, ai], kp[:, bi]
            am, aM, bm, bM = a.min(), a.max(), b.min(), b.max()
            rw = (a - am) / max(aM - am, 1e-3)
            if ai == 0:            # 世界 x 与图像 x 反方向 → 镜像
                rw = 1.0 - rw
            px = ox + rw * w
            py = oy + (1.0 - (b - bm) / max(bM - bm, 1e-3)) * h
            for j in range(17):
                if 0 <= px[j] < S and 0 <= py[j] < S:
                    hm[t, v] += np.exp(-((xx - px[j]) ** 2 + (yy - py[j]) ** 2) / (2 * var))
    return np.clip(hm, 0, 1).astype(np.float32)


def _random_erase_t(x: torch.Tensor, p: float, rng: np.random.Generator) -> None:
    """时序一致随机擦除（原地）：以概率 p 在 x[T,C,H,W] 选单个矩形区域置中间灰，跨帧一致。
    仿真实体/传感器遮挡的跨被试域偏移。x 须在 [0,1] 且未归一化。
    """
    if rng.random() >= p or x.numel() == 0:
        return
    T, C, H, W = x.shape
    rh = int(H * float(rng.uniform(0.06, 0.30)))
    rw = int(W * float(rng.uniform(0.06, 0.30)))
    rh = max(rh, 1)
    rw = max(rw, 1)
    y0 = int(rng.integers(0, max(H - rh, 1)))
    x0 = int(rng.integers(0, max(W - rw, 1)))
    x[:, :, y0:y0 + rh, x0:x0 + rw] = 0.5


class DepthIRVideoDataset(Dataset):
    """Depth_Color(3ch) + IR(1ch) → [T, 4, H, W] 视频片段。"""

    def __init__(
        self,
        clips: List[ClipIndex],
        num_frames: int = 16,
        size: int = 128,
        is_train: bool = True,
        crop_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        use_ir_mask: bool = False,
        use_frame_diff: bool = False,
        seed: int = 0,
        brightness_alpha: float = 1.0,
        brightness_beta: float = 0.0,
        aug_strength: int = 2,
        sample_offset: float = -1.0,
        sample_mode: str = "uniform",
        track_crop: bool = False,
        box_path: str = "",
        person_margin: float = 1.1,
        return_key: bool = False,
        skel_map: Optional[dict] = None,
    ):
        self.clips = clips
        self.num_frames = num_frames
        self.size = size
        self.is_train = is_train
        self.crop_cache = crop_cache or {}
        self.skel_map = skel_map or {}   # {action_id/subject/sample: pred_dir} → 拼骨架 heatmap 通道(M1)
        self.use_ir_mask = use_ir_mask
        self.use_frame_diff = use_frame_diff
        self.sample_offset = sample_offset  # 时间 TTA：-1=endpoint uniform；0~1=窗口起点偏移
        self.sample_mode = sample_mode      # uniform | segment（全球覆盖分段采样，14th-place 移植）
        self.track_crop = track_crop        # main 3D 主战场：逐帧跟人裁剪 + 轨迹通道
        self.return_key = return_key        # 蒸馏/教师对齐用：__getitem__ 追加 clip key
        self.person_margin = person_margin
        self._pf_frame = {}
        if track_crop and box_path:
            import json as _json
            raw = _json.loads(Path(box_path).read_text(encoding="utf-8"))
            for key, v in raw.items():
                if isinstance(v, dict) and "names" in v and "boxes" in v:
                    self._pf_frame[key] = {}
                    for _nm, _bx in zip(v["names"], v["boxes"]):
                        fr = frame_num_of(_nm)
                        if fr is not None and _bx is not None:
                            self._pf_frame[key][fr] = _bx
                else:
                    self._pf_frame[key] = {}
            print(f"[DepthIRVideoDataset] track_crop box_path={box_path} clips={len(self._pf_frame)}", flush=True)
        self.rng = np.random.default_rng(seed)
        # 亮度/对比度校正（测试侧亮度捷径验证用；默认 1.0/0.0 不改变行为）
        self.brightness_alpha = brightness_alpha
        self.brightness_beta = brightness_beta
        # 训练增强强度档位（0=无 1=温和 2=当前强 3=更强 4=域鲁棒），供组合搜索
        self.aug_strength = aug_strength
        self.aug_perchannel = False   # 4 档：逐通道独立亮度抖动（Depth 伪彩/IR 各自）
        self.aug_erase = 0.0          # 4 档：时序一致随机擦除概率（实体遮挡）
        self.aug_speed = 0.0          # 4 档：时间速度抖动概率（跨被试动作速度域偏移）
        if aug_strength == 0:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.0, (1.0, 1.0), (1.0, 1.0)
            self.aug_scale, self.aug_shift = (1.0, 1.0), 0.0
        elif aug_strength == 1:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (0.9, 1.1), (0.9, 1.1)
            self.aug_scale, self.aug_shift = (0.9, 1.1), 0.06
        elif aug_strength == 3:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (0.75, 1.25), (0.75, 1.25)
            self.aug_scale, self.aug_shift = (0.8, 1.2), 0.10
        elif aug_strength == 4:  # 域鲁棒配方：擦除+逐通道扰动+时间速度抖动+更大尺度
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (1.0, 1.0), (1.0, 1.0)
            self.aug_scale, self.aug_shift = (0.75, 1.3), 0.14
            self.aug_perchannel = True
            self.aug_erase = 0.3
            self.aug_speed = 0.3
        else:  # 2 = 当前强（默认）
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (0.8, 1.2), (0.8, 1.2)
            self.aug_scale, self.aug_shift = (0.85, 1.15), 0.08

        # Kinetics 均值/方差（3ch）+ IR 通道用三通道均值；帧差通道 mean=0 std=0.25
        mean3 = (0.43216, 0.394666, 0.37645)
        std3 = (0.22803, 0.22145, 0.216989)
        if use_frame_diff:
            C = 8
            self.mean = torch.tensor((*mean3, sum(mean3) / 3.0, 0.0, 0.0, 0.0, 0.0)).view(1, C, 1, 1)
            self.std = torch.tensor((*std3, sum(std3) / 3.0, 0.25, 0.25, 0.25, 0.25)).view(1, C, 1, 1)
        else:
            self.mean = torch.tensor((*mean3, sum(mean3) / 3.0)).view(1, 4, 1, 1)
            self.std = torch.tensor((*std3, sum(std3) / 3.0)).view(1, 4, 1, 1)
    def _append_skel_heat(self, x: torch.Tensor, clip, do_flip: bool, box=None):
        """M1: 骨架 heatmap 通道并入 x(pixel-aligned 于 bbox 窗; 拼在归一化后)。
        无骨架的 clip 也补 3ch 零 → batch 恒 7ch(避免 collate 通道不一致)。"""
        key = f"{clip.action_id}/{clip.subject}/{clip.sample}"
        pd2 = self.skel_map.get(key)
        T, _, S, _ = x.shape
        if pd2 is None:
            hm = torch.zeros(T, 3, S, S)
        else:
            hm_np = _skeleton_heatmap(pd2, T=T, S=S, box=box)
            hm = torch.from_numpy(hm_np) if hm_np is not None else torch.zeros(T, 3, S, S)
        if do_flip:
            hm = torch.flip(hm, dims=(3,))
        return torch.cat([x, hm.to(x.dtype)], dim=1)
    def __len__(self):
        return len(self.clips)

    def _uniform_indices(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if self.sample_mode == "segment":
            return segment_indices(n, self.num_frames, self.is_train, self.rng)
        if n <= self.num_frames:
            # 帧数不足：重复采样到 num_frames
            idx = np.linspace(0, n - 1, self.num_frames).round().astype(int)
            return idx
        if self.sample_offset >= 0:
            # 时间 TTA：窗口 [start, start+num_frames) 内均匀采样
            max_start = n - self.num_frames
            start = int(self.sample_offset * max_start)
            return np.linspace(start, start + self.num_frames - 1,
                               self.num_frames).round().astype(int)
        return np.linspace(0, n - 1, self.num_frames).round().astype(int)

    def __getitem__(self, i: int):
        if self.is_train:
            # 修复：DataLoader fork 复制同一 self.rng → 各 worker 同序列；用 worker 种源+样本序号重派生
            #（只按 initial_seed 会让同一 worker 内所有样本增广参数相同）
            self.rng = np.random.default_rng(int((torch.initial_seed() + i) & 0x7FFFFFFF))
        clip = self.clips[i]
        depth_map = list_images(clip.depth_dir)
        ir_map = list_images(clip.ir_dir)
        # 取两者共有的帧号（同相机应一致）
        common = sorted(set(depth_map.keys()) & set(ir_map.keys()))
        if not common:
            common = sorted(depth_map.keys()) or sorted(ir_map.keys())
        idx = self._uniform_indices(len(common))
        # 4 档：时间速度抖动（跨被试速度域偏移仿真：非线性重映射帧索引 → 快/慢动作）
        if self.is_train and self.aug_speed > 0 and len(common) > 4 and self.rng.random() < self.aug_speed:
            _n = len(common)
            stride = float(self.rng.uniform(0.8, 1.25))
            pos = np.arange(len(idx)) * stride
            if pos[-1] > 1e-6:
                pos = pos * (_n - 1) / pos[-1]
            idx = np.round(np.clip(pos, 0, _n - 1)).astype(int)
        picked = [common[j] for j in idx]

        crop = self.crop_cache.get(self._clip_key(clip), None)

        # ---- 训练增强参数（per-clip 一致）：按 aug_strength 档位配置 ----
        do_flip = False
        bright, contrast = 1.0, 1.0
        if self.is_train:
            do_flip = self.rng.random() < self.aug_flip
            bright = float(self.rng.uniform(*self.aug_bright))
            contrast = float(self.rng.uniform(*self.aug_contrast))
            if crop is not None:
                s = float(self.rng.uniform(*self.aug_scale))
                dx = float(self.rng.uniform(-self.aug_shift, self.aug_shift))
                dy = float(self.rng.uniform(-self.aug_shift, self.aug_shift))
                x1, y1, x2, y2 = crop
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nw, nh = w * s, h * s
                crop = (max(cx - nw / 2.0 + dx * w, 0.0), max(cy - nh / 2.0 + dy * h, 0.0),
                        min(cx + nw / 2.0 + dx * w, 1.0), min(cy + nh / 2.0 + dy * h, 1.0))

        frames = np.zeros((self.num_frames, 4, self.size, self.size), dtype=np.float32)
        traj = np.zeros((self.num_frames, 4), dtype=np.float32)
        pf_map = self._pf_frame.get(self._clip_key(clip), {}) if self.track_crop else None
        last_box = crop
        for t, fn in enumerate(picked):
            # 逐帧跟人：优先用该帧 bbox；缺失回退 clip 框；并采集 [cx,cy,bw,bh] 轨迹（位移/尺度显式回补）
            box_use = crop
            if pf_map is not None:
                b0 = pf_map.get(fn)
                if b0 is not None:
                    box_use = b0
                    last_box = b0
                elif last_box is not None:
                    box_use = last_box
            if box_use is not None:
                traj[t] = [(box_use[0] + box_use[2]) / 2.0, (box_use[1] + box_use[3]) / 2.0,
                           (box_use[2] - box_use[0]), (box_use[3] - box_use[1])]
            else:
                traj[t] = [0.5, 0.5, 0.5, 0.5]
            # Depth_Color → RGB 3ch；IR → 1ch；均应用该帧窗口裁剪（保运动）
            d = self._load(depth_map.get(fn), 3, box_use)
            r = self._load(ir_map.get(fn), 1, box_use)
            if d is None and r is None:
                continue
            if d is None:
                d = np.zeros((self.size, self.size, 3), dtype=np.float32)
            if r is None:
                r = np.zeros((self.size, self.size, 1), dtype=np.float32)
            # IR 人像掩码：用 IR 灰度 Otsu 阈值抑制 Depth 背景（IR 与 Depth 同相机同帧号）
            if self.use_ir_mask and r is not None and r.any():
                d = d * self._otsu_mask(r[..., 0])[..., None]
            cat = np.concatenate([d, r], axis=2)  # [H, W, 4]
            if self.brightness_alpha != 1.0 or self.brightness_beta != 0.0:
                cat = np.clip(cat * self.brightness_alpha + self.brightness_beta, 0.0, 255.0)
            frames[t] = cat.transpose(2, 0, 1) / 255.0

        x = torch.from_numpy(frames)
        # 训练增强：亮度/对比度（只动 brightness/contrast，保护 Depth 伪彩色几何语义）
        if self.is_train:
            if self.aug_perchannel:
                gains = torch.tensor(self.rng.uniform(0.88, 1.12, size=x.shape[1]),
                                     dtype=x.dtype).view(1, -1, 1, 1)
                x = torch.clamp(x * gains, 0.0, 1.0)
            else:
                x = x * bright
                x = (x - 0.5) * contrast + 0.5
                x = torch.clamp(x, 0.0, 1.0)
        if self.use_frame_diff:
            diff = torch.zeros_like(x)
            diff[1:] = (x[1:] - x[:-1]).abs()
            x = torch.cat([x, diff], dim=1)  # [T, 8, H, W]：前 4 原始 + 后 4 帧差
        if do_flip:
            x = torch.flip(x, dims=(3,))  # 水平翻转（对 4/8ch 一致）
            traj[:, 0] = 1.0 - traj[:, 0]
        if self.is_train and self.aug_erase > 0:
            _random_erase_t(x, self.aug_erase, self.rng)
        x = (x - self.mean) / self.std
        if self.skel_map:
            x = self._append_skel_heat(x, clip, do_flip, crop)
        label = clip.action_id
        out = (x, label, clip.subject)
        if self.track_crop:
            out = (x, torch.from_numpy(traj), label, clip.subject)
        if self.return_key:
            out = out + (f"{clip.action_id}/{clip.subject}/{clip.sample}",)
        return out

    def _load(self, path: Optional[Path], channels: int,
              crop: Optional[Tuple[float, float, float, float]] = None) -> Optional[np.ndarray]:
        if path is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype == np.uint16:
            img = (img / 65535.0 * 255.0).astype(np.uint8)
        # 固定窗口裁剪（归一化坐标，先裁剪再 resize；保运动）
        if crop is not None:
            H, W = img.shape[:2]
            x1, y1, x2, y2 = crop
            ix1, iy1, ix2, iy2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            if ix2 > ix1 + 1 and iy2 > iy1 + 1:
                img = img[iy1:iy2, ix1:ix2]
        if channels == 3:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # 1ch
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if channels == 1:
            img = img[..., None]
        return img.astype(np.float32)

    def _clip_key(self, clip: ClipIndex) -> str:
        # 必须含 action_id：同一被试不同动作的 sample 名会重复，subject/sample 不唯一
        return f"{clip.action_id}/{clip.subject}/{clip.sample}"

    @staticmethod
    def _otsu_mask(gray: np.ndarray) -> np.ndarray:
        """Otsu 阈值 + 膨胀，得到 0/1 人像掩码（灰度值 0-255 float）。"""
        g = np.clip(gray, 0, 255).astype(np.uint8)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
        return (m > 0).astype(np.float32)


# ============ Thermal 单模态（独立相机，Step 3）============

class ThermalClipIndex:
    def __init__(self, action_id: int, subject: str, sample: str, thermal_dir: Path):
        self.action_id = action_id
        self.subject = subject
        self.sample = sample
        self.thermal_dir = thermal_dir


def build_thermal_index(train_root: Path) -> List[ThermalClipIndex]:
    root = Path(train_root)
    disc = root / "Thermal"
    if not disc.is_dir():
        disc = root / "Depth_Color"
    clips = []
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
                tdir = root / "Thermal" / action_dir.name / subj_dir.name / sample_dir.name
                clips.append(ThermalClipIndex(action_id, subj_dir.name, sample_dir.name, tdir))
    return clips


class ThermalVideoDataset(Dataset):
    """Thermal 3ch 伪彩色视频片段 → [T, 3, H, W]。"""

    def __init__(self, clips: List[ThermalClipIndex], num_frames: int = 16, size: int = 128,
                 is_train: bool = True,
                 crop_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                 use_frame_diff: bool = False,
                 seed: int = 0, aug_strength: int = 2,
                 sample_offset: float = -1.0,
                 sample_mode: str = "uniform",
                 mean_std: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
                 track_crop: bool = False,
                 box_path: str = "",
                 person_margin: float = 1.1,
                 return_key: bool = False):
        self.clips = clips
        self.num_frames = num_frames
        self.size = size
        self.is_train = is_train
        self.crop_cache = crop_cache or {}
        self.use_frame_diff = use_frame_diff
        self.return_key = return_key
        self.sample_offset = sample_offset  # 时间 TTA：-1=endpoint uniform；0~1=窗口起点偏移
        self.sample_mode = sample_mode      # uniform | segment（全球覆盖分段采样）
        self.mean_std = mean_std            # 自定义归一化（如 ImageNet / 灰度拉伸），None=默认 Kinetics
        self.track_crop = track_crop        # 逐帧跟人裁剪 + 轨迹通道（人物满框 + 位移/尺度显式回补）
        self.person_margin = person_margin
        self._pf = {}
        if track_crop and box_path:
            import json as _json
            raw = _json.loads(Path(box_path).read_text(encoding="utf-8"))
            for key, v in raw.items():
                if isinstance(v, dict) and "names" in v and "boxes" in v:
                    self._pf[key] = {nm: (bx if bx is not None else None)
                                     for nm, bx in zip(v["names"], v["boxes"])}
                else:
                    self._pf[key] = {}
            print(f"[ThermalVideoDataset] track_crop box_path={box_path} clips={len(self._pf)}", flush=True)
        self.rng = np.random.default_rng(seed)
        # 训练增强强度档位（与 DepthIR 一致：0=无 1=温和 2=强 3=更强）
        self.aug_strength = aug_strength
        self.aug_perchannel = False
        self.aug_erase = 0.0
        self.aug_speed = 0.0
        if aug_strength == 0:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.0, (1.0, 1.0), (1.0, 1.0)
            self.aug_scale, self.aug_shift = (1.0, 1.0), 0.0
        elif aug_strength == 1:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (0.9, 1.1), (0.9, 1.1)
            self.aug_scale, self.aug_shift = (0.9, 1.1), 0.06
        elif aug_strength == 3:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (0.75, 1.25), (0.75, 1.25)
            self.aug_scale, self.aug_shift = (0.8, 1.2), 0.10
        elif aug_strength == 4:
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (1.0, 1.0), (1.0, 1.0)
            self.aug_scale, self.aug_shift = (0.75, 1.3), 0.14
            self.aug_perchannel = True
            self.aug_erase = 0.3
            self.aug_speed = 0.3
        else:  # 2 = 当前强（默认）
            self.aug_flip, self.aug_bright, self.aug_contrast = 0.5, (0.8, 1.2), (0.8, 1.2)
            self.aug_scale, self.aug_shift = (0.85, 1.15), 0.08
        # Kinetics 均值/方差（R2+1D-18/34 + IG-65M 预训练域，与 DepthIR 一致）；帧差 mean=0 std=0.25
        mean3 = (0.43216, 0.394666, 0.37645)
        std3 = (0.22803, 0.22145, 0.216989)
        if mean_std is not None:  # 自定义归一化（如 VideoMAE 用 ImageNet 0.485/0.229）
            mean3, std3 = mean_std
        if use_frame_diff:
            self.mean = torch.tensor((*mean3, 0.0, 0.0, 0.0)).view(1, 6, 1, 1)
            self.std = torch.tensor((*std3, 0.25, 0.25, 0.25)).view(1, 6, 1, 1)
        else:
            self.mean = torch.tensor(mean3).view(1, 3, 1, 1)
            self.std = torch.tensor(std3).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.clips)

    def _sample_indices(self, n: int) -> np.ndarray:
        """时间采样索引（-1=endpoint uniform；sample_offset>=0=窗口起点；供时间 TTA/骨架对齐复用）。"""
        if n <= 0:
            return np.array([], dtype=int)
        if self.sample_mode == "segment":
            return segment_indices(n, self.num_frames, self.is_train, self.rng)
        if n <= self.num_frames:
            return np.linspace(0, n - 1, self.num_frames).round().astype(int)
        if self.sample_offset >= 0:
            max_start = n - self.num_frames
            start = int(self.sample_offset * max_start)
            return np.linspace(start, start + self.num_frames - 1,
                               self.num_frames).round().astype(int)
        return np.linspace(0, n - 1, self.num_frames).round().astype(int)

    def __getitem__(self, i: int):
        if self.is_train:
            # 同 DepthIR：worker 种源+样本序号重派生（只按 initial_seed → 同 worker 增广参数全同）
            self.rng = np.random.default_rng(int((torch.initial_seed() + i) & 0x7FFFFFFF))
        clip = self.clips[i]
        files = sorted(clip.thermal_dir.glob("*.jpg")) + sorted(clip.thermal_dir.glob("*.png"))
        if not files:
            self._last_do_flip = False
            return torch.zeros(self.num_frames, 3, self.size, self.size), clip.action_id, clip.subject
        n = len(files)
        idx = self._sample_indices(n)
        # 4 档：时间速度抖动（跨被试速度域偏移仿真，与 DepthIR 流同协议）
        if self.is_train and self.aug_speed > 0 and n > 4 and self.rng.random() < self.aug_speed:
            stride = float(self.rng.uniform(0.8, 1.25))
            pos = np.arange(len(idx)) * stride
            if pos[-1] > 1e-6:
                pos = pos * (n - 1) / pos[-1]
            idx = np.round(np.clip(pos, 0, n - 1)).astype(int)
        crop = self.crop_cache.get(f"{clip.action_id}/{clip.subject}/{clip.sample}")
        # 训练增强参数（per-clip 一致）：翻转 + 亮度/对比度 + 缩放/平移（按 aug_strength）
        do_flip = False
        bright, contrast = 1.0, 1.0
        if self.is_train:
            do_flip = self.rng.random() < self.aug_flip
            bright = float(self.rng.uniform(*self.aug_bright))
            contrast = float(self.rng.uniform(*self.aug_contrast))
            if crop is not None and self.aug_strength >= 1:
                s = float(self.rng.uniform(*self.aug_scale))
                dx = float(self.rng.uniform(-self.aug_shift, self.aug_shift))
                dy = float(self.rng.uniform(-self.aug_shift, self.aug_shift))
                x1, y1, x2, y2 = crop
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nw, nh = w * s, h * s
                crop = (max(cx - nw / 2.0 + dx * w, 0.0), max(cy - nh / 2.0 + dy * h, 0.0),
                        min(cx + nw / 2.0 + dx * w, 1.0), min(cy + nh / 2.0 + dy * h, 1.0))
        self._last_do_flip = do_flip  # 子类（video+骨架等）据此同步镜像非图像目标
        frames = np.zeros((self.num_frames, 3, self.size, self.size), np.float32)
        # 逐帧跟人裁剪（人物满框 + 背景≈0）：pf_map 每帧 box；缺失帧回退 clip 单框
        traj = np.zeros((self.num_frames, 4), np.float32)   # [cx, cy, bw, bh] 相对画面（位移/尺度显式通道）
        pf_map = self._pf.get(f"{clip.action_id}/{clip.subject}/{clip.sample}", {}) if self.track_crop else None
        last_box = crop
        for t, j in enumerate(idx):
            img = cv2.imread(str(files[j]), cv2.IMREAD_UNCHANGED)
            box_use = crop
            if pf_map is not None:
                b0 = pf_map.get(files[j].name)
                if b0 is not None:
                    box_use = b0
                    last_box = b0
                elif last_box is not None:
                    box_use = last_box
            if box_use is not None:
                bx1, by1, bx2, by2 = box_use
                traj[t] = [(bx1 + bx2) / 2.0, (by1 + by2) / 2.0, (bx2 - bx1), (by2 - by1)]
            else:
                traj[t] = [0.5, 0.5, 0.5, 0.5]
            if img is None:
                continue
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255.0).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if box_use is not None:
                H, W = img.shape[:2]
                x1, y1, x2, y2 = box_use
                ix1, iy1, ix2, iy2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                if ix2 > ix1 + 1 and iy2 > iy1 + 1:
                    img = img[iy1:iy2, ix1:ix2]
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            frames[t] = img.transpose(2, 0, 1) / 255.0
        x = torch.from_numpy(frames)
        if self.is_train:
            if self.aug_perchannel:
                gains = torch.tensor(self.rng.uniform(0.88, 1.12, size=x.shape[1]),
                                     dtype=x.dtype).view(1, -1, 1, 1)
                x = torch.clamp(x * gains, 0.0, 1.0)
            else:
                x = x * bright
                x = (x - 0.5) * contrast + 0.5
                x = torch.clamp(x, 0.0, 1.0)
        if self.use_frame_diff:
            diff = torch.zeros_like(x)
            diff[1:] = (x[1:] - x[:-1]).abs()
            x = torch.cat([x, diff], dim=1)  # [T, 6, H, W]：前 3 原始 + 后 3 帧差
        if do_flip:
            x = torch.flip(x, dims=(3,))
            traj[:, 0] = 1.0 - traj[:, 0]   # 水平翻转后人物 x 中心镜像
        if self.is_train and self.aug_erase > 0:
            _random_erase_t(x, self.aug_erase, self.rng)
        x = (x - self.mean) / self.std
        if self.track_crop:
            return x, torch.from_numpy(traj), clip.action_id, clip.subject
        out = (x, clip.action_id, clip.subject)
        if self.return_key:
            out = out + (f"{clip.action_id}/{clip.subject}/{clip.sample}",)
        return out


def build_balanced_sampler(labels: List[int], num_samples: Optional[int] = None):
    """按类频率取反权重的 WeightedRandomSampler（长尾比 29.5 必须处理）。"""
    from collections import Counter
    cnt = Counter(labels)
    weights = [1.0 / cnt[l] for l in labels]
    if num_samples is None:
        num_samples = len(labels)
    return torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64), num_samples=num_samples, replacement=True)
