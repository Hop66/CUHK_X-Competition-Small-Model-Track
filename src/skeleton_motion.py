"""骨架 → 运动特征（体现方向/距离/速度/朝向），供 thermal 双流的「运动通路」使用。

数据格式（已在 src/skeleton_dataset.py + 真实样例确认）:
  <root>/Skeleton/<Action>/<Subject>/<sample>/predictions/*.json
  json = [ { "keypoints":[17,3], "keypoint_scores":[17] } ]   # conf 实测恒 1.0
  坐标 = 米制，轴序 [左右(x), 前后/深度(y), 高度(z, 0~2.27m)]
  拓扑 = H3.6M-17: 0骨盆 1-3右腿 4-6左腿 7-10脊柱头
                   11左肩 12左肘 13左腕 14右肩 15右肘 16右腕

运动特征（每帧 K=29 维），全部除以「躯干长(骨盆0-颈8)」做量纲/体型归一（跨被试、防体型捷径）:
  0-5  根位移/空间动态（探测自愈）：root 水平 x/y 有变化→绝对位移+速度（含走近走远/横向；
        数据若已根中心化（x/y=0 恒定）→ 切到 根高度z+vz / 伸手速度 / 抬臂速度 / 躯干倾角速度，
        因为根水平无信息、相机空间距离另由 bbox 轨迹/雷达提供）
  6-20 左右腕/左右踝/颈 5 关节速度    → 局部运动方向（15 维）
  21   全体 17 关节速度范数均值        → 运动幅度（剧烈/平缓）
  22   左右手腕速范数之和              → 手部剧烈度（26/40 类手/上肢主导）
  23   肩线水平朝向角 atan2(Δy,Δx)   → 躯干朝向（转身/朝向）
  24   肩线朝向角速度                  → 转身速率
  25-26 左右腕 z-肩 z（抬臂高度）      → 举手/挥臂
  27-28 左右腕-骨盆距离                → 伸手范围

缓存: build_motion_cache() 产出 {clip_key: arr[N,29] float16}（N=骨架帧数）；
      训练侧再按需比例重采样到 T 帧（帧对齐用比例，非帧号硬对齐）。
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# H3.6M-17 关键关节
J_ROOT, J_NECK = 0, 8
J_SHOULDER_L, J_SHOULDER_R = 11, 14
J_WRIST_L, J_WRIST_R = 13, 16
J_ANKLE_L, J_ANKLE_R = 3, 6
KEY_JOINTS = [J_WRIST_L, J_WRIST_R, J_ANKLE_L, J_ANKLE_R, J_NECK]  # 5 个局部速度关节
K_DIM = 29


def frame_num_of(name: str) -> Optional[int]:
    m = re.search(r"(\d{8})", name)
    return int(m.group(1)) if m else None


def load_skeleton(pred_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读 predictions/*.json → (kp [N,17,3], conf [N,17])。"""
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        return np.zeros((0, 17, 3), np.float32), np.zeros((0, 17), np.float32)
    files = sorted(pred_dir.glob("*.json"), key=lambda f: frame_num_of(f.name) or 0)
    kps, confs = [], []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for fr in (data if isinstance(data, list) else [data]):
            if not isinstance(fr, dict) or "keypoints" not in fr:
                continue
            kp = np.asarray(fr["keypoints"], dtype=np.float32).reshape(17, 3)
            cf = np.asarray(fr.get("keypoint_scores", [1.0] * 17), dtype=np.float32).reshape(17)
            kps.append(kp)
            confs.append(cf)
    if not kps:
        return np.zeros((0, 17, 3), np.float32), np.zeros((0, 17), np.float32)
    return np.stack(kps, 0), np.stack(confs, 0)


def resample_to_T(arr: np.ndarray, T: int) -> np.ndarray:
    """[N, ...] → [T, ...] 沿帧轴比例线性重采样（导热像自由帧数 24fps vs 骨架 10fps）。"""
    n = arr.shape[0]
    if n == 0:
        return np.zeros((T,) + arr.shape[1:], dtype=arr.dtype)
    if n == 1:
        return np.repeat(arr, T, axis=0)
    old = np.linspace(0, n - 1, n)
    new = np.linspace(0, n - 1, T)
    flat = arr.reshape(n, -1)
    out = np.stack([np.interp(new, old, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((T,) + arr.shape[1:]).astype(np.float32)


def extract_motion_features(kp: np.ndarray, T: Optional[int] = None,
                            speed_scale: float = 1.0,
                            resample: Optional[int] = None) -> np.ndarray:
    """kp[N,17,3]（米制 , 轴序[x左右, y前后=深度, z高度]）→ 运动特征 [T, K_DIM]。

    - 先提取每帧特征（N 行 K_DIM），再按 T 比例重采样（T 省略则返回 [N, K_DIM]）。
    - 全部除以躯干长（0-8 平均距离），量纲无关、跨被试体型归一。
    - speed_scale：对时间差分类 dim（速度/角速度）的幅值缩放。**测试骨架与训练骨架
      采样密度/fps 可能不同（实测测试每帧速度幅值≈训练×2.09，但帧数相近，说明测试
      每帧跨越的真实时间更长，即测试实际更「稀」或含关节噪声）** → 速度特征(/帧)跨域
      → MotionNet OOD。推理时传 训练fps/测试fps 以对齐训练域；训练/val 提取传 1.0。
    - resample（int）：**真·重采样** —— 在特征提取前先把原始 kp [N,17,3] 线性重采样
      到 M=resample 帧（kp 级，非特征级）。作用：改变每帧跨越的真实时间 → 重算出的
      速度特征才真正对齐目标 cadence（speed_scale 只是提取后标量缩放，不改变时序）。
      传入 M 大于 N 为升采样（每帧物理时间变短→每帧速度变小）；小于 N 为降采样。
    """
    kp = kp.astype(np.float32)
    if resample is not None and kp.shape[0] > 0:
        kp = resample_to_T(kp, int(resample))
    n = kp.shape[0]
    if n == 0:
        return np.zeros((T or 0, K_DIM), np.float32)
    torso = float(np.nanmean(np.linalg.norm(kp[:, J_ROOT] - kp[:, J_NECK], axis=1))) + 1e-6

    root = kp[:, J_ROOT, :]                     # [N,3] 骨盆（数据常已根中心化）
    body = kp - root[:, None, :]                # [N,17,3] 中心化（局部运动）

    feat = np.zeros((n, K_DIM), np.float32)
    # --- 0-5 根/空间位移动态：探测自愈 ---
    # 实测(本地样例/服务器同构)：root x/y 恒 0（数据按根中心化）→ 绝対水平位移拿不到，
    #   自动切到「根高度 + 伸手/抬臂/躯干倾角的动态」；若确有水平位移则用绝对项。
    has_abs = float(np.std(root[:, :2])) > 1e-3
    if has_abs:
        feat[:, 0:3] = root / torso             # 骨盆绝对 x/y/z（空间位置/距离）
        vr = np.zeros_like(root)
        vr[1:] = root[1:] - root[:-1]
        feat[:, 3:6] = vr / torso               # 根速度（方向+快慢）
    else:
        feat[:, 0] = root[:, 2] / torso                   # 根高度 z（站/蹲）
        vz = np.zeros(n, np.float32); vz[1:] = root[1:, 2] - root[:-1, 2]
        feat[:, 1] = vz / torso                            # 根高度速度
        wrl = np.linalg.norm(body[:, J_WRIST_L], axis=1)  # 伸手距离
        wrr = np.linalg.norm(body[:, J_WRIST_R], axis=1)
        for c, d in [(2, wrl), (3, wrr)]:
            dd = np.zeros(n, np.float32); dd[1:] = d[1:] - d[:-1]
            feat[:, c] = dd / torso                        # 伸手/收手速度
        handz = body[:, J_WRIST_L, 2] - body[:, J_SHOULDER_L, 2]  # 抬臂高度
        dhz = np.zeros(n, np.float32); dhz[1:] = handz[1:] - handz[:-1]
        feat[:, 4] = dhz / torso                           # 抬/放臂速度
        tilt = np.arctan2(body[:, J_NECK, 1], body[:, J_NECK, 2])  # 躯干前后倾角
        dtilt = np.zeros(n, np.float32); dtilt[1:] = tilt[1:] - tilt[:-1]
        feat[:, 5] = dtilt                                  # 倾角变化速度
    # 6-20 5 个关键关节速度（局部运动方向）
    for c, ji in enumerate(KEY_JOINTS):
        v = np.zeros_like(body[:, ji])
        v[1:] = body[1:, ji] - body[:-1, ji]
        feat[:, 6 + c * 3 : 9 + c * 3] = v / torso
    # 21 全体速度范数均值（运动幅度）
    dv = np.zeros_like(body)
    dv[1:] = body[1:] - body[:-1]
    feat[:, 21] = np.linalg.norm(dv, axis=2).mean(axis=1) / torso
    # 22 手腕速范数之和（手部剧烈度）
    feat[:, 22] = (np.linalg.norm(dv[:, J_WRIST_L], axis=1)
                   + np.linalg.norm(dv[:, J_WRIST_R], axis=1)) / torso
    # 23 肩线朝向角（转身/躯干朝向）
    sh = body[:, J_SHOULDER_L] - body[:, J_SHOULDER_R]
    feat[:, 23] = np.arctan2(sh[:, 1], sh[:, 0])
    # 24 肩线朝向角速度
    dang = np.zeros(n, np.float32)
    dang[1:] = feat[1:, 23] - feat[:-1, 23]
    feat[:, 24] = dang
    # 25-26 抬臂高度（腕 z - 肩 z）
    feat[:, 25] = body[:, J_WRIST_L, 2] - body[:, J_SHOULDER_L, 2]
    feat[:, 26] = body[:, J_WRIST_R, 2] - body[:, J_SHOULDER_R, 2]
    # 27-28 伸手范围（腕-骨盆 距离）
    feat[:, 27] = np.linalg.norm(body[:, J_WRIST_L], axis=1)
    feat[:, 28] = np.linalg.norm(body[:, J_WRIST_R], axis=1)

    # 帧率归一：仅缩放时间差分类 dim（速度/角速度），位置/距离/角度类保持
    # （dim0 根高, dim23 朝向角, dim25-28 高度/距离 不动）
    if speed_scale != 1.0:
        VEL = [1, 2, 3, 4, 5] + list(range(6, 23)) + [24]
        feat[:, VEL] *= speed_scale

    if T is not None:
        return resample_to_T(feat, T)
    return feat


def build_motion_cache(train_root, out_pkl: str, predicate=None) -> Dict[str, np.ndarray]:
    """全量预抽取：{clip_key(f"{action_id}/{subject}/{sample}"): arr[N,K_DIM] float16}，pickle 落盘。
    predicate(clip)->bool 可选过滤（如只看 thermal 索引覆盖的 clip）。
    """
    from src.skeleton_dataset import build_skeleton_index
    clips = build_skeleton_index(Path(train_root))
    cache: Dict[str, np.ndarray] = {}
    n_missing = 0
    for i, c in enumerate(clips):
        if predicate is not None and not predicate(c):
            continue
        kp, _ = load_skeleton(c.pred_dir)
        if kp.shape[0] == 0:
            n_missing += 1
            continue
        key = f"{c.action_id}/{c.subject}/{c.sample}"
        cache[key] = extract_motion_features(kp).astype(np.float16)
        if (i + 1) % 500 == 0:
            print(f"[motion] {i + 1}/{len(clips)}  n_missing={n_missing}", flush=True)
    out = Path(out_pkl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"[motion] cache saved {len(cache)} clips (missing={n_missing}) -> {out}", flush=True)
    return cache


if __name__ == "__main__":
    import sys
    # 快速自测：python -m src.skeleton_motion <pred_dir或root>
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/Training/data/Skeleton")
    if (d / "predictions").is_dir():
        d = d / "predictions"
    kp, conf = load_skeleton(d)
    print(f"kp {kp.shape} conf {conf.shape}")
    if kp.shape[0]:
        fe = extract_motion_features(kp)
        print(f"feat {fe.shape} range=[{fe.min():.3f},{fe.max():.3f}]")
        print(f"root深度(y,0-1) {fe[0,1]:.3f}->{fe[-1,1]:.3f} | 根高 z {fe[0,2]:.3f}->{fe[-1,2]:.3f}")
        print(f"肩线朝向角 0={fe[0,23]:.3f} last={fe[-1,23]:.3f} | 间腹速 mean={fe[:,21].mean():.4f}")
        print(f"抬手 左腕z-肩z max={fe[:,25].max():.3f} | 伸手 右腕-根 max={fe[:,28].max():.3f}")
