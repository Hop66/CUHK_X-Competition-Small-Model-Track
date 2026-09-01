"""本地自测：Step 3 双分支（共享 backbone + 双输入头）——合成数据验证。

不依赖真实数据/GPU，验证：
  1. build_dual_pairs 过滤（3D+2D 都齐全）
  2. DualSkeletonDataset 取样本形状 + 增强路径
  3. DualBranchActionNet forward/backward 形状
  4. train_dual_branch 一轮 mini 训练可跑通

用法: python tests/selftest_dual_branch.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.dual_branch import DualBranchActionNet
from src.dual_dataset import (DualSkeletonDataset, Thermal2DSkeletonDataset,
                              build_dual_pairs)
from src.motionbert.dstformer import DSTformer
from src.split import split_by_subject


def make_synth(root: Path, thermal_out: Path, n_subj=4, n_action=3, n_sample=2, n_frame=20):
    """合成 Skeleton 3D json + 热像 2D npz（H36M-17 顺序，模拟真实命名）。"""
    rng = np.random.default_rng(0)
    for a in range(n_action):
        adir = f"{a}_action{a}"
        for s in range(n_subj):
            subj = f"S{s:02d}"
            for sm in range(n_sample):
                sample = f"{subj}_sample{sm}"
                # build_skeleton_index 从 Depth_Color 遍历目录结构（内容不读）
                (root / "Depth_Color" / adir / subj / sample).mkdir(parents=True, exist_ok=True)
                # 3D json：每帧 list 长度 1，keypoints[17,3] 米制 + scores
                pred = root / "Skeleton" / adir / subj / sample / "predictions"
                pred.mkdir(parents=True, exist_ok=True)
                for t in range(n_frame):
                    kp = rng.normal(0, 0.5, (17, 3)).astype(np.float32)
                    kp[0] = 0  # 骨盆原点
                    cf = rng.uniform(0.3, 0.99, 17).astype(np.float32)
                    (pred / f"{t:08d}.json").write_text(
                        json.dumps([{"keypoints": kp.tolist(), "keypoint_scores": cf.tolist()}]),
                        encoding="utf-8")
                # 2D npz：kp[17,2] 归一化 + conf 原始分数（logit 尺度）
                dst = thermal_out / f"{a}" / subj / f"{sample}.npz"
                dst.parent.mkdir(parents=True, exist_ok=True)
                kp2 = rng.normal(0, 0.5, (n_frame, 17, 2)).astype(np.float32)
                conf = rng.normal(2.0, 3.0, (n_frame, 17)).astype(np.float32)
                np.savez_compressed(dst, kp=kp2, conf=conf)
    # 让一个 clip 缺 2D npz（验证过滤）
    missing = thermal_out / "0" / "S00" / "S00_sample1.npz"
    if missing.is_file():
        missing.unlink()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dual_selftest_"))
    root = tmp / "HAR"
    thermal_out = tmp / "thermal"
    # build_skeleton_index 用 Depth_Color discovery
    (root / "Depth_Color").mkdir(parents=True, exist_ok=True)
    make_synth(root, thermal_out)
    print(f"[1] 合成数据: {root}")

    pairs, conf_global = build_dual_pairs(root, thermal_out, minmax=True)
    assert len(pairs) == 4 * 3 * 2 - 1, f"pairs={len(pairs)}（应 23，缺 1 个 2D）"
    print(f"[2] build_dual_pairs OK: pairs={len(pairs)} conf_global={conf_global}")

    ds = DualSkeletonDataset(pairs, thermal_out, 16, True, conf_norm="minmax",
                             rot_angle=30.0, conf_global=conf_global)
    x2, x3, aid, subj = ds[0]
    assert x2.shape == (16, 17, 3) and x3.shape == (16, 17, 3), (x2.shape, x3.shape)
    assert torch.isfinite(x2).all() and torch.isfinite(x3).all()
    print(f"[3] DualSkeletonDataset OK: x2={tuple(x2.shape)} x3={tuple(x3.shape)} "
          f"action={aid} subject={subj}")
    print(f"    x2[0,0]={x2[0,0].tolist()}（第3通道=conf∈[0,1]）")
    print(f"    x3[0,0]={x3[0,0].tolist()}（第3通道=z）")

    # 2D-only 单分支数据集（诊断用）——继承 DualSkeletonDataset
    ds2 = Thermal2DSkeletonDataset(pairs, thermal_out, 16, True, conf_norm="sigmoid")
    x2b, aidb, subjb = ds2[0]
    assert x2b.shape == (16, 17, 3) and torch.isfinite(x2b).all()
    assert 0.0 <= x2b[..., 2].min() <= x2b[..., 2].max() <= 1.0  # conf∈(0,1)
    print(f"[3b] Thermal2DSkeletonDataset OK: x2={tuple(x2b.shape)} conf∈(0,1) "
          f"action={aidb} subject={subjb}")
    # 回归：子类必须能调 _mirror（之前 50% 概率才触发的 bug，服务器 DataLoader worker 崩）
    feat = np.random.randn(4, 17, 3).astype(np.float32)
    m = DualSkeletonDataset._mirror(feat)
    assert m.shape == feat.shape and np.isfinite(m).all()
    assert np.allclose(m[:, :, 0], -feat[:, :, 0]) is False  # 交换关节后 x 不一定等于 -feat
    for k in range(10):  # 多次采样覆盖 mirror/非 mirror 两条分支
        a, _, _ = ds2[k % len(ds2)]
        assert a.shape == (16, 17, 3) and torch.isfinite(a).all()
    print("[3c] Thermal2D 继承 _mirror + 多次采样（覆盖 mirror/非 mirror）OK")

    # 模型：小 DSTformer 即可验证形状（不用预训练权重）
    backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=64, dim_rep=128, depth=2,
                         num_heads=4, num_joints=17, maxlen=243)
    model = DualBranchActionNet(backbone=backbone, dim_rep=128, num_classes=40,
                                dropout_ratio=0.5, hidden_dim=256, num_joints=17,
                                fusion="learn")
    b = 2
    a = torch.randn(b, 16, 17, 3)
    c = torch.randn(b, 16, 17, 3)
    fused, l2, l3 = model(a, c)
    assert fused.shape == (b, 40) and l2.shape == (b, 40) and l3.shape == (b, 40)
    w = float(model.fusion_weight().item())
    print(f"[4] DualBranchActionNet forward OK: fused={tuple(fused.shape)} "
          f"2d={tuple(l2.shape)} 3d={tuple(l3.shape)} fusion_w={w:.3f}")

    # 一轮 backward（fused + dual 两种 loss）
    crit = torch.nn.CrossEntropyLoss()
    y = torch.randint(0, 40, (b,))
    for mode in ["fused", "dual"]:
        opt = torch.optim.AdamW([{"params": model.parameters()}], lr=1e-3)
        opt.zero_grad()
        fused, l2, l3 = model(a, c)
        loss = crit(fused, y) if mode == "fused" else crit(l2, y) + crit(l3, y)
        loss.backward()
        opt.step()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        print(f"[5] backward({mode}) OK: loss={loss.item():.3f} "
              f"grad覆盖={sum(grads)}/{len(grads)}")

    # 每折重载 + 训练循环冒烟（train_dual_branch 核心逻辑）
    from torch.utils.data import DataLoader
    from src.dataset import build_balanced_sampler
    folds = split_by_subject(pairs, n_folds=3)
    tr_idx, va_idx = folds[0]
    tr_clips = [pairs[i] for i in tr_idx]
    va_clips = [pairs[i] for i in va_idx]
    tr_ds = DualSkeletonDataset(tr_clips, thermal_out, 16, True, conf_norm="sigmoid", rot_angle=30.0)
    va_ds = DualSkeletonDataset(va_clips, thermal_out, 16, False, conf_norm="sigmoid", rot_angle=30.0)
    sampler = build_balanced_sampler([c.action_id for c in tr_clips])
    tr_loader = DataLoader(tr_ds, batch_size=8, sampler=sampler, num_workers=0, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=0)
    bb = DSTformer(dim_in=3, dim_out=3, dim_feat=64, dim_rep=128, depth=2, num_heads=4,
                   num_joints=17, maxlen=243)
    m2 = DualBranchActionNet(backbone=bb, dim_rep=128, num_classes=40, dropout_ratio=0.5,
                             hidden_dim=256, num_joints=17, fusion="learn")
    opt = torch.optim.AdamW([{"params": m2.parameters()}], lr=1e-3)
    crit2 = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for x2b, x3b, yb, _ in tr_loader:
        opt.zero_grad()
        f_, _, _ = m2(x2b, x3b)
        loss = crit2(f_, yb)
        loss.backward()
        opt.step()
    # eval
    m2.eval()
    with torch.no_grad():
        corr = tot = 0
        for x2b, x3b, yb, _ in va_loader:
            f_, _, _ = m2(x2b, x3b)
            corr += (f_.argmax(-1) == yb).sum().item()
            tot += yb.numel()
    print(f"[6] mini 训练 + eval 冒烟 OK: val_acc={corr/max(tot,1):.3f}")

    print("\n==== 全部自测通过 ====")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
