"""
CUHK-X —— 跨被试划分（subject-fold）

测试被试从未出现在训练中，随机切 clip 会通过"身份/房间捷径"高估。
因此必须按被试分组：同一被试的所有 clip 只在 train 或 val 之一。
"""

from typing import Dict, List, Tuple

import numpy as np


def split_by_subject(
    clips,
    n_folds: int = 3,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Args:
        clips: list[ClipIndex]（含 .subject 与 .action_id）
        n_folds: 折数。18 被试 / 3 = 每折 6 个被试做验证。
    返回: [(train_idx, val_idx), ...] 每折一对索引。
    """
    subj_to_idx: Dict[str, List[int]] = {}
    for i, c in enumerate(clips):
        subj_to_idx.setdefault(c.subject, []).append(i)

    subjects = sorted(subj_to_idx.keys())
    n_subj = len(subjects)
    assert n_subj >= n_folds, f"subjects({n_subj}) < folds({n_folds})"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(subjects)

    folds = []
    fold_size = n_subj // n_folds
    for f in range(n_folds):
        start = f * fold_size
        end = n_subj if f == n_folds - 1 else (f + 1) * fold_size
        val_subjs = set(perm[start:end].tolist())
        val_idx = [i for s in val_subjs for i in subj_to_idx[s]]
        train_idx = [i for s in subjects if s not in val_subjs for i in subj_to_idx[s]]
        # 防泄漏断言
        assert set(clips[i].subject for i in train_idx).isdisjoint(
            set(clips[i].subject for i in val_idx))
        folds.append((train_idx, val_idx))
    return folds
