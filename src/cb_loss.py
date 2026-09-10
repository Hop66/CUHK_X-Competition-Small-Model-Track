"""Class-Balanced Loss（有效样本数加权，针对长尾分布）

基于 Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019).
核心思想：每个类的权重 = (1 - beta) / (1 - beta^{n_c})，其中 n_c 是该类样本数，beta∈[0,1)。
beta→1 时趋近均匀；beta→0 时趋近逆频率加权。

用法：
    from src.cb_loss import ClassBalancedLoss
    crit = ClassBalancedLoss(class_counts, num_classes=40, beta=0.9999)
    loss = crit(logits, targets)  # 自动按类加权
"""

import torch
import torch.nn as nn


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Cross-Entropy Loss.

    Args:
        class_counts: list/array of per-class sample counts (length = num_classes)
        num_classes: number of classes
        beta: smoothing parameter (0~1); typical values: 0.9, 0.99, 0.999, 0.9999
        label_smoothing: optional label smoothing (same semantics as nn.CrossEntropyLoss)
    """

    def __init__(self, class_counts, num_classes: int = 40, beta: float = 0.9999,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.label_smoothing = label_smoothing

        # 计算每类权重: w_c = (1-beta) / (1-beta^{n_c})
        counts = torch.tensor(class_counts, dtype=torch.float32)
        # 防止除零：n_c=0 的类给极小权重
        counts = counts.clamp(min=1e-8)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # 归一化到均值为 1
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C], targets: [N]"""
        ce_per_sample = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        # 按目标类的权重加权（权重随输入设备走，防调用处未 .to(device) 的 CPU/GPU 错位）
        w = self.class_weights.to(targets.device)[targets]
        return (ce_per_sample * w).mean()
