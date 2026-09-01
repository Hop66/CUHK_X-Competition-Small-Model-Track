"""
ActionNet —— MotionBERT 骨架动作识别分类头。

来源: https://github.com/Walter0807/MotionBERT (Apache-2.0)
文件: lib/model/model_action.py

输入:  [N, M, T, 17, 3]   (M=人数, 单人 M=1)
输出:  logits [N, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60,
                 num_joints=17, hidden_dim=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        # feat: (N, M, T, J, C)
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0.,
                 version='class', hidden_dim=2048, num_joints=17, part_aware=False):
        """part_aware=True: 可学习部位权重（每关节 1 个，初始=1.0 原行为，训练中学增强/抑制）。
        对应 ST-GCN edge-importance / CTR-GCN part attention 思想：动作由不同部位主导。
        """
        super().__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        assert version in ('class', 'embed')
        if version == 'class':
            self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep,
                                                 num_classes=num_classes, num_joints=num_joints,
                                                 hidden_dim=hidden_dim)
        else:
            self.head = None
        self.part_aware = part_aware
        if part_aware:
            # 1 + tanh(logits)：logits=0 → 权重=1.0（完全原行为），范围 (0,2)
            self.part_logits = nn.Parameter(torch.zeros(num_joints))

    def forward(self, x):
        # x: (N, M, T, 17, 3)
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        feat = self.backbone.get_representation(x)   # (N*M, T, J, dim_rep)
        feat = feat.reshape(N, M, T, self.feat_J, -1)
        if self.part_aware:
            w = (1 + torch.tanh(self.part_logits)).view(1, 1, 1, self.feat_J, 1)
            feat = feat * w
        out = self.head(feat)
        return out


# ---------------- 双专家（BHaRNet-B 最小复刻：体/上肢分离） ----------------
# 依据 diagnose_skeleton_parts.py：手精细(25)组 val_acc 0.451 << 下肢全身(11) 0.753，
# 但手组内大量 ≥0.4 可分信号（Wash_face 0.714 / Tap_keyboard 0.703 / Eat_food 0.625 / Take_selfie 0.700…）
# → 分离"上肢专家"与"躯干/下肢专家"，各司其职，减少全身平均化对手类的稀释。
# H3.6M-17：0 骨盆 1-6 腿 7 脊柱 8 胸脊 9 颈 10 头 11-13 左臂 14-16 右臂
UPPER_JOINTS = [8, 9, 10, 11, 12, 13, 14, 15, 16]   # 胸脊+头颈+双臂（手/上肢主导 28 类）
LOWER_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7]             # 骨盆+双腿+脊柱（下肢/全身 12 类）


class ActionHeadSubset(nn.Module):
    """ActionHeadClassification 变体：只取指定关节子集再 GAP→fc（专家头）。"""
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60,
                 num_joints=17, hidden_dim=2048, joint_idx=None):
        super().__init__()
        self.joint_idx = joint_idx
        j = len(joint_idx) if joint_idx is not None else num_joints
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * j, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        # feat: (N, M, T, J, C)；joint_idx 切片 J 维
        if self.joint_idx is not None:
            feat = feat[..., self.joint_idx, :]
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class ActionNetDualExpert(nn.Module):
    """BHaRNet-B 最小复刻：共享 backbone，体/上肢两个 disjoint 专家头。
    forward 默认返回确定性 logit 和（推理）；need_components=True 返回 (u, l, sum)（训练）。
    训练 loss（BHaRNet III-B/IV-D）：
      L = λu·CE(ŷu) + λl·CE(ŷl) + λcpl·CE((ŷu+ŷl)/2) + λnor·CE(NoisyOR(ŷu,ŷl))
    NoisyOR: p_k = 1 - ∏_i (1 - softmax(ŷ_{i,k})) → 至少一专家置信即支持该类。
    """
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0.,
                 hidden_dim=2048, num_joints=17,
                 upper_joints=UPPER_JOINTS, lower_joints=LOWER_JOINTS):
        super().__init__()
        self.backbone = backbone
        self.upper_joints = list(upper_joints)
        self.lower_joints = list(lower_joints)
        self.head_upper = ActionHeadSubset(dropout_ratio, dim_rep, num_classes,
                                           num_joints, hidden_dim, self.upper_joints)
        self.head_lower = ActionHeadSubset(dropout_ratio, dim_rep, num_classes,
                                           num_joints, hidden_dim, self.lower_joints)

    def forward(self, x, need_components=False):
        # x: (N, M, T, 17, 3)
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)
        feat = self.backbone.get_representation(x)   # (N*M, T, J, dim_rep)
        feat = feat.reshape(N, M, T, J, -1)
        lo_u = self.head_upper(feat)   # (N, num_classes)
        lo_l = self.head_lower(feat)
        lo_s = lo_u + lo_l
        if need_components:
            return lo_u, lo_l, lo_s
        return lo_s
