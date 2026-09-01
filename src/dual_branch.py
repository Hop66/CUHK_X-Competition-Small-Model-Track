"""DualBranchActionNet —— 共享 backbone + 双输入头（热像 2D [x,y,conf] + NYX 3D [x,y,z]）。

方法论（2026-08-23 顶会调研定稿，见 /memories/repo/thermal-pose-bridge.md）：
  * 共享 DSTformer encoder：两模态共享运动先验 + 参数（100MB 约束下省 ~40MB），
    缓解小数据过拟合；风险=模态干扰 → 用模态特定输入头吸收域差（多模态共享-encoder 成熟 trade-off）。
  * 双输入头：input_head_2d 处理 [x,y,conf]（热像 2D），input_head_3d 处理 [x,y,z]（NYX 3D）。
    两者都是 Linear(3, dim_feat)，用预训练 joints_embed 初始化（MotionBERT 多源预训练已见 2D+3D）。
  * 双分类头 + logits 加权融合（默认可学习权重，sigmoid 约束 (0,1)，初始 0.5）。
  * 绕竖直轴旋转增强在数据集里做（补两相机视点差，NTU cross-view 标准做法）。

forward: (x_2d [B,T,17,3], x_3d [B,T,17,3]) -> (logits_fused, logits_2d, logits_3d)
"""

import torch
import torch.nn as nn

from src.motionbert.action_net import ActionHeadClassification


class DualBranchActionNet(nn.Module):
    def __init__(self, backbone, dim_rep: int = 512, num_classes: int = 40,
                 dropout_ratio: float = 0.5, hidden_dim: int = 512,
                 num_joints: int = 17, fusion: str = "learn"):
        super().__init__()
        self.backbone = backbone  # DSTformer，仅复用 encoder 部分（blocks/norm/pre_logits/pos_embed/...）
        dim_feat = backbone.dim_feat

        # 双输入头：都是 Linear(3, dim_feat)，用预训练 joints_embed 初始化（多源预训练兼容 2D+3D）
        w = backbone.joints_embed.weight.data
        b = backbone.joints_embed.bias.data
        self.input_head_2d = nn.Linear(3, dim_feat)
        self.input_head_3d = nn.Linear(3, dim_feat)
        self.input_head_2d.weight.data.copy_(w)
        self.input_head_2d.bias.data.copy_(b)
        self.input_head_3d.weight.data.copy_(w)
        self.input_head_3d.bias.data.copy_(b)

        # 替换 backbone 的 joints_embed / head（训练用不到，且避免出现在 optimizer）
        self.backbone.joints_embed = nn.Identity()
        self.backbone.head = nn.Identity()

        # 双分类头
        self.head_2d = ActionHeadClassification(
            dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes,
            num_joints=num_joints, hidden_dim=hidden_dim)
        self.head_3d = ActionHeadClassification(
            dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes,
            num_joints=num_joints, hidden_dim=hidden_dim)

        # logits 融合权重：learn=sigmoid(可学参数) 初始 0.5；fixed=固定
        self.fusion = fusion
        if fusion == "learn":
            self.logit_w = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("logit_w", torch.full((1,), 0.5))
        self.feat_J = num_joints

    def _branch(self, x: torch.Tensor, input_head: nn.Module) -> torch.Tensor:
        """单个分支：input_head + 共享 DSTformer encoder → [B, T, J, dim_rep]。

        复刻 DSTformer.forward 的 encoder 部分（joints_embed 替换为外部 input_head）。
        """
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        x = input_head(x)                        # [B*F, J, dim_feat]
        x = x + self.backbone.pos_embed          # [1,17,dim_feat] 广播
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C) + self.backbone.temp_embed[:, :F, :, :]  # [1,F,1,dim_feat] 广播
        x = x.reshape(B * F, J, C)
        x = self.backbone.pos_drop(x)
        for idx, (blk_st, blk_ts) in enumerate(zip(self.backbone.blocks_st,
                                                   self.backbone.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.backbone.att_fuse:
                att = self.backbone.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.backbone.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.backbone.pre_logits(x)          # [B, F, J, dim_rep]
        return x

    def fusion_weight(self) -> torch.Tensor:
        """返回 (0,1) 内的融合权重：learn=sigmoid(可学参数)；fixed=固定值。"""
        if self.fusion == "learn":
            return torch.sigmoid(self.logit_w)
        return self.logit_w

    def forward(self, x_2d: torch.Tensor, x_3d: torch.Tensor):
        """x_2d [B,T,17,3]（热像 2D [x,y,conf]），x_3d [B,T,17,3]（NYX 3D [x,y,z]）。"""
        rep_2d = self._branch(x_2d, self.input_head_2d)   # [B,T,17,dim_rep]
        rep_3d = self._branch(x_3d, self.input_head_3d)
        logits_2d = self.head_2d(rep_2d.unsqueeze(1))     # [B,num_classes]
        logits_3d = self.head_3d(rep_3d.unsqueeze(1))
        w = self.fusion_weight()
        logits = w * logits_2d + (1.0 - w) * logits_3d
        return logits, logits_2d, logits_3d
