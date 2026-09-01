"""FrameNet —— 14th-place 逐帧 2D CNN + per-frame logits mean（从零训练，无预训练）。

它为什么在"同人高度相关 + 低数据 + 无预训练"下可能高效（机制笔记）：
  1. 每帧独立分类 → logits 平均 = 隐式 bagging 投票，靠相邻帧冗余压方差；
  2. 无任何时序递归/池化 → 难以把"特定人的房间风格/步态"当跨帧身份捷径学走
     （3D 时序模型在同人高度相关的数据上最容易犯这个泄漏错误——这正是作者
       强调 subject-wise 的深层原因）；
  3. 局部/短促动作只要 1-2 帧落在关键段即贡献投票，不被长 clip 稀释；
  4. 小模型（~3-5M）在 2891/40 类小数据上过拟合风险低。
用途：作为轻量独立热像模型，量化后几乎不占 100MB 预算。

输入: [B, T, 3, H, W]（与 src/dataset.ThermalVideoDataset 输出直接兼容）
输出: [B, num_classes]（per-frame logits 平均）
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, a, b, s=1):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b))
        self.d = (nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))
                  if (a != b or s != 1) else nn.Identity())

    def forward(self, x):
        return torch.relu(self.c(x) + self.d(x))


class FrameNet(nn.Module):
    """逐帧 2D CNN → 每帧 40 类 logits → 时间维度平均（帧级投票）。"""

    def __init__(self, num_classes=40):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            Block(32, 32), Block(32, 64, 2), Block(64, 128, 2), Block(128, 256, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        z = self.f(x.reshape(b * t, c, h, w)).flatten(1)   # (b*t, 256)
        return self.fc(z).reshape(b, t, -1).mean(1)        # (b, num_classes) = 帧级 logits 平均

    @staticmethod
    def n_params(model):
        return sum(p.numel() for p in model.parameters())


class FrameNetPretrained(nn.Module):
    """逐帧 2D 预训练 CNN + per-frame logits mean（notebook 帧级投票 + 强预训练骨干）。

    修正 FrameNet(从零) 判负的过早结论：0.8 组方法的本质是"逐帧 2D 分类 + logits 时间平均"，
    它并不是绑定"从零小 CNN"。用 torchvision ImageNet 预训练 2D CNN（resnet18/34）复刻同一聚合。
    traj_dim>0：每帧特征 cat 每帧轨迹 [cx,cy,bw,bh]（逐帧跟人时的位移/尺度显式通道）→ 帧级 logits 平均。
    """
    def __init__(self, features, feat_dim, num_classes=40, traj_dim=0):
        super().__init__()
        self.f = features       # 预训练 2D CNN（已去分类头），输出 (b*t, feat_dim)
        self.fc = nn.Linear(feat_dim + traj_dim, num_classes)

    def forward(self, x, traj=None):
        b, t, c, h, w = x.shape
        z = self.f(x.reshape(b * t, c, h, w))       # (b*t, feat_dim)
        if self.fc.in_features > z.shape[1]:        # 轨迹通道开启（traj_dim>0）
            if traj is None:                        # 无轨迹时补零（切换可比）
                traj = torch.zeros(b * t, self.fc.in_features - z.shape[1], device=z.device)
            else:
                traj = traj.reshape(b * t, -1)
            z = torch.cat([z, traj], dim=-1)
        return self.fc(z).reshape(b, t, -1).mean(1)


def build_frame_net_pretrained(arch="resnet18", num_classes=40, traj_dim=0):
    """torchvision ImageNet 预训练 2D CNN → per-frame logits mean 模型（可含轨迹通道）。"""
    import torchvision
    if arch == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=w)
    elif arch == "resnet34":
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet34(weights=w)
    else:
        raise ValueError(arch)
    dim = net.fc.in_features
    net.fc = nn.Identity()
    return FrameNetPretrained(net, dim, num_classes, traj_dim=traj_dim)
