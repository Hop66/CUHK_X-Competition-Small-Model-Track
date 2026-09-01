"""骨干对比工厂（B 线实验，独立于 model.py，不触碰现有主线）。

支持骨干（torchvision Kinetics 预训练，小型标准 CNN/Transformer）：
  - r2plus1d34 : 现有主线（IG-65M，fold0 基线 thermal 0.6339 / main 0.6609）
  - x3d_m / x3d_l : 轻量高效 CNN
  - s3d         : 轻量 CNN
  - mvit_v2_s   : Multiscale Vision Transformer（更强）
  - swin3d_t    : Swin3D 小型 Transformer

规则：no large pretrained backbones = 禁 LLM/VLM 基础模型；小型标准预训练
（如 ResNet18 ~44MB）允许。这些 backbone 参数 3-35M（int5 2-22MB），属小型。
输入：3ch（thermal 直接匹配预训练域）；4ch（Depth+IR 用 adapt_conv3d 扩 stem）。
用法配合 scripts/train_backbone_compare.py。
"""

import torch
import torch.nn as nn

from src.model import adapt_conv3d


def _replace_final_linear(module, num_classes):
    """把 torchvision 视频模型的最后一个分类层替换为 40 类（Linear 或 1x1x1 Conv3d，如 S3D）。"""
    if isinstance(module, nn.Linear):
        return nn.Linear(module.in_features, num_classes)
    if isinstance(module, nn.Conv3d) and all(k == 1 for k in module.kernel_size):
        return nn.Conv3d(module.in_channels, num_classes, 1)
    # Sequential 容器：替换最后一个 Linear 或 1x1x1 Conv3d 的层
    for i in range(len(module) - 1, -1, -1):
        if isinstance(module[i], nn.Linear):
            module[i] = nn.Linear(module[i].in_features, num_classes)
            return module
        if isinstance(module[i], nn.Conv3d) and all(k == 1 for k in module[i].kernel_size):
            module[i] = nn.Conv3d(module[i].in_channels, num_classes, 1)
            return module
    raise RuntimeError(f"未找到可替换的分类层: {type(module)}")


def _adapt_in_channels(net, name, in_channels):
    """4ch → 扩展输入 stem conv（3→4），匹配预训练权重（新通道用预训练第 1 通道均值初始化）。"""
    if in_channels == 3:
        return
    if name in ("x3d_m", "x3d_l"):
        adapt_conv3d(net.blocks[0].conv[0], in_channels)
    elif name == "s3d":
        # S3D 输入 conv: features[0] 是 Conv3d
        conv = net.features[0]
        if isinstance(conv, nn.Conv3d):
            net.features[0] = _new_conv3d(conv, in_channels)
        else:
            adapt_conv3d(net.features[0].conv[0], in_channels)
    elif name == "mvit_v2_s":
        conv = net.stem.conv
        if isinstance(conv, nn.Conv3d):
            net.stem.conv = _new_conv3d(conv, in_channels)
        else:
            adapt_conv3d(net.stem.conv.proj, in_channels)
    elif name == "swin3d_t":
        conv = net.patch_embed.proj
        if isinstance(conv, nn.Conv3d):
            net.patch_embed.proj = _new_conv3d(conv, in_channels)
        else:
            adapt_conv3d(net.patch_embed.proj, in_channels)


def _new_conv3d(conv: nn.Conv3d, in_channels: int) -> nn.Conv3d:
    """新建 in_channels 输入的 Conv3d，新通道用第 0 通道权重均值初始化。"""
    new = nn.Conv3d(in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        new.weight[:, 0] = conv.weight[:, 0]  # 新通道复制第 0 通道
        if in_channels == 4:
            new.weight[:, 3] = conv.weight[:, 1]  # IR 通道近似用第 1 通道
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_backbone(name: str, num_classes: int = 40, in_channels: int = 3,
                   weights_path: str = None, traj_dim: int = 0):
    """构建指定骨干（torchvision Kinetics 预训练 / r2plus1d34 IG-65M）。"""
    if name == "r2plus1d34":
        from src.model import R2Plus1D34
        return R2Plus1D34(num_classes, in_channels, weights_path, traj_dim=traj_dim)

    import torchvision

    if name in ("x3d_m", "x3d_l"):
        from torchvision.models.video import X3D_M_Weights, X3D_L_Weights, x3d_m, x3d_l
        w = X3D_M_Weights if name == "x3d_m" else X3D_L_Weights
        net = x3d_m(weights=w.KINETICS400_V1) if name == "x3d_m" else x3d_l(weights=w.KINETICS400_V1)
        _adapt_in_channels(net, name, in_channels)
        net.head = _replace_final_linear(net.head, num_classes)
    elif name == "s3d":
        from torchvision.models.video import S3D_Weights, s3d
        net = s3d(weights=S3D_Weights.KINETICS400_V1)
        _adapt_in_channels(net, name, in_channels)
        net.classifier = _replace_final_linear(net.classifier, num_classes)
    elif name == "mvit_v2_s":
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
        net = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        _adapt_in_channels(net, name, in_channels)
        net.head = _replace_final_linear(net.head, num_classes)
    elif name == "swin3d_t":
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t
        net = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        _adapt_in_channels(net, name, in_channels)
        net.head = _replace_final_linear(net.head, num_classes)
    else:
        raise ValueError(f"未知骨干: {name}")
    return net
