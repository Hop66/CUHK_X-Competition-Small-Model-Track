"""
CUHK-X —— 视频主干（Step 1）

两个候选：
- R2Plus1D-18（Kinetics-400 预训练，验证过的主干；输入 4ch 需改造 stem）
- TSM + ResNet18（轻量备选，ImageNet 预训练 + 时序移位）

注意 Depth_Color 是伪彩色深度图（非自然 RGB），预训练是"结构迁移"而非"外观迁移"。
"""

import torch
import torch.nn as nn


def adapt_conv3d(conv: nn.Conv3d, channels: int) -> nn.Conv3d:
    """把 3 通道的 Conv3d stem 改造为 channels 通道：复制前 3 通道，新增通道取均值。"""
    repl = nn.Conv3d(channels, conv.out_channels, conv.kernel_size,
                     conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        repl.weight[:, :3] = conv.weight
        repl.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True).expand(-1, channels - 3, -1, -1, -1)
        if conv.bias is not None:
            repl.bias.copy_(conv.bias)
    return repl


def adapt_conv2d(conv: nn.Conv2d, channels: int) -> nn.Conv2d:
    repl = nn.Conv2d(channels, conv.out_channels, conv.kernel_size,
                     conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        repl.weight[:, :3] = conv.weight
        repl.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True).expand(-1, channels - 3, -1, -1)
        if conv.bias is not None:
            repl.bias.copy_(conv.bias)
    return repl


class R2Plus1D18(nn.Module):
    """torchvision r2plus1d_18，4ch 输入，40 类。"""

    def __init__(self, num_classes: int = 40, in_channels: int = 4, pretrained: bool = True):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        try:
            net = r2plus1d_18(weights=weights)
        except Exception:
            net = r2plus1d_18(weights=None)
        if in_channels != 3:
            net.stem[0] = adapt_conv3d(net.stem[0], in_channels)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, num_classes))

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.head(self.encoder(x))


class _Conv2Plus1D(nn.Sequential):
    """老版 torchvision Conv2Plus1D：midplanes 按 (in,out) 自算，忽略传入值。

    当前 torchvision 的 Conv2Plus1D 接受外部传入的 midplanes（由 BasicBlock 统一算），
    而 IG-65M 权重来自老版（每个 Conv2Plus1D 自己算 midplanes），两者在
    inplanes≠planes 的 block 上会产生尺寸不一致（如 layer2.0.conv2 288 vs 230）。
    此实现复刻老版公式，匹配 IG-65M 权重。
    """

    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)
        super().__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding), bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class R2Plus1D34(nn.Module):
    """R(2+1)D-34（IG-65M + Kinetics 预训练，70+ 队同款骨干）。

    架构 = VideoResNet(BasicBlock, [老版Conv2Plus1D]*4, layers=[3,4,6,3])，61.8M 参数。
    权重来自 moabitcoin/ig65m-pytorch（GitHub releases，IG-65M+Kinetics clip32）。
    fp32 约 247MB，超 100MB 需 int8/int6 量化（Step 5）。
    """

    def __init__(self, num_classes: int = 40, in_channels: int = 4, weights_path: str = None,
                 traj_dim: int = 0):
        super().__init__()
        from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem
        net = VideoResNet(block=BasicBlock, conv_makers=[_Conv2Plus1D] * 4,
                          layers=[3, 4, 6, 3], stem=R2Plus1dStem, num_classes=400)
        if weights_path is not None:
            sd = torch.load(weights_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = net.load_state_dict(sd, strict=False)
            print(f"R2Plus1D34: 加载 IG-65M 权重，missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
        if in_channels != 3:
            net.stem[0] = adapt_conv3d(net.stem[0], in_channels)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        # 可选：显式位移/尺度轨迹分支（每一帧 [cx,cy,bw,bh] → 1D conv 保留时序动态 → 时间聚合 → 拼特征）
        # 铁律：不能对轨迹做时间平均——那会把"走动/走近走远"的动态抹成静态平均框，轨迹即失去意义。
        # 用 Conv1d(k=5) 先抓局部时间模式，再 mean → 32 维（保留"框随时间怎么动"）。
        self.traj_dim = traj_dim
        traj_hidden = 32
        self.traj_mlp = (nn.Sequential(
            nn.Conv1d(traj_dim, traj_hidden, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
        ) if traj_dim > 0 else None)
        in_head = feat + traj_hidden if traj_dim > 0 else feat
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_head, num_classes))

    def forward(self, x, traj=None):
        x = x.permute(0, 2, 1, 3, 4)
        f = self.encoder(x)          # (B, feat)
        if self.traj_mlp is not None:
            if traj is not None and traj.dim() == 3:
                # [B,T,C] → [B,C,T] → conv1d → [B,32,T] → mean(时间) → [B,32]
                t = self.traj_mlp(traj.permute(0, 2, 1)).mean(dim=2)
            else:
                t = torch.zeros(f.shape[0], 32, device=f.device)
            f = torch.cat([f, t], dim=1)
        return self.head(f)


class TemporalShift(nn.Module):
    def __init__(self, n_segment: int = 16, fold_div: int = 4):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):  # [B*T, C, H, W]
        nt, c, h, w = x.size()
        if nt % self.n_segment != 0:
            return x
        nb = nt // self.n_segment
        x = x.view(nb, self.n_segment, c, h, w)
        fold = c // self.fold_div
        if fold == 0:
            return x.view(nt, c, h, w)
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class TSMResNet18(nn.Module):
    """ResNet18 + TSM（时序移位注入 layer3 后），4ch 输入。"""

    def __init__(self, num_classes: int = 40, in_channels: int = 4,
                 n_segment: int = 16, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = tvm.resnet18(weights=w)
        except Exception:
            net = tvm.resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = adapt_conv2d(net.conv1, in_channels)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.tsm = TemporalShift(n_segment=n_segment)
        self.layer4 = net.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.tsm(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x.view(B, T, -1).mean(dim=1)  # 帧级 logits 平均（TSM 惯例）


def build_model(name: str, num_classes: int = 40, in_channels: int = 4,
                n_segment: int = 16, pretrained: bool = True, weights_path=None):
    if name == "r2plus1d":
        return R2Plus1D18(num_classes, in_channels, pretrained)
    if name == "r2plus1d34":
        return R2Plus1D34(num_classes, in_channels, weights_path)
    if name == "tsm_resnet18":
        return TSMResNet18(num_classes, in_channels, n_segment, pretrained)
    raise ValueError(f"unknown backbone: {name}")
