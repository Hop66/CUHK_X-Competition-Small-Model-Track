"""骨架运动流网络（thermal 双流 / main 双流共用）。

MotionNet：motion[T,K29] → 两层 1D conv（保留时序）→ 时间聚合 → logits。
"""
import torch.nn as nn

from src.skeleton_motion import K_DIM


class MotionNet(nn.Module):
    def __init__(self, in_dim: int = K_DIM, hid: int = 64, num_classes: int = 40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hid, 5, padding=2, bias=False), nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid, hid, 5, padding=2, bias=False), nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True))
        self.head = nn.Linear(hid, num_classes)

    def forward(self, m):                    # m: [B,T,29]
        z = self.conv(m.permute(0, 2, 1))    # [B,hid,T]
        return self.head(z.mean(dim=2))      # [B,40]
