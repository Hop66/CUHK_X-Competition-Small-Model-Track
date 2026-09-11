"""MixStyle (ICLR'21, Zhou et al.) — Domain Generalization 轻量模块, 零推理代价.

P1 (2026-09-11, 见 idea.md §5 第二优先级):
  - 训练: 以概率 p 对 batch 内随机配对的 feature stats(μ,σ) 做插值 → 跨域增广。
  - 测试: 关闭 (pass-through) → 推理零成本。
  - 与"难对重权/GRL"等不同: 直接针对 subject/environment shift, 不碰难对规则。
插在哪: 时序移位后、分类头前; 对 [B*T, C, H, W] 特征做 per-sample stats mixing。

用法: 在模型 forward 里 `x = self.ms(x, T)` .
"""
import random

import torch
import torch.nn as nn


def _get_bn_stats(x):
    """返回 (μ, σ): 逐 [N,C] 通道统计 (N=B*T, C 通道)。"""
    N, C = x.size(0), x.size(1)
    mu = x.reshape(N, C, -1).mean(dim=2)            # [N,C]
    sig = x.reshape(N, C, -1).std(dim=2) + 1e-6     # [N,C]
    return mu, sig


def _apply_mixstats(x, mu, sig):
    """用给定 μ,σ 重缩放特征 (IN). x [N,C,H,W]."""
    N, C = x.size(0), x.size(1)
    mu = mu.unsqueeze(-1).unsqueeze(-1)
    sig = sig.unsqueeze(-1).unsqueeze(-1)
    mu_cur, sig_cur = _get_bn_stats(x)
    mu_cur = mu_cur.unsqueeze(-1).unsqueeze(-1)
    sig_cur = sig_cur.unsqueeze(-1).unsqueeze(-1)
    return (x - mu_cur) / sig_cur * sig + mu


class MixStyle(nn.Module):
    """跨域 style mixing: 概率 p 内, 随机选另一行做 stats 插值. 测试(p=0)恒等."""
    def __init__(self, p=0.5, alpha=0.1, enable=True):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.enable = enable
        self._training = True

    def set_enable(self, e):
        self.enable = e

    def forward(self, x):
        if (not self._training) or (not self.enable) or (self.p <= 0):
            return x
        if random.random() > self.p:
            return x
        N, C, H, W = x.shape
        lam = random.uniform(self.alpha, 1 - self.alpha)
        perm = torch.randperm(N, device=x.device)
        mu, sig = _get_bn_stats(x)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return _apply_mixstats(x, mu_mix, sig_mix)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._training = True
        else:
            # eval 时 _training=False → pass-through; 保留 enable 可对全链路开关
            self._training = False
        return self
