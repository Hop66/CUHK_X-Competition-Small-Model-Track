"""VideoMAE-S（ViT-S）接入 —— 手写结构 + 官方权重加载（无 transformers 依赖）。

基于 inspect 实测权重结构（VideoMAE-S K400 1600ep pretrain）：
  - encoder.patch_embed.proj: Conv3d(3→384, kernel/stride=(2,16,16))
  - encoder.blocks.{i}.attn: qkv(Linear 384→1152, bias=False) + q_bias/v_bias 分离 + proj
  - encoder.blocks.{i}.mlp: fc1(384→1536) + fc2(1536→384)
  - encoder.norm: LayerNorm(384)
  - ⚠️ 预训练权重不含 pos_embed/cls_token → 随机初始化（微调学）
  - mask_token/decoder 为预训练专用，丢弃

输入 [B,3,T,H,W]（T=16, H=W=128 → 8×8×8=512 patch tokens + cls = 513）。
归一化：VideoMAE 用 ImageNet mean/std（0.485/0.229），需在数据集侧配合。
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 结构
# ---------------------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    """Conv3d patch embedding（时间 stride 2、空间 16×16）。"""

    def __init__(self, in_channels=3, embed_dim=384, t_kernel=2, s_kernel=16):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=(t_kernel, s_kernel, s_kernel),
                              stride=(t_kernel, s_kernel, s_kernel))

    def forward(self, x):
        # x: [B,3,T,H,W] -> [B,384,T/2,H/16,W/16]
        return self.proj(x)


class Attention(nn.Module):
    """ViT attention，qkv bias 分离（q_bias/v_bias）。"""

    def __init__(self, dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # qkv bias：q 用 q_bias，k 用 0，v 用 v_bias
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = self.qkv(x) + qkv_bias                          # [B,N,1152]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B,H,N,D]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim=384, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoMAEEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4, num_frames=16, img_size=128, t_patch=2, s_patch=16):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, t_patch, s_patch)
        n_t = num_frames // t_patch
        n_sp = img_size // s_patch
        num_patches = n_t * n_sp * n_sp                        # 8*8*8 = 512
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                                # [B,384,8,8,8]
        x = x.flatten(2).transpose(1, 2)                       # [B,512,384]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                         # [B,513,384]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                         # [B,384] cls token


class VideoMAES(nn.Module):
    """VideoMAE-S 分类模型：encoder + 40 类头。输入 [B,3,T,H,W]。"""

    def __init__(self, num_classes=40, in_channels=3, num_frames=16, img_size=128):
        super().__init__()
        self.encoder = VideoMAEEncoder(in_channels=in_channels, num_frames=num_frames,
                                       img_size=img_size)
        self.head = nn.Linear(self.encoder.pos_embed.shape[-1], num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# 权重加载（官方 pretrain：{"model": {...encoder + decoder...}}）
# ---------------------------------------------------------------------------
def load_videomae_pretrained(model, ckpt_path, device="cpu"):
    """加载官方 VideoMAE-S pretrain 权重（encoder 部分），验证加载覆盖率。

    只加载 encoder.patch_embed/blocks/norm；pos_embed/cls_token 随机（预训练无）；
    mask_token/decoder 丢弃（预训练专用）。
    返回 (matched, total_encoder_params)。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # 提取 encoder 参数（排除 pos_embed/cls_token/decoder/mask_token）
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.") and "pos_embed" not in k and "cls_token" not in k}
    model_dict = model.encoder.state_dict()
    # 应加载的 key（pos_embed/cls_token 预训练无，随机初始化，不算覆盖率分母）
    loadable = {k for k in model_dict if "pos_embed" not in k and "cls_token" not in k}
    matched = discarded = 0
    for k, v in enc_sd.items():
        if k in loadable and model_dict[k].size() == v.size():
            model_dict[k] = v
            matched += 1
        else:
            discarded += 1
    model.encoder.load_state_dict(model_dict)
    total = len(loadable)
    print(f"[videomae] 加载 encoder: matched={matched}/{total} discarded={discarded} "
          f"（覆盖率 {matched/total:.1%}，pos_embed/cls_token 随机初始化）", flush=True)
    return matched, total
