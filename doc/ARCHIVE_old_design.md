# 旧框架归档（2026-08-15 重构前）

> 本文件是旧设计/代码的**存档**，仅用于防止有价值的事实与可复用代码丢失。
> **新规划已完全独立于本文件**（见 `doc/PLAN.md`），本文件内容不构成新方案的约束或依据。

---

## 1. 已验证的数据事实（本地实测 + 论文）

### 1.1 竞赛形态

- CUHK-X 小模型赛道：**RGB-Free**、模型 **≤100MB**、**跨被试**、40 类日常动作。
- 测试集：**405 个 clip**（`small_model_track_test/SM_test_XXXX/`），每个 clip 含全部 6 个模态子文件夹：`Depth_Color/ IR/ Skeleton/ Thermal/ IMU/ Radar/`。
- 提交格式：`path,prediction`（0-39），405 行。`test.csv` 给出 path，`sample_submission.csv` 是样例。
- 类名映射见 `data/Training/class_mapping.csv`（0_Wash_face ... 39_Take_body_temperature）。
- 训练集（服务器上，本地无）：约 2891~3000 clips，结构预期为 `HAR/data/<Modality>/<action_id>_<name>/<user>/<trial>/`（以服务器实际为准，**必须核查**）。

### 1.2 各模态格式（本地样例实测）

| 模态 | 命名示例 | 格式 | 分辨率/通道 | 采样率 | 备注 |
|------|---------|------|------------|--------|------|
| Thermal | `frame_000389.jpg` | jpg | 320×240, 3ch 伪彩色 | ~30fps | 只有帧号无时间戳 |
| Depth_Color | `Depth_<时间戳>_<帧号>_Color.png` | png | 640×480, 3ch | ~10fps | 时间戳到毫秒 |
| IR | `IR_<时间戳>_<帧号>.png` | png | 640×480, 1ch | ~10fps | 与 Depth 同相机同帧号 |
| Skeleton | `Color_<时间戳>_<帧号>.json` | json | 17×[x,y,z] + keypoint_scores | ~10fps | `Color_` 前缀=来自 Depth_Color |
| IMU | `down(LL+RL).csv` / `up(LA+RA+C).csv` | csv | 5 传感器 × 18 数值列 | ~100Hz | 见 §1.3 |
| Radar | `radar_output_T<时间戳>.csv` | csv | 8 列稀疏点云 | — | 45.4% 有效 |

**Depth_Color 与 IR 帧号相同**（同相机 NYX 650，天然帧对齐，可直接按帧号配对做通道拼接）。

### 1.3 IMU CSV 结构（实测）

- 编码：GBK/GB2312（读取需 `encoding='gbk'` 或先探测）。
- 列：`时间, 设备名称, 加速度X/Y/Z(g), 角速度X/Y/Z(°/s), 角度X/Y/Z(°), 磁场X/Y/Z(uT), 四元数0~3, 温度(°C), 版本号, 电量(%)` → 共 21 列，去掉 时间/设备名称/版本号 = **18 个数值列**。
- 传感器设备名前缀（固定顺序，跨 clip 对齐用）：`WTLL`(左腿) `WTRL`(右腿) `WTC`(躯干/腰) `WTLA`(左臂) `WTRA`(右臂)。MAC 地址后缀会变，只用前缀。
- `down(LL+RL).csv` 含 LL+RL 2 传感器；`up(LA+RA+C).csv` 含 LA+RA+C 3 传感器。
- **两文件时间戳交错乱序** → 必须按设备名分组后各自排序。
- 静止时加速度 **Y≈1g、X/Z≈0** → 重力在传感器 Y 轴，重力分离可绕开四元数（低通滤波或直接减 [0,1,0]）。

### 1.4 Radar CSV 结构（实测）

- 列：`timestamp,frame,DetObj#,x,y,z,v,snr,noise` —— 稀疏点云（每帧多个检测点），常有空文件。
- 官方 baseline 用 PointNet；后续专门方法见 OG-PCL(arXiv:2511.08910)、PCFEx(点云 GNN)、milliFlow(ECCV'24)、MiliPoint(NeurIPS'23)。

### 1.5 硬件拓扑与骨架来源（论文 §4.3 / §5.2）

- 1 台 **Vzense NYX 650** 相机同时出 **RGB + Depth + IR**（同一视场，天然对齐）。
- 1 台 **Hikvision TB4117** 独立热像仪出 **Thermal**（独立视场/帧率，与其它视觉模态不对齐）。
- 5 个 WitMotion WT9011 穿戴 IMU（双腕/双踝/腰）；IWR6843 毫米波雷达。
- **Skeleton = MMPose 在 RGB 上提取的 17 关节 3D 姿态**（论文 §5.2.1）→ 坐标基准 NYX 650 视场，与 Thermal 跨相机错位。**骨架坐标可用于 crop Depth/IR（同相机），绝不能用于 crop Thermal。**

### 1.6 论文 benchmark 参考值

- 随机 80/20（论文 Table 3）：Thermal 92.57% / RGB 90.89% / Depth 90.46% / IR 90.22% / Skeleton 79.08% / mmWave 46.63% / IMU 45.52%。
- **cross-subject（LOSO，§6.1.3）：最优配置仅 56.38%，论文明言 SOTA 只有 ~60%** → 随机划分的 90%+ 是身份/房间捷径灌水，真实跨被试水平 50-60%。
- 官方 baseline 0.667；靠前队伍 0.711（LB 公开）。

---

## 2. 70+ 队伍 notebook 关键决策（LB 0.711，已验证）

1. **模态选择**：只取 `Depth_Color(3ch) + IR(1ch)` = **4 通道**拼成单输入 → 两者同相机天然对齐；Thermal 独立相机只能单用。
2. **模型**：**R(2+1)D-34**（视频模型），IG-65M + Kinetics-400 强预训练（3k clips 小样本下不可替代）。
3. **人体检测 + crop**：YOLO11n 只在 IR 灰度图上检测；每 clip 均匀取 8 帧探测 → 保留最高置信度框 → **union 成一个固定窗口**（1.4x 放大、转正方形）→ 全程固定。**逐帧移动 crop 会抹掉"人在画面中移动"的运动线索，是反模式。**
4. **验证**：**subject-fold**（不是随机切 clip），双 fold 集成 0.5/0.5。
5. **采样**：16 帧 endpoint-uniform，128×128，bilinear resize；水平翻转 TTA（logits 相加）。
6. **打包**：int5/int6 权重量化，93.7MB < 100MB。

---

## 3. 血泪教训（旧框架实测失败记录）

- **6 模态 concat 联合训练 = 32.14%，低于 Thermal 单模态 37.85%** → 2891 clips 小数据下联合训练模态竞争 + 过拟合。**结论：单模态先达标，再 late fusion/集成。**
- **person crop 负收益（29.36% < 37.85%）**：根因是"用骨架坐标 crop Thermal"——骨架来自 RGB(NYX 相机)、Thermal 来自 Hikvision 相机，跨相机错位；且用了全序列外接框 + 非等比拉伸。**结论：crop 只能在目标模态自身坐标系内做（如 YOLO 在 IR 上），且必须固定窗口。**
- 时序弱化（16 帧 + mean pooling）是 Thermal 37.85% 天花板的原因之一 → 时序建模必须一等公民。
- 9 个废弃方法（已证无效/过拟合，不再采用）：MMCosine / OGM-GE / PMR / Bottleneck / HARMamba / ModDrop / MDA-KD / Clip-Weight / CORAL（CORAL 由 MixStyle 替代）。

---

## 4. 可复用代码（文本存档，删除 src/ 前保存）

### 4.1 ST-GCN（`src/models/backbones/stgcn.py`，完整可用）

关键点：COCO-17 16 条边、spatial configuration partitioning（3 子集邻接 + 对称归一化）、joint/bone 双流、`data_bn`、10 block 通道 64×4→128×3→256×3、第 5/8 block 时序 stride=2、节点特征 4 维 `[x,y,z,confidence]`。

```python
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def build_adjacency(num_joints, edges, center_joint=0):
    adj = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0; adj[j, i] = 1.0
    hop = np.full(num_joints, -1, dtype=np.int64)
    q = deque([center_joint]); hop[center_joint] = 0
    while q:
        u = q.popleft()
        for v in range(num_joints):
            if adj[u, v] == 1.0 and hop[v] == -1:
                hop[v] = hop[u] + 1; q.append(v)
    A = np.zeros((3, num_joints, num_joints), dtype=np.float32)
    A[0] = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        if hop[i] < hop[j]:
            A[2, i, j] = 1.0; A[1, j, i] = 1.0
        else:
            A[1, i, j] = 1.0; A[2, j, i] = 1.0
    for k in range(3):
        d = A[k].sum(axis=1); d = np.maximum(d, 1e-6)
        d_inv = 1.0 / np.sqrt(d)
        A[k] = A[k] * d_inv[:, None] * d_inv[None, :]
    return torch.tensor(A, dtype=torch.float32)

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, num_subsets=3):
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=(1, 1), bias=True)
    def forward(self, x, A):
        x = self.conv(x)                      # [B, C_out*3, T, V]
        n, kc, t, v = x.size()
        x = x.view(n, self.num_subsets, kc // self.num_subsets, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A.to(x.device)))
        return x.contiguous()

class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels); self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels)
        self.tcn = UnitTCN(out_channels, out_channels, kernel_size=9, stride=stride)
        self.register_buffer("A", A)
        if not residual: self.residual = lambda x: 0.0
        elif in_channels == out_channels and stride == 1: self.residual = lambda x: x
        else: self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x, self.A); x = self.tcn(x); x = x + res
        return self.relu(x)

def joint_to_bone(joint, edges):
    bones = [joint[..., j] - joint[..., i] for (i, j) in edges]
    return torch.stack(bones, dim=-1)   # [B, C, T, E]

class STGCN(nn.Module):
    def __init__(self, in_channels=4, num_joints=17, hidden_dim=64, num_classes=40, stream="joint"):
        super().__init__()
        self.edges = COCO17_EDGES
        if stream == "joint":
            self.A = build_adjacency(num_joints, self.edges)
            self.data_bn = nn.BatchNorm1d(in_channels * num_joints); self.n_nodes = num_joints
        else:
            self.A = build_bone_adjacency(self.edges)
            self.data_bn = nn.BatchNorm1d(in_channels * len(self.edges)); self.n_nodes = len(self.edges)
        h = hidden_dim
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, h, self.A, stride=1, residual=False),
            STGCNBlock(h, h, self.A, stride=1), STGCNBlock(h, h, self.A, stride=1),
            STGCNBlock(h, h, self.A, stride=1),
            STGCNBlock(h, h * 2, self.A, stride=2), STGCNBlock(h * 2, h * 2, self.A, stride=1),
            STGCNBlock(h * 2, h * 2, self.A, stride=1),
            STGCNBlock(h * 2, h * 4, self.A, stride=2), STGCNBlock(h * 4, h * 4, self.A, stride=1),
            STGCNBlock(h * 4, h * 4, self.A, stride=1),
        ])
        self.feat_dim = h * 4
        self.classifier = nn.Linear(self.feat_dim, num_classes)
    def forward(self, x):                      # x: [B, T, 17, 4]
        B, T, V, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
        if self.stream == "bone":
            x = joint_to_bone(x, self.edges)
        x = x.reshape(B, -1, T); x = self.data_bn(x); x = x.reshape(B, C, T, self.n_nodes)
        for block in self.blocks: x = block(x)
        feat = x.mean(dim=(2, 3))
        return self.classifier(feat), feat
```

> `build_bone_adjacency` 见原文件（骨骼图：共享关节即相连，中心骨骼 `(5,11)`）。

### 4.2 TSM（`src/models/backbones/tsm.py`，完整可用）

```python
class TemporalShift(nn.Module):
    def __init__(self, n_segment=16, fold_div=4):
        super().__init__()
        self.n_segment = n_segment; self.fold_div = fold_div
    def forward(self, x):                     # [B*T, C, H, W]
        nt, c, h, w = x.size()
        if nt % self.n_segment != 0: return x
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        if fold == 0: return x.view(nt, c, h, w)
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)
```

### 4.3 按 user 分组划分（`src/data/split.py`，完整可用）

```python
def split_by_user(samples, val_ratio=0.2, seed=42):
    user_to_indices = {}
    for idx, s in enumerate(samples):
        user_to_indices.setdefault(s.get("user", "unknown"), []).append(idx)
    users = sorted(user_to_indices.keys())
    n_val = max(1, int(round(len(users) * val_ratio)))
    n_val = min(n_val, len(users) - 1)
    rng = np.random.default_rng(seed)
    val_users = set(rng.choice(users, size=n_val, replace=False))
    train_idx, val_idx = [], []
    for u in users:
        (val_idx if u in val_users else train_idx).extend(user_to_indices[u])
    assert {samples[i].get("user") for i in train_idx}.isdisjoint(
        {samples[i].get("user") for i in val_idx})
    return train_idx, val_idx
```

### 4.4 Skeleton 加载关键逻辑（`src/data/dataset.py`）

```python
# 每帧每关节 4 维特征 [x, y, z, confidence]，低置信度关节坐标置 0
conf = np.asarray(data[0]["keypoint_scores"], dtype=np.float32)  # [17]
kp   = np.asarray(data[0]["keypoints"], dtype=np.float32)        # [17,3]
kp   = kp[:, [0, 2, 1]]          # [水平, 深度, 垂直] → [水平, 垂直, 深度]
valid = (conf > 0.2).astype(np.float32)
kp   = kp * valid[:, None]
feat = np.concatenate([kp, conf[:, None]], axis=-1)  # [17,4]
# 尺度归一化：除以肩宽（左肩5-右肩6 距离），先归一化再算 bone
scale = np.linalg.norm(kp[5] - kp[6])
kp = kp / (scale + 1e-8)
```

### 4.5 random moving 增强 + SGD/warmup 训练配置

```python
STGCN_CFG = dict(lr=0.1, momentum=0.9, weight_decay=4e-4, warmup_epochs=5,
                 lr_milestones=(35, 55), lr_gamma=0.1, batch_size=64, epochs=65)
# random_moving：首末帧各采样一组旋转/平移/缩放，中间帧线性插值（模拟相机平滑移动）
# 骨架识别不用 AdamW；SGD+momentum+warmup 是小数据收敛关键
```

### 4.6 时序编码器（`src/models/backbones/sensor_encoder.py`）

`SensorEncoder(input_dim, hidden_dim=256)`：Conv1D(k7,64)→BN→ReLU→Conv1D(k5,128)→BN→ReLU→Conv1D(k3,256)→BN→ReLU→BiGRU(128, 双向)→T 轴 maxpool→[B,256]。输入 `[B,T,F]`。适合 IMU/Radar。

---

## 5. 依赖清单（`requirements.txt` 归档）

```
torch>=2.1.0, torchvision>=0.16.0, numpy>=1.24.0, pandas>=2.0.0, pyyaml>=6.0,
opencv-python>=4.8.0, tensorboard>=2.13.0, tqdm>=4.65.0, matplotlib>=3.7.0,
seaborn>=0.12.0, torchinfo>=1.8.0
# 需另装（新方案）：ultralytics(YOLO11), 视频预训练权重(IG65M/Kinetics)
```

---

## 6. 结论

旧框架的核心问题：**把"多模态统一 concat 联合训练"当主线**（已证 32% 低于单模态），**把"person crop"当目的而非手段**（且用错坐标系），**时序建模被 mean pooling 弱化**。这些教训全部用于修正 `doc/PLAN.md`，但新方案不继承旧架构。
