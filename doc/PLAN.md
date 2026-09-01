# CUHK-X 小模型赛道 — 全局规划

> 目标：**LB 80%+**。方法从数据物理含义推导，不照搬任何单一队伍。
> 旧内容归档于 `doc/ARCHIVE_old_design.md`。

---

## 1. 目标与约束

- **任务**：给定测试 clip（6 模态子文件夹），输出 40 类动作之一。
- **约束**：RGB-Free（Thermal/Depth_Color/IR/Skeleton/IMU/Radar）；跨被试（测试 405 clip 全是没见过的 12 人）；模型 ≤100MB。
- **目标校准**：官方 baseline 0.667；LB 0.711 只是第 143/201 名 → **顶部队列在 80%+，目标定 80%+**。

---

## 2. 数据本质（实测定死）

### 2.1 规模与结构
- 训练 2891 clip / 40 类 / **18 被试**（user1–9 + user16–24）；测试 405 clip 全 6 模态。
- 结构 `<Modality>/<Action>/<Subject>/<sample>/<files>`；**Skeleton 在 `predictions/` 子目录**。
- **长尾比 29.5**（Walk=324 vs Watch_TV=11）；缺失率 Thermal 0% / Depth·IR 3.6% / IMU 4.5% / Radar 4.2%。

### 2.2 硬件拓扑 → 对齐关系
| 关系 | 模态 |
|------|------|
| **同相机天然对齐**（帧号一致） | Depth_Color + IR + Skeleton（NYX 650，~10fps） |
| 独立相机，只能单用 | Thermal（Hikvision，~25fps，320×240） |
| 独立时间轴 | IMU（~100Hz 5 传感器）、Radar（20fps 点云，47% 有效） |

### 2.3 物理含义（实测）
| 模态 | 物理含义 |
|------|---------|
| Depth_Color | **伪彩色深度图**（jet 色带，非自然 RGB）→ 语义是 3D 深度几何 |
| IR | 主动红外灰度（人体剪影）→ **最适合 YOLO 检测人** |
| Thermal | ironbow 伪彩色热力图 |
| Skeleton | **3D 米制坐标（非像素）+ Human3.6M-17 拓扑** → 只能独立建模，不能用于图像定位 |
| IMU | 肢体运动学（5 传感器 ×18 列） |
| Radar | 稀疏点云 + 多普勒速度 |

---

## 3. 方法论总纲

**P1 运动优先**：动作 = 身体随时间的运动模式，时序建模一等公民。
**P2 检测定位、裁剪保运动**：目标检测只负责"人在哪"；裁剪用**单一固定窗口**（多帧并集），保留位移；**绝不做逐帧裁剪**（抹掉位移）。
**P3 模态按对齐分组**：同相机早融合；异源独立建模 + 晚融合（集成）。

### 3.1 两层增强框架
- **数据级增强**（输入层互相利用）只在同相机间成立：Depth+IR 4ch 早融合、帧差运动通道。
- **训练级增强**（蒸馏/监督）跨模态成立：跨模态蒸馏、logit 集成。

### 3.2 各模态抓运动方式
| 模态 | 抓运动方式 | 定位 |
|------|-----------|------|
| Depth+IR | 4ch 早融合 + **帧差通道**（显式运动） | YOLO 在 IR 上 → 固定窗口 |
| Thermal | 独立视频模型（结构迁移） | 无 crop |
| Skeleton | ST-GCN joint+bone+**velocity**（H3.6M 边） | 无需定位 |
| IMU/Radar | 低优先 | — |

### 3.3 训练原则
单模态先达标 → logit 平均集成。不搞 6 模态联合训练（旧框架 32% < 单模态 37.85%）。

---

## 4. 各模态处理与模型

- **视觉主线**：Depth_Color(3ch)+IR(1ch)=4ch，按帧号配对；16 帧 uniform 采样；128×128；YOLO 在 IR 检测→固定窗口；**可选帧差通道（4→8ch）**。模型 R2+1D-18（Kinetics）。
- **Thermal**：3ch 独立视频模型（R2+1D-18），无 crop。
- **Skeleton**：H3.6M-17 边 ST-GCN；置信度掩码（**若恒 1.0 则去掉**）+ 中心化 + 肩宽归一化 + joint/bone 双流 + **velocity 通道（[T,17,7]）**；SGD+momentum+warmup。
- **IMU**：重力分离 + 按设备分组 + 1D-CNN（低优先）。
- **Radar**：PointNet（最低优先，可弃）。

---

## 5. 实测结果（持续更新）

| 实验 | 配置 | val | 结论 |
|------|------|-----|------|
| E0 | TSM-R18, 无裁剪, 3折×30ep | 0.3615 | 基线 |
| E7 | R2+1D-18, 无裁剪 | 0.4618 | ✅ 主干升 R2+1D，+10% |
| E1 | R2+1D-18 + crop, 3折 | 0.5033 | 裁剪 +4% |
| **E1'** | R2+1D-18 + **正确 crop**（修 bug）, fold0 | **0.5167** | bug 只损失 1.3% |
| **E4** | R2+1D-18 + crop + **帧差**, fold0 | **0.5339** | ✅ **帧差 +1.7%，抓运动有效** |
| E2 | IR Otsu 掩码 | 0.2734 | ❌ 作废 |
| E3 | 32 帧 | 0.3305 | ❌ 作废（clip 仅 ~29 帧） |
| R34 | R2+1D-34(IG-65M) + crop, 60ep, fold0 | 0.4252 | ❌ **根因=lr 1e-3 冲垮 IG-65M**（非容量问题），lr=1e-4 重试中 |
| S1 | 骨架 ST-GCN joint+bone, 3折 | 0.3479 | 基线 |
| **S2** | 骨架 **MotionBERT**(MB_lite) 3折 | **0.4177** | ✅ 预训练 +7%（修复跨 fold 泄漏后，std 0.01） |
| T1 | Thermal, fold0 | 0.4771 | ✅ 强第二模态 |

**关键结论**：
1. 主线 = **R2+1D-18 + 固定 crop + 帧差**（0.5339 fold0，目前最强）。
2. R2+1D-34 失败根因是 lr=1e-3（不是容量），lr=1e-4 重试中。
3. 骨架 MotionBERT=0.4177（ST-GCN=0.3479），预训练 +7%，三模态里仍最弱。

---

## 6. 下一步计划（按优先级，2026-08-18 更新）

1. **第三轮视觉定版**（round3，锁 A100）：主线 R2+1D-34 + 帧差 + lr=1e-4，3折 60ep（fold0=0.5942）；Thermal R2+1D-34 + IG-65M + crop，3折 60ep（fold0=0.5646）。
2. **骨架定版**：MB_lite（0.4177）；NTU 权重（0.4211）只高 0.34 点且体积 2 倍（188MB），**不采用**。
3. **集成**：主线(0.59) + Thermal(0.56) + 骨架(0.42) logit 加权（`--auto_weight --prob_avg`）+ 翻转 TTA + 多折平均 → 预期 **65%+**。
4. **量化打包**：R2+1D-34 升级后 3 模型 int8 ≈ 146MB **超 100MB**，需 **int5/int6 bit-packing**（主线 63M、Thermal 63M、骨架 20M）或减模型。
5. **测试 bbox**：`submit_detect.sbatch` 生成 bbox_test.json / bbox_thermal_test.json（集成推理前）。
6. **帧差结论**：只对主线（Depth+IR）有效（+1.7%）；对 Thermal 和骨架**有害**（语义错配/噪声），不再用于这两者。

### 6.1 后续优化方向（多模态对比/蒸馏，暂缓）

调研 9 个 VLM（CLIP/ViLT/ALBEF/VLMO/BLIP/CoCa/BEiT-3/Uni-Perceiver/PaLI）结论：
- 这些是**视觉-语言**模型，依赖文本监督；本项目 HAR **无文本**，不能直接套用。
- 统一架构（VLMO/BEiT-3/Uni-Perceiver）不适用（6 模态联合训练已被证明 < 单模态，且超 100MB）。

**可借鉴的 2 条**（⭐ 论文 §6.1.3 的 cross-subject 实验已证明对比学习有效）：
1. **跨模态对比学习（CLIP 改造）**：同 clip 的 Depth+IR 特征 ↔ Thermal 特征作正对、跨 clip 作负对，InfoNCE 对齐 → 模态不变表征，提升 cross-subject 泛化。作为主线训练的辅助损失，预期 +2-5%。
2. **跨模态蒸馏（ALBEF/BLIP 动量蒸馏）**：强模态（主线 0.59）软标签蒸馏弱模态（骨架 0.42），动量模型稳定蒸馏目标。

### 6.2 集成阶段优化（三个思路评估，2026-08-18）

1. **GAF 时序转图像**：只适用于一维时序（IMU 加速度/角速度）；IMU cross-subject<45% 价值低已暂缓，骨架已有 MotionBERT 更强 → **暂缓**（若补 IMU 才用 GAF+CNN）。
2. **个性+共性网络**：个性网络=各模态独立模型（已完成）；共性网络**简化为 Stacking 元学习器**——用 val 的 3 模态 logits(120维) 训练逻辑回归（参数极少防过拟合），比手工加权更优，预期 +1-2%。
3. **时间一致性投票**：= 时间 TTA——clip 用不同起始偏移采样多段 16 帧分别预测后投票，减少采样抖动误判，**不增模型大小**（推理多次前向），预期 +1-2%。

**结论**：②③ 属集成阶段优化（排在第三轮定版后），① 暂缓。

### 6.3 训练正则化技巧（暂缓，2026-08-18）

现状：主线 R2+1D head `Dropout(0.3)`、`weight_decay=1e-4`（Adam 非 AdamW）、无标签平滑；骨架 head `Dropout(0.5)`、`weight_decay=0.01`（官方）。

| 技巧 | 建议 | 收益 | 风险 |
|------|------|------|------|
| 标签平滑 0.1 | 值得加（两模态），soft label 校准置信度 + 对 logit 集成有益 | +0.5~1% | 零 |
| Dropout 0.3→0.5 | R2+1D-34(63M) 可试，只调 head 不动 backbone（预训练） | 不确定 | 低 |
| 权重衰减 1e-4→5e-4 | 可试，但 Adam（非 AdamW）会侵蚀预训练权重，谨慎 | 有限 | 中 |

**结论**：三个都是"锦上添花"，排在集成/TTA/量化之后。标签平滑最值得（一行、零风险）；Dropout/权重衰减需实验对比。

---

## 7. 预训练权重可用性（已搜 HF/GitHub/arxiv）

| 模态 | 可用的预训练 |
|------|-------------|
| Depth+IR 视觉 | IG-65M（GitHub，已下载）+ Kinetics（torchvision） |
| Skeleton | **MotionBERT**（HF `walterzhu/MotionBERT`，H3.6M-17 拓扑匹配）——**已落地**：`src/motionbert/` + `train_skeleton_motionbert.py`，MB_lite 82MB fp32 符合约束 |
| Thermal | ❌ 无热红外专用预训练；但 **IG-65M 3ch 完美匹配（stem 不改，100% 迁移）** + YOLO 检测裁剪（已加 `--modality thermal`） |
| IMU | ❌ 无现成权重（MuJo 是方法非权重；COMODO 是蒸馏方法） |

---

## 8. 风险与开放问题

1. 100MB 确切含义（总模型大小 vs 单模型）→ 提交前确认（见 §9）。
2. Depth_Color 色带能否反演（DMHI 需要）→ 待查 Vzense 文档。
3. R2+1D-34 是否值得救（lr=1e-4 重试）→ 用一次小实验判断。
4. IMU/Radar 增量价值（基线 45%/46%）→ 集成时用数据说话。

---

## 9. 模型大小账（100MB 约束）

| 模型 | 参数 | fp32 | int8 |
|------|------|------|------|
| 主线 R2+1D-18（Depth+IR 4ch） | 33M | 132MB ❌ | 33MB |
| Thermal R2+1D-18 | 33M | 132MB ❌ | 33MB |
| 骨架 MotionBERT-Lite | 20.5M | 82MB ✅ | 20MB |
| **三模型集成 fp32** | — | **346MB ❌** | **86MB ✅** |

**结论**：单个 R2+1D-18 fp32（132MB）就已超 100MB，集成更是必超。
**唯一路线 = 量化**：int8 三模型 86MB 合规；int6 约 65MB、int5 约 54MB（70+ 队用 int5/6 压到 93.7MB）。
⚠️ 需确认：100MB 指提交总模型大小（大概率）；量化是否允许（int8 基本无损，int5/6 有轻微损失）。

**已实现**（2026-08-17）：`src/quantize.py`（per-channel int8 量化/反量化）+ `scripts/quantize_pack.py`（打包+检查<100MB+多折权重平均）+ `scripts/ensemble_inference.py`（三模态加权+量化加载+翻转 TTA）。流程：训练(fp32) → quantize_pack 打包(int8) → ensemble_inference 推理。注意：多折用 `--average` 合并成一份，否则 3折×3模态=9份必超。

> ⚠️ **2026-08-19 更正**：上面"多折用 `--average` 合并"已被证明是**错误做法**（SWA 权重平均导致预测塌缩），见 §10.1。当前正确做法 = **每折独立打包 + 推理时 logits 平均**。

---

## 10. 关键修复与后续方向（2026-08-19 更新）

### 10.1 🔴 关键修复：SWA 权重平均导致预测塌缩（LB 0.39→0.62）

**根因**：多折模型用 `--average` 做权重平均（SWA）→ 决策边界模糊、**测试预测塌缩到少数类**（仅 26 类，模型在测试上无法区分）。**修复**：改为**每折独立量化打包 + 推理时 logits 平均**（`quantize_pack.py` 不 `--average` 时输出 `fold{i}` 独立包；`pack_main_perfold.sbatch` 用 main 2 折 int5=80MB + logits 平均 + flip TTA），测试预测恢复 38-39 类，**LB 0.39→0.62**。

**铁律**：多折模型**只用 logits 平均（每折独立推理），绝不用权重平均（SWA）**——权重平均使决策边界模糊、预测塌缩。

**对照 0.711 公开方案**（`notebooks/lb-0-711-yolo-person-crop-r2plus1d-100mb.ipynb`，公开 LB 0.71144）：
- **只用 Depth+IR 4ch（无 frame_diff）**、R2Plus1D34 + IG65M + head Dropout(0.3)
- **2 折独立推理 + logits 平均**（fold0 int5 + fold1 int6，权重 0.5/0.5）
- **parent_val_acc = [0.7156, 0.7187]**（我们 8ch frame_diff 3fold 仅 0.59 → 差距在训练 recipe，非架构）
- YOLO 固定 crop + flip TTA + int5/int6 打包 93.7MB
- 训练细节（lr/epochs）未公开；推理路径完整公开

**重要实测**：thermal/skeleton 加入集成**反而拖累 main**（无泄漏 val 集成 0.566 < main 单模态 0.59）。与 0.711 只用 Depth+IR 一致——**模态不在多，在精**。

**已达成（2026-08-20）**：全量训练 + 数据增强 + label smoothing → **LB 0.71144**（seed42+777 集成），追平 0.711 公开方案。详见 §10.4 当前最佳配置。

### 10.2 剩余模态探索（IMU / Radar，尚未启用）

| 模态 | 物理语义 | 状态 | 建议 |
|------|---------|------|------|
| IMU | 身体内部惯性（加速度/角速度），不受遮挡/光照影响 | ✅ 格式已探索（未训练） | **1D CNN + 统计特征**；5 传感器（手腕×2+脚踝×2+腰）×16 列；重力分离+按设备分组；先做单模态 val（成本低） |
| Radar/mmWave | 外部反射（Range-Doppler），测位置/运动 | ✅ 格式已探索（未训练） | 稀疏点云（x/y/z/v/snr/noise，20fps，47% 有效）；PointNet/统计特征；同样先做单模态 val |
| Thermal | 温度热像 | ✅ 0.563 | 已测；集成中拖累 main，暂不优先 |
| Skeleton | 关节运动（MotionBERT） | ✅ 0.42-0.51 | 已测；SWA 后 0.51；抗跨被试，可作辅助 |

**原则（数据说话）**：任何模态**先在 val 上用单模态 cross-subject 实测**，达标且与 main 错误分布**互补**才考虑集成；否则"加了没好处甚至拖累"（thermal/skeleton 已证明）。

### 10.3 动态集成（PMR 原型重平衡 / 置信加权 / Stacking）

**理念**：当某模态对某动作类别特征模糊时，**动态降低其权重**，让更可靠的模态主导决策（比固定权重 softmax(val_acc) 更智能）。skeleton / IMU / mmWave 可构成互补的"运动视角"（内部惯性 + 外部反射 + 关节）。

**前提（重要）**：动态加权只在模态**互补**（A 在某样本模糊时 B 恰好清晰）时才有效。我们实测 thermal/skeleton **整体拖累 main**，说明当前这些模态质量不足或高度相关——**先把各模态 val 提上来，再谈动态集成**。

**实现选项（按成本排序）**：
1. **Stacking 元学习器**（见 §6.2.2）：用 val 的 3 模态 logits（120 维）训练逻辑回归 → 40 类，参数极少防过拟合，比手工加权优，预期 +1-2%。
2. **样本级置信加权**（简化 PMR）：对每个测试样本，按各模态 max-softmax 置信度动态加权（无需训练，但需 val 验证有效性）。
3. **完整 PMR**（原型重平衡）：每类维护模态原型，融合时按样本到原型的距离动态调节模态权重——最复杂，**需先验证模态价值**再投入。

**优先级**：Depth+IR 已推到 0.71144（§10.4）。后续方向：增强组合搜索 → 若更强增强有效则冲 0.72+；否则转 IMU 单模态（§10.2）或动态集成（§10.3）。

### 10.4 当前最佳配置（2026-08-20，LB 0.71144 = 追平公开 0.711）

**完整 recipe**：
- **模态**：Depth+IR 4ch（无 frame_diff）+ 固定 crop（YOLO 在 IR，union 窗口 1.4x）
- **模型**：R2Plus1D34 + IG65M + head Dropout(0.3)
- **训练**：全量 18 被试 2931 clip（`--full` 不分折）+ 80ep + lr 1e-4 CosineAnnealing + weight_decay 1e-4
- **数据增强（aug_strength=2）**：随机翻转 0.5 + 亮度/对比度 0.8-1.2 + 缩放 0.85-1.15 + 平移 ±8%
- **label smoothing 0.1**
- **推理**：seed42 + seed777 每折独立 int5 打包 + logits 平均 + flip TTA（80MB）
- **LB = 0.71144**

**关键教训**：
1. 全量训练（18 被试）比 2fold 数据 +45%：单 seed42 0.706 → seed42+777 集成 0.711
2. 全量无 val，seed 质量只能靠 LB 试错：seed42+777 互补（0.711），seed2024 拖累（0.686）→ **只集成不拖累的 seed**
3. 增强 + label smoothing 治过拟合（无增强 fold0 0.5666 → 增强 0.6192）
4. 帧差在强时序模型 + 固定 crop + 增强下冗余有害（8ch 0.596 < 4ch 0.619）
5. 统一测试脚本 `scripts/test_submission.sh`（登录节点 CPU：`CUDA_VISIBLE_DEVICES="" bash scripts/test_submission.sh "42 777"`）

**下一步（2026-08-20）**：
1. **增强组合搜索**（`aug_search.sbatch`，fold0+60ep 对比 aug_strength 0/1/2/3）——进行中
2. 若更强增强（3）有效 → 更强增强全量重训冲 **0.72+**
3. 若 2 已最优 → 锁定 0.71144，转其他方向（IMU §10.2 / 时间 TTA / 动态集成 §10.3）

---

## 11. 数据增强与弱模态利用方法论（2026-08-19 记录，未舍弃）

### 11.1 计算机图形学（CG）增强——按模态分性价比

| 模态 | CG 增强方式 | 可行性 | 结论 |
|------|------------|--------|------|
| **Skeleton（3D 关节）** | 关节角度扰动、全局旋转/平移/缩放、左右镜像、时序拉伸、**运动重定向**（动作映射到不同体型） | ⭐ 最高（3D 几何变换天然匹配，几乎零成本） | **最划算的 CG 路径**；骨架抗跨被试，正缺泛化 |
| Depth | SMPL 渲染虚拟深度、视角重投影 | 中（需 SMPL+姿态估计+渲染管线） | 高成本低确定收益，暂缓**但不舍弃** |
| IR / Thermal | 光照/纹理变化 | 低（热像/红外语义特殊，渲染不真实） | 最后考虑 |

**认知**：0.711 纯视觉（无 CG 渲染）就到 0.715 → 视觉 CG 渲染不是瓶颈；CG 用在骨架上是"性价比之王"。

### 11.2 模态间互相增强——现状完整清单

| 机制 | 类型 | 状态 |
|------|------|------|
| Depth+IR 4ch 早融合（同相机对齐） | 数据级（输入层） | ✅ 已用（主力） |
| 帧差运动通道（显式运动） | 数据级 | ✅ 已用（8ch 版；4ch 重训去掉） |
| logit 集成（晚融合） | 训练级（跨模态） | ✅ 已用（但 thermal/skeleton 拖累 main） |
| **跨模态蒸馏**（main→skeleton 软标签） | 训练级（跨模态） | 📋 计划中，未实现（§6.1） |
| **跨模态对比学习**（Depth+IR↔Thermal InfoNCE） | 训练级（跨模态） | 📋 计划中，未实现（§6.1，论文 §6.1.3 证明 cross-subject 有效） |
| **跨模态一致性正则**（main vs 弱模态预测 KL） | 训练级（跨模态） | 💡 新思路（§11.4），未实现 |

### 11.3 蒸馏——可做，但有明确前提

- **最有价值**：main(teacher 0.59) → skeleton(student 0.42) 软标签蒸馏。骨架抗跨被试，若提升到 0.5+ 且与 main 错误互补 → 集成才有救。这是"救骨架"的合理路径。
- **可类比**：main → thermal 蒸馏（thermal 0.56，独立视角）。
- **低价值**：main→main 自蒸馏、fold 间自蒸馏（为做而做，无实际意义）。
- **前提（铁律）**：弱模态提升后**必须与 main 错误互补**才有集成价值，否则仍会拖累（thermal/skeleton 已实测拖累）。

### 11.4 弱模态的正确用法（关键认知，2026-08-19 更新 08-20）

**澄清**：弱模态（thermal/skeleton/IMU/Radar）是**同一 clip 的不同物理视角**，不是"更多数据"。它们对 main 的正确用法 = **作为辅助约束/自监督信号，让 main 学到跨模态不变的本质表征 → 提升 main 泛化**（而不是"数据增强"喂数据，也不是"再训一个独立分类器"）。

**⚠️ 重要更正（2026-08-20）**：数据集每个 clip 的多模态**天然配对**（同一 subject/sample 的 6 模态 = 同一人同一时刻同一动作）→ **正对/负对天然存在，跨模态监督（对比/蒸馏/一致性）不需要等单模态达标，可以直接做**。两条路径并行：①单模态 val 实测（基线，用于集成决策）；②跨模态监督（直接用配对数据提升 main）。

| 方法 | 机制 | 对 main 的作用 | 成本 | 数据前提 |
|------|------|---------------|------|---------|
| **跨模态对比学习**（CLIP 式 InfoNCE） | 同 clip 的 main 特征 ↔ thermal/IMU/skeleton 特征正对、跨 clip 负对 | main 学到模态不变表征 → 提升 cross-subject 泛化（论文 §6.1.3 已证） | 中（改训练+辅助损失，重训） | ✅ 配对天然满足 |
| **跨模态一致性正则** | main 预测 vs thermal/skeleton 预测一致性损失（KL/蒸馏） | main 决策对模态扰动鲁棒 → 减过拟合具体视觉线索 | 中低 | ✅ 配对天然满足 |
| **跨模态蒸馏** | main(teacher)→弱模态(student) 软标签 | 提升弱模态本身（前提见 §11.3） | 低 | ✅ 配对天然满足 |

**当前最优方向（2026-08-20）**：main 已 0.71144，用**跨模态对比学习**（thermal 独立视角 + IMU 未测但互补）辅助 main，预期 +2-5%。

### 11.5 决策流程（2026-08-20 战略更新：对比学习优先于集成）

```
1. 【已达成】main（Depth+IR）= 0.71144 单模态主力
2. 【主攻】跨模态对比学习（用 thermal/IMU/skeleton 配对数据监督 main）
   → 收益直接加在 main 上（0.711 → 0.72+），不依赖弱模态集成质量
3. 【降级】集成弱模态（thermal/skeleton 已证明拖累，IMU/Radar 大概率也拖累）
   → 仅在对比学习后、弱模态 val 显著提升且互补时才考虑
4. 【次要】单模态 val 实测（仅作基线参考，不再是"集成决策"前提）
```

**战略纠正（2026-08-20）**：集成弱模态的实际收益 < 跨模态对比学习。理由：弱模态单模态 val 仅 0.42-0.56，集成时大概率拖累强模态 0.71；而对比学习用弱模态的**配对数据监督 main**，收益直接加到 main，且论文证明有效。**当前主攻 = 跨模态对比学习（main↔thermal/IMU），集成决策降级为次要。**

### 11.6 全量数据最终训练（full-data 冲刺，2026-08-19 记录）

**关键洞察**：0.711 用 2 fold，**每折只训练 9 个被试**（另 9 个留 val）→ 最终提交的每个模型只见过一半被试。若确定方法论后用**全部 18 被试（2931 clip）训练**最终模型，数据量 +45%（~2000→2931）→ **大概率超越 0.711**（数据翻倍）。

**方案**：
1. **--full 模式**（需给 `train_step1.py` 加 flag）：全部 clip 训练、不分折、不评估 val（或仅打印 train loss 监控）
2. **选 checkpoint**：CosineAnnealing 退火后 lr 最小 → **取最后 epoch**（无 val 早停，靠退火收敛）；可选存中间快照备用
3. **多样性集成**：3 个不同 seed（42/2024/777）各训一个 full-data 模型，推理时 logits 平均
4. **大小**：3 模型 int5=120MB 超限 → **2 模型 int5=80MB** ✅ 或 **3 模型 int4=96MB** ✅
5. **过拟合控制**：全量训练无 val 早停，更需正则 → **label smoothing 0.1 + head Dropout 0.5**（见 §6.3）

**触发条件**：4ch 重训（§10.1）确认方法论（模型/增强/超参）后 → 转 full-data 全量冲刺（此步是最终提交前的最后一环）。

### 10.5 多方向实验结论（2026-08-22，全部验证完毕）

**① 增强搜索（aug_search，fold0+60ep）**
| aug_strength | fold0 | 结论 |
|------|-------|------|
| 0 | 0.6179 | 过拟合明显 |
| 1 | **0.6652** | ⭐ fold 最优 |
| 2 | 0.6609 | 当前定版 |
| 3 | 0.6609 | 与 2 持平 |

**② 🔴 fold 不能映射全量（关键教训）**：全量 s1 seed42 = **0.652**（失败，s1 全量不适用），全量 s2 seed42 = 0.706（保底已验证）→ **aug_strength=2 定版不可动摇**，s1 路线彻底关闭。

**③ worker_init_fn = 多余（验证定论）**：self.rng + WeightedRandomSampler 已让样本增强逐 epoch 变化（sampler 每 epoch 重排样本→worker 位置）。worker_init_fn 行为等价、无额外收益 → 已移除。

**④ 骨架（MotionBERT）**：无增强 0.4177 → 温和 0.4380 → 强化 0.4513 → 强化+正则(ls0.1+早停30ep) **0.4314**（正则失败，fold1 早停过早 0.3822，std 0.038 波动大）。骨架 ~0.45 弱水平，只作对比/动态集成素材，不独立集成。

**⑤ 对比学习（main↔thermal）**：s1+对比 fold0=0.6771（+1.2%）；**s2+对比 fold0=0.6437（-1.7%）**，3 折 mean 0.6475。对比学习价值依赖增强强度：s1 有效、s2 负贡献；但 s1 全量已死 → **main↔thermal 对比线放弃**。

**⑥ 互补分析（oracle 公式修复后）**：skeleton rescue 0.18/oracle 0.68；thermal rescue 0.25/oracle 0.71 → 简单加权拖累，但 **oracle 高 = 样本级动态选择有空间**。

**⑦ fold0 续跑**：0.6663 < 0.6771 → 对比学习 60ep 已收敛，续跑无益。

**⑧ 当前唯一有效路径** = 保底 0.71144（s2 seed42+777）不动；剩余未试：时间 TTA（推理侧零风险）、IMU/Radar 单模态、main↔skeleton 对比（异源）、动态集成（样本级置信加权/Stacking）。

**⑨ IMU 单模态摸底（1D CNN 3折 30ep，2026-08-22）**：mean val **0.3547**（fold0 0.3326 / fold1 0.3387 / fold2 0.3929，std 0.027）。**没完全收敛**（loss 3.45→1.48 仍在降、fold2 val 缓升），但论文 IMU 基线仅 38–45% → 真实天花板 ~0.40–0.45，**够不到 >0.5 判据**。结论：**IMU 线关闭，不集成**（即使 60ep 重跑确认天花板 ~0.4 也不改变结论）。

**⑩ 骨架进输入定论（2026-08-22）**：
- `bbox_tight_train.json`（紧框，`detect.py --margin 1.1 --no_square`）**只用于骨架 3D→2D 投影锚定**（验证+热图渲染），主 4ch 裁剪窗口 `bbox_train.json`（1.4x 方框）完全不动。
- 实验矩阵（frame_diff 8ch 有害、对比学习 s2 有害、thermal/skeleton 集成拖累、IMU 0.35）→ **像素级热图 5ch 预期收益低**（main 已从图像学到姿态，热图=冗余+投影噪声）。
- 决策门：①紧框验证投影能否贴人体（CPU 2min）→ ②能则仅 fold0 快速对照（5ch vs 4ch 基线 0.6609），fold0 ≥0.665 才考虑全量；③不能/对照不过 → 关闭骨架进输入线，转**时间 TTA**（推理侧零风险，唯一剩余高 EV 路径）。

**⑪ 弱模态分档总表（2026-08-22）**——"关闭"= 停止投训练算力（拖累 main），**数据全保留**，保底 0.71144 随时可提交：
| 档位 | 模态 | 状态 | 依据 |
|------|------|------|------|
| 🔴 已关闭 | IMU（0.35） | 天花板 <0.5 | 论文基线 38–45%，训满 ~0.4 |
| 🔴 已关闭 | thermal/skeleton **集成** | 拖累 main | 无泄漏 val 0.566 < main |
| 🔴 已关闭 | main↔thermal **对比学习** | 负贡献 | s2 fold0 -1.7% |
| 🟡 待一测 | **骨架热图 5ch** | 紧框验证 + fold0 对照（3h 定生死） | 唯一未跑完的进输入路径 |
| ⚪ 未测 | Radar | 可低成本摸底 | 外部视角但基线 46%，预期低 |
- **推理侧最后一手**（零风险，不碰训练）：置信度门控救援（main 低置信查 aux）——但 oracle（skel 0.68/th 0.71）≈ main 自身，预期 +0-1%。
- **剩余 EV 排序**：时间 TTA（零风险）> fold0 5ch 对照（若验证过）> Radar 摸底 / 门控救援（预期 +0-1%）。

### 10.6 多模态策略修正（2026-08-22 读论文后，推翻"弱模态全关"的过早结论）

**论文 HAR 基线（Table 3，注意是 80/20 随机分，非 cross-subject）**：Thermal **92.57**（最强）/ RGB 90.89 / Depth 90.46 / IR 90.22 / **Skeleton 79.08** / mmWave 46.63 / IMU 45.52。→ 骨架是**强模态**（我们 0.45 只是 cross-subject 下掉 34pt，与 RGB 90.89→56.38 掉幅一致 → **0.45≈骨架 cross-subject 天花板**，不是没做好）。

**论文 cross-subject（LOSO）结论（§6.1.3）**：
- RGB 随机 90.89 → LOSO 大幅掉；**加对比学习（Contra.）进一步强化 subject-invariant 表征**；w/o LT + Contra + w/o CD = **56.38%**。
- **类平衡重采样**对 RGB/IMU/Skeleton 都有提升（论文 Fig 6a，RGB 90.89→96.16）。
- 我们的 0.71144 cross-subject **已超过论文 cross-subject RGB 56.38%**。

**修正后的多模态策略（按 EV）**：
1. **main↔skeleton 对比学习**（**未试**，论文明确背书 cross-subject 有效；异源 3D 结构信息量 > thermal；之前 main↔thermal 在 s2 失败是配方/辅助模态问题，不是方法问题）→ fold0 对照。
2. **骨架重训**：类平衡重采样（论文验证有效）+ **去掉伤它的正则**（ls0.1/早停已证反效果）+ velocity 特征 → 目标 0.45→0.5+（cross-subject 天花板附近小幅提升即可让动态集成受益）。
3. **Thermal 提升**（随机分 92% 是所有模态最强，cross-subject 0.56 可能还有涨）→ 加 balanced/更强增强重训。
4. **集成侧**（用提升后的 aux logits）：类感知加权（每类看 val 上谁准）> 置信门控救援（main 低置信才查 aux）> Stacking 元学习器 —— 都是"只救不伤"，避免朴素平均的拖累。
5. 时间 TTA 并行（零风险）。

**铁律修正**：之前"thermal/skeleton 集成拖累 main"只对**朴素 logit 平均**成立；**类感知/置信门控/对比学习**这三个"利用弱模态"的正路**全部未试**。弱模态不是没用，是不能乱用。

**⑫ 统一框架 + 骨架/热图定位修正（2026-08-22）**：
- **骨架 79%（论文）= 原生 3D 形态的成绩**（MotionBERT 直接建模 3D 序列，不投影）。"5ch 热图投影"把 3D 降维成 2D → 丢深度 + 加误差 + 与 Depth+IR 图像冗余 → **EV 最低**。骨架价值在**原生 3D 独立模型**（已有 MotionBERT 0.45）→ 走对比学习 + 输出层动态集成，**不塞进像素输入**。
- **Thermal 是论文最强模态（92.57% > Depth 90.46%）**；我们 0.56 vs main 0.71 的差距在 **recipe**（无 crop、没吃到 aug/ls/2-seed/80ep 全套）→ **配齐 main 的 recipe（thermal YOLO crop + balanced + aug + ls + 多 seed）目标 0.65+**，是最高性价比未利用资产（唯一独立相机视角，互补性最强）。
- **统一三层次**：输入层（热图，丢 3D ❌）/ 特征层（双流，保留 3D，成本中）/ 输出层（logits 动态/类感知融合，零风险 ✅）。**推荐输出层 + 对比学习**，不碰 0.71144。
- **修正后优先级**：①Thermal 全套 recipe 升级（最高 EV）→ ②main↔skeleton 对比（原生 3D，论文背书）→ ③输出层动态/类感知集成 → ④5ch 热图降级后备（坐标系已验证，EV 低）。
- **已实现（2026-08-22）**：统一提交脚本 `scripts/multimodal_pipeline.sbatch`（[1]生成 bbox_thermal_train.json → [2]Thermal 全套 recipe fold0 60ep → [3]main↔skeleton 对比 fold0 60ep）。
  - **发现 thermal 归一化 bug**：`ThermalVideoDataset` 之前用 ImageNet 均值(0.485...) 但模型是 R2+1D-34/IG-65M（Kinetics 域）→ 归一化不匹配是 thermal 落后的实因之一，已改 Kinetics(0.43216...) + 补 aug_strength 缩放/平移。
  - `train_contrastive.py` 新增 `--aux skeleton`：SkeletonEncoder(1D-CNN, [B,T,17,6]→512) + CE(main)+λInfoNCE(z_main,z_skel)，只保存 main（直接提升主线，骨架原生 3D 不投影）。

**⑬ 多模态管线结果（2026-08-22, job 41643）**：
- **P1 Thermal 全套 recipe fold0 = 0.6190**（旧 0.5646，**+5.4pt**）✅：功臣 = **Kinetics 归一化 bug 修复**（ImageNet→Kinetics 匹配 IG-65M）+ crop+balanced+aug2+ls0.1。离 0.65 目标还差一点，但独立视角真增量 → **值得 3 折 + 全量 2-seed 冲提交级**。
- **P2 main↔skeleton 对比 fold0 = 0.6609 = 基线（中性）**⚠️：nce 2.0→0.25（对齐在学）但 main val 零转移 → 与 thermal s2 -1.7% 同族（弱 aux 监督强 main 无效）。**骨架监督线关闭**。
- **后续最高 EV**：Thermal 3 折确认 → 全量 2-seed（~0.65）→ **main(0.71144)+thermal(0.65) 类感知/置信门控动态集成**（独立视角，天然互补）。骨架/IMU 监督线全部关闭。

**⑭ 骨架两个真 bug 修复（2026-08-22，用户洞察验证）**：
- **Bug1 输入表示错配**：`MotionBertSkeletonDataset` 喂 `[x,y,conf]`（2D + conf 恒 1.0），**丢原生深度**；但预训练权重是 `model_pos`（H36M/AMASS **3D** 预训练）→ 3D 预训练知识全浪费、输入退化（第 3 通道常数）。论文也用"17 **3D** joints"。修复：默认 `input3d=True`（`[水平,垂直,深度]=[0,2,1]`，中心化+3D 肩宽归一化）。
- **Bug2 增强非每 epoch 随机**：`train_skeleton_motionbert.py` 不传 `--balanced` 时 `sampler=None+shuffle=False` → 同一样本增强**逐 epoch 确定性**（过拟合固定扰动）。修复：无 sampler 时 `shuffle=True`。
- 对照实验 `scripts/skeleton_3d.sbatch`：同设置（强化增强+balanced+无LS+无早停，3折30ep）跑 2d vs 3d → 验证表示增益 + shuffle 修复增益（预期 > 旧 0.4513）。

**⑮ "3D 骨架加强 2D 视频"研究结论 + 实现（2026-08-22）**：
- **核心认知**：骨架 3D 是"外观/身份无关的显式身体结构"→ 训练时注入 = 域不变正则。同相机逐帧对齐 → 投影/逐帧监督免费。**骨架当"几何监督源"，不是"预测教师"**。
- **方法排序（研究子代理报告，含引用）**：①**辅助 3D 位姿回归头**（multi-task CE+MPJPE，视频学人体几何，测试摘头零开销）＝首选；②位姿引导空间注意力+热图早融合（PoseConv3D 式，但又有投影误差问题）＝次选；③部件级跨模态对比（需先修好骨架）＝候选。**不做**：弱教师 logits 蒸馏（0.45→0.711 拖累）、全局 InfoNCE（已实测中性，文献=异质粗对齐坍缩的已知失败模式）。
- **骨架深挖**：MotionBERT（AMASS/H36M 3D 预训练）本身就是 cross-subject 最强（NTU 97.2%）；关键 = 喂对原生 3D（已修）+ 完整 3D 旋转增强 + 可选 MAMP 域内预训练。
- **已实现 `scripts/auxpose.sbatch`**（辅助位姿头 fold0 60ep λ=0.1，对照 4ch 基线 0.6609）：`src/pose_aux.py`（PoseAlignedVideoDataset 视频帧↔对齐骨架帧）+ `scripts/train_auxpose.py`（encoder 输出 layer4 特征 → 每帧回归 [T',17,3] 原生3D，CE+λ·MPJPE）。测试摘头 → 模型大小不变、推理零开销。
- 并行三线：骨架_3d（骨架模型深挖）、auxpose（3D→2D 转移）、thermal_push（已跑，等结果）。

**⑯ auxpose v1 结果 + 根因（2026-08-23，job 41787）**：**fold0 = 0.6555 < 4ch 基线 0.6609（-0.54pt）**。机制在工作（mpjpe 12.3→4.6，模型确实在学 3D 位姿）但没转成跨被试增益。**根因两层**：①实现层——位姿头架在 layer4 池化特征（空间细节丢、无法定位关节），mpjpe~4.6（肩宽单位）≈ 噪声监督，污染分类梯度；②信息层——Depth+IR 本身已编码人体几何，骨架对 main 信息冗余（与帧差/对比学习同族：aux 给强 main 的都是冗余 → 无增益或轻微负）。**v2**（`auxpose.sbatch`）：位姿头改 **layer3**（空间细节）+ 小 λ（0.05/0.02 扫）→ 排除"位姿头弱=噪声监督"这一因素。判据：λ小+位姿头强 → main_val ≥0.665 则 3D 监督有效；否则信息冗余结论成立，关线。

**⑰ 模态"独特价值"结论（2026-08-23 读论文+物理拓扑）**：
- **关键物理事实**：骨架与 Depth+IR **同相机（NYX 650）→ 对 main 天然冗余视角**（同遮挡/同视角/同场景）。这统一解释所有 aux 实验失败：main↔skeleton 对比中性、aux 位姿头 -0.5pt、oracle≈main。
- **独特价值排序（对 main）**：**Thermal（独立相机，唯一真独立视角，论文随机分最强 92.57%）> IMU（身上惯性，遮挡无关但 cross-subject 弱）> Radar（非视觉 RF，遮挡/黑暗可用但弱）> Skeleton（同相机，冗余视角）**。
- **用法纠正**："置信门控/类感知救援"该用在 **Thermal**（独立视角，main 不确定时有真不同的看法），**不是骨架**（冗余视角救不了）。骨架 = 独立 3D 模型自用，别指望抬 main。
- **最高 EV 收束**：main(0.71144) + 提升后的 thermal(0.65，P1 已 0.619) 做类感知/置信门控集成 ← 这才是"多模态真正用起来"的位置。

**⑱ thermal 3 折 + "热像姿态桥"评估（2026-08-23）**：
- **Thermal 3 折（job 41823）**：fold0 0.6339 / fold1 0.5697 / fold2 0.6124 → **mean 0.6053（std 0.027）**。方差大主因 = **fold1 被试难度**（val_loss ~2.35 下不来）+ **热像主体敏感**（体温/衣物随人变，无深度通道 → thermal 天生 < main 且更不稳）。bbox 覆盖率 95.9%（2771/2891），YOLO 在铁火伪彩上不如灰度 IR 稳。增强与 main 同款够用但非针对性。
- **"热像姿态桥"想法（用户提出）**：姿态是跨传感器不变表示（文献：ThermalPose 等热像姿态估计存在）；从热像提姿态 = **独立视角 + 主体不变** → 理论可同时治"冗余"和"方差"。**但硬阻断 = 无热像姿态标签 + 无跨相机外参**（NYX↔Hikvision 标定缺失）→ 训不出可靠模型；且 thermal 图像模型已隐式含姿态。结论：概念成立但不可行，**不投入**。
- **Thermal 后续**：full 2-seed（thermal_push 跑完）+ **main→thermal 蒸馏**（distill_thermal.sbatch，治主体敏感）→ 目标 0.65 → main+thermal 类感知集成。

**⑲ 热像姿态门禁 **通过**（job 42110，2026-08-23）——推翻 ⑱"不投入"判断**：
- **结果**（`feasibility_thermal_pose.py`，KeypointRCNN COCO 预训练，torchvision 0.28 自归一化 `[0,1]`）：8 个动作 sample，**人检测率 mean=1.000，骨架完整(≥10kp)率 mean=0.950，关键点置信 6.5-13.3**。
- **修正 ⑱ 的两条"硬阻断"**：①无热像姿态标签 → **用 COCO 预训练 KeypointRCNN 零样本迁移**，不训热像姿态估计器；②无跨相机外参 → **2D 骨架归一化融合**（热像 2D + NYX 3D 各自归一化后拼接输入），不做 3D 投影对齐。
- **结论**：热像上能稳定检出人+骨架 → **"NYX 3D + 热像 2D 双视角骨架"模型值得建**。热像姿态 = 独立视角 + 主体不变，信息上与 main 互补性 > thermal 图像（结构化几何 vs 外观纹理）。
- **两条线关系**：热像姿态桥不是"叠加第三个模型"，而是"给骨架补独立视角" → 若成功，**替代 thermal 图像**（更小 20MB + 互补性更强），而非与之叠加。
- **大小账**：main int5 + thermal 图像 int5 = 79.5MB（✅ 短期提交）；main int5 + 双视角骨架 int8 = 59.9MB（✅ 长期替代）；三者全上 int8=100MB ⚠️ 卡线、int4 骨架=89.8MB ✅。
- **Next**：全量热像骨架提取（Step 2）→ 双视角骨架模型训练（Step 3，MotionBERT 系）→ 与 main 门控集成（Step 4）。

**⑳ 双视角骨架训练方法论定稿（2026-08-23 顶会调研 + 用户纠正）**：
- **✅ 数据确认为同步多视角**（用户纠正我"无同步多视角"的错误）：论文明写 "seven synchronized modalities" + "global time 对齐" + "well-aligned data pairs" → NYX 3D 与热像 2D 是同一动作同步对。**对比视点不变方法（2209.11634 类）可用**，这加强了 ViA/对比对齐路线。
- **P0 硬结论**：①绝不原始坐标 concat（[T,17,6] 病态）；②COCO-17 ≠ H3.6M-17（定义+顺序不同）必须写映射：12 直接映射 + 4 派生（骨盆=L髋R髋中点、颈=双肩中点、头=鼻颈中点、脊柱=骨盆颈中点）；③两个源各自 root-relative(减骨盆)+肩宽归一化（必要不充分，残留旋转差靠绕竖直轴随机旋转增强补齐）；④融合默认双分支+logits 加权（最稳），二阶 lifting 统一后特征级 concat。
- **置信度**：先统计是否退化（官方 Skeleton conf 曾恒 1.0 全退化！）。非退化 → [x,y,conf] 三通道（MotionBERT 2D 输入兼容）+ conf 加权 loss；退化 → 丢 conf。
- **预训练**：MotionBERT(ICCV23) 是事实多源(2D+3D)骨架预训练标准，H36M-17 权重兼容 [x,y,conf] 和 [x,y,z]。
- **PoseC3D 热图体积**（CVPR22 Oral）：唯一官方支持 2D/3D 双源输入、最抗噪，但实现重（要建 3D 热图体积 + 专用数据管线 + 大存储），放远期。
- **Step 2 已升级**（extract_thermal_skeleton.py）：提取时即做 COCO→H36M 映射 + root-relative + 肩宽归一化 + conf 退化统计（单元测试通过：骨盆归零/颈头派生/肩宽=1）。产物 kp[17,2] H36M 顺序归一化 + conf[17]。
- **P2 远期（用户确认保留）**：**PoseC3D / PoseConv3D 热图体积统一表示**（CVPR22 Oral，arXiv 2104.13586）——唯一官方支持 2D/3D 双源输入、最抗噪。难在工程：建 3D 热图体积（存储/I/O 膨胀几十倍）+ 专用数据管线 + 需 3D-CNN 主干（换架构、与 MotionBERT 资产脱钩）+ 超参敏感（σ/分辨率/置信度加权）。当前先用双分支+logits 加权融合（轻量、复用 MotionBERT），PoseC3D 排远期。

**㉑ Step 2 热像骨架提取完成（job 42136，2026-08-23）**：
- **2873/2891 clip 成功（99.4%）**，skip 18（无热像帧/无检测）。
- **置信度非退化 ✅**：conf mean=2.94 std=4.11 min=-6.87 max=25.03，conf>0.5 占 68%，**conf==1.0 占 0%**（无官方 Skeleton 的恒 1.0 退化问题）。
- ⚠️ **conf 是未归一化分数（非 [0,1]）**——训练时需处理（softmax/min-max），不能直接当概率。
- **结论**：2D 分支用 `[x,y,conf]` 三通道 + conf 加权 loss **可行**（先对 conf 归一化）；产物 `outputs/thermal_skeleton/<Action>/<Subject>/<sample>.npz`（kp[17,2] H36M 顺序归一化 + conf[17]）。
- **Next**：Step 3 双分支训练（2D 分支 [x,y,conf] + 3D 分支 [x,y,z] + logits 加权 + 绕竖直轴旋转增强）。

**㉒ Step 3 双分支训练已实现（用户确认架构：共享 backbone + 双输入头，2026-08-23）**：
- **架构（顶会支撑）**：共享 DSTformer encoder（两模态共享运动先验 + 100MB 下省 ~40MB + 缓解小数据过拟合；风险=模态干扰 → 模态特定输入头吸收域差，成熟 trade-off）+ 双输入头 `input_head_2d`([x,y,conf]) / `input_head_3d`([x,y,z])（都 Linear(3,dim_feat)，用预训练 joints_embed 初始化——MotionBERT 多源预训练已见 2D+3D）+ 双分类头 + logits 加权融合（默认可学习权重，sigmoid 初始 0.5）。
- **关键代码**：`src/dual_branch.py`（DualBranchActionNet，_branch 复刻 DSTformer encoder，替换 joints_embed=head=Identity 防误用）；`src/dual_dataset.py`（DualSkeletonDataset + build_dual_pairs 过滤两源齐全；3D 分支绕竖直轴 ±rot_angle 旋转补视点差——NTU cross-view 标准；2D 分支 xy 平面增强，conf 不动）；`scripts/train_dual_branch.py`（fused/dual loss、learn/fixed fusion、conf_norm sigmoid/minmax/none）；`scripts/dual_branch.sbatch`（cuhkx-dualbranch）。
- **conf 归一化**：默认 sigmoid（KeypointRCNN 原始分数 -6.87~25 → (0,1) 概率域，贴近 MotionBERT 2D 预训练输入）；minmax 需全局扫描（build_dual_pairs 返回）；none=恒 1.0 对照。
- **自测通过**（tests/selftest_dual_branch.py，合成数据）：build_dual_pairs 过滤 23/24、数据集形状 (16,17,3)、forward 形状 (B,40)、fused/dual backward、mini 训练 + eval 冒烟。
- **Next**：`sbatch scripts/dual_branch.sbatch` 跑 3 折（基线对照：skeleton_3d 单分支 3D ~0.45；热像 2D 单分支待 auxpose/skeleton 对照）。判读：fused_val 应 > 单分支 3D；2d/3d 分支各自 acc 看模态贡献；w 收敛方向看主导模态。

**㉓ 双分支首次训练 + 2D 分支不学诊断（job 42191，2026-08-23）**：
- **结果**：3 折 mean 0.4583（std 0.02），收敛正常（loss 3.27→0.83，fold0 早停 ep19）。但 **2D 分支 acc 恒定 ~0.1（接近随机）**，3D 分支 0.40-0.47 正常；w 收敛 0.425（偏 3D）。
- **代码检查**：dual_branch/dual_dataset/train_dual_branch 逻辑正确、自测过，无 bug。2D 不学 = 共享 encoder + fused loss 下 3D 主导压制（系统性架构特性，非代码缺陷）。
- **根因二选一**：①热像 2D 骨架质量差（数据） vs ②3D 主导压制（架构）。诊断脚本已写：`scripts/train_thermal_2d.py`（2D-only 单分支 + `--mode stats` 数据质量统计）+ `scripts/diag_dual_branch.sbatch`（三连：stats → 2D-only → dual loss 双分支）。
- **顶会资料（模态不平衡/主导压制，检索确认）**：
  - **AIM: Adaptive Intra-Network Modulation**（arXiv:2508.19769）——关键洞察：**简单抑制主导模态救弱模态会损害整体性能**；方案=按网络深度评估不平衡、自适应调节调制强度，把主导模态欠优化参数解耦为 Auxiliary Blocks 供弱模态联合训练。
  - **Multimodal Negative Learning**（NeurIPS 2025, arXiv:2510.20877）——不强迫弱模态对齐主导模态（Positive Learning 会压制弱模态独有信息），改为让主导模态动态引导弱模态**抑制非目标类**（Negative Learning），提升 Unimodal Confidence Margin。
  - 其它：Contribution-Guided Asymmetric Learning（arXiv:2510.26289）、GMM-Guided Adaptive Loss（arXiv:2510.21797）、Mixup 抗强模态过拟合（arXiv:2510.10986）、PMR（CVPR 2023）、OGM-GE（梯度消除）。
- **判读规则**：stats 显示关节大量无效/坐标退化 → 根因①（转提高热像提取质量）；2D-only acc 0.3+ 但双分支里 ~0.1 → 根因②（用 AIM/MNL 式模态平衡：按深度调制、负学习、或两阶段）。

**㉔ 热像姿态桥诊断结论 + main+thermal 融合突破（2026-08-24）**：
- **热像姿态桥收束（诊断 job 43723 三连）**：
  - **[1] stats**：热像 2D 骨架**下肢全废**——踝关节有效比例仅 4-5%、膝 24-26%；上半身（头/颈/躯干/肩 0.93-0.99）可靠；kp 坐标 std=8.49（归一化被失效肩宽放大，异常）。根因=热像 320×240 低分辨率 + 视角下 KeypointRCNN 只能稳定检测上半身。
  - **[2] 2D-only 单分支**：3 折 mean **0.2409**（fold0 0.279）→ 热像 2D 骨架天花板极低。**根因①（数据质量）确认为主因**。
  - **[3] dual loss 双分支**：0.4547（fold0 0.4900），2D 分支从 fused 的 ~0.1 → 0.21-0.26（共享压制 ② 证实存在），但整体无增益（vs fused 0.4583）——2D 信息量本身不足。
  - **结论**：热像 2D 骨架不是好表示（天花板 0.24 << 3D 0.45），把热像降维成骨架反而丢掉热像最值钱的**温度/外观纹理**。热像正确用法 = **视频模型直接用**（R2Plus1D thermal 已 0.6053）。
- **⭐ main + thermal 融合 LB 0.73（2026-08-24，重大突破）**：
  - 配置：main 单 seed42 int5(40MB) + thermal 单 seed42 int5(40MB) = **80MB**，prob 平均 + flip TTA（`submit_fusion_thermal.sbatch`）。
  - **LB 0.73，比 0.71144 高 +1.86pt！** 等预算（80MB）下"1×main + 1×thermal 独立视角" > "2×main" → **融合线成立**。
  - 验证了 §⑪/§⑫ 早期判断："Thermal 独立视角天然互补"——热像作为视频模态融合是主收益路径。
  - **Next（满预算 100MB 优化）**：a) main seed42+777 双 seed + thermal（int4/降位压缩到 ≤100MB）；b) thermal 双 seed 42+777；c) 加权融合调优（thermal 稍降权）；d) 加 3D 骨架弱成员（谨慎，可能稀释）。

**㉕ 方案 A 满预算失败（2026-08-24）——int4 量化崩，锁定 0.73**：
- 配置：main int4 双 seed(42+777, 64MB) + thermal int4(42, 32MB) = 96MB（submit_fusion_full.sbatch 自适应降级命中 [2]）。
- **LB 0.71 < 0.73** → **int4 对 R2Plus1D 精度损失严重**（双 seed 增益 + thermal 增益都盖不住 int4 掉精度）。
- **结论**：int4 不可用（量化误差放大到分类崩溃）。**0.73 配置（main int5 单 seed42 + thermal int5 单 seed42 = 80MB）是当前最优保底**，submission_fusion_mT.csv。
- **铁律更新**：R2Plus1D 提交级最低 int5，int4 不进提交线。
- **零成本下一试**（权重已全训完，只重打包+推理）：换 seed 配比 main(777)+thermal(42)、main(42)+thermal(777)、main(777)+thermal(777)——可能找到 >0.73 的组合。20MB 余量塞不进任何合格模态（skeleton int5 40MB 超），暂不加成员。
- **⚠️ 纠正（2026-08-24）**：main/thermal 全量**都已用类平衡重采样**（train_step1.py 默认 balanced，`--no_balanced` 才是关闭）——我之前"thermal 没用 balanced"是误判（只看了 sbatch 没看 train_step1.py 默认值）。无需重训。
- **时间 TTA 已实现但窗口实现降质（2026-08-24，⚠️ 已回退）**：src/dataset.py 给 DepthIRVideoDataset/ThermalVideoDataset 加 `sample_offset`；ensemble_inference.py 加 `--time_tta N`。**但 `--time_tta 3` 实测把 0.73 拖到 0.65174（-8pt）**——根因：实现用**连续 16 帧窗口**（offset 窗口 = linspace(start, start+15)），10fps 只覆盖 1.6 秒局部快照，丢了全 clip 时间多样性；而原 endpoint-uniform 覆盖全 clip。**视频模型（R2Plus1D）不能丢时间多样性**。已从 submit_fusion_thermal.sbatch 回退（去 --time_tta，恢复 0.73 配置）。教训：视频 TTA 的窗口必须**段内均匀覆盖**（或带偏移的全程均匀），不能用连续帧窗口；flip TTA 已足够，时间 TTA 暂缓（需正确实现再启用）。另外 job 43793 大小检查显示 thermal_fold0_int5.pth=0 字节（du 对残留路径误报嫌疑大，因推理未崩；若真 0 字节需重新打包）。
- **IMU/Radar 判读（2026-08-24）**：论文 §5.2.1 它们是最弱模态（IMU 1D-CNN 45.52% / Radar PointNet 46.63%，80/20 随机分）。推理融合等权大概率降质量（弱模态噪声拖累，thermal/skeleton 等权拖累教训）；加权/类感知才可能正，风险>收益。对比学习论文 §6.1.3 有效但小数据(2891)未复现（s2 负贡献）。**IMU/Radar 非 80+ 关键，不投入。**

**㉕b 竞赛规则澄清（2026-08-24，组织方官方回复）**：
- **"no large pretrained backbones" 具体指**：禁 LLM / 大型 VLM 基础模型。**小型标准预训练 CNN（如 ImageNet ResNet18 ~44MB）明确允许**。
- **IG65M R2Plus1D-34（63M 参数，int5 40MB）合规** ✅——标准 CNN 架构，非基础模型。维持 0.73 保底不动。
- 允许：自定义融合模块（attention/cross-attention/gating）、MLP head、投影层；非 DL（GBDT/SVM）可作 pipeline。
- **唯一硬限制**：推理加载的全部权重打包 <100MB（我们 80MB ✅）。
- **含义**：①IG65M 可放心用；②"更强预训练"不必换（VideoMAE 基础模型反有风险）；③Stacking（GBDT on val logits）合规可做；④数据固定 2891 无额外。
- **auxpose-thermal 已实现（2026-08-24）**：骨架几何监督注入 **thermal**（main 的 Depth 已含深度几何冗余，v1 注入 main 失败 -0.54pt 印证）。新代码：`ThermalVideoDataset._sample_indices`（抽采样方法）+ `src/pose_aux.py: ThermalPoseAlignedVideoDataset`（继承 ThermalVideoDataset，**跨相机按帧比例对齐骨架**——thermal 25fps vs skeleton 10fps 不同基准，global time 同步 → round(j·(n_s-1)/(n_t-1))；缺失帧 mask=0）+ `train_auxpose.py --modality thermal`（3ch，crop 用 bbox_thermal_train.json）。合成自测通过（形状/跨相机对齐/缺失掩码）。提交：`scripts/auxpose_thermal.sbatch`（fold0 60ep λ=0.05/0.02 layer3）。**判读：main_val ≥ 0.64（thermal fold0 基线 0.6339）则骨架几何监督 thermal 有效 → 全量重训 thermal 融合冲 0.73+**。
- **❌ auxpose-thermal 失败（job 43828，2026-08-24）**：λ=0.05 → fold0 **0.5966**（-3.7pt vs 基线 0.6339）；λ=0.02 → fold0 **0.6211**（-1.3pt）。都收敛（loss/mpjpe 平滑下降）但 val 落后基线，**λ 越大越差 → MPJPE 骨架监督是负向正则**。根因：MPJPE 收敛到 ~5.8 非 0（跨相机比例对齐监督有噪声）+ 位姿回归分散分类注意力。**骨架几何监督路线彻底关闭**（main v1 -0.54pt + thermal -1.3/-3.7pt 两次失败）。骨架作为独立分类（0.45 弱）或检测（YOLO 已用）保留，不做监督。0.73 保底不动。**Next：seed 配比探索 / Stacking（规则允许 GBDT on val logits）**。
- **B 线骨干对比已实现（2026-08-24，用户确认先冲 B）**：新增 `src/backbone_compare.py`（骨干工厂，torchvision Kinetics 预训练 + 40 类/4ch 适配）+ `scripts/train_backbone_compare.py`（fold0，复用 train_step1 recipe）+ `scripts/backbone_compare.sbatch`（r2plus1d34 基线 + swin3d_t + x3d_m，thermal 3ch 先验证）。**本地验证：swin3d_t（27.9M，128×128 直接跑，3ch/4ch 都 OK）✅；s3d/mvit_v2_s 需 224 输入（128 下 avgpool/patch 崩）排除；x3d_m 待服务器（torchvision 0.28 应有）**。判读：谁 fold0 > 0.6339（thermal 基线）→ 换骨干全量重训。**全部新增文件，不触碰现有主线（可复位）。**
- **骨架平滑重训已实现（2026-08-24，用户强调骨架有用）**：`src/skeleton_smooth.py`（smooth_sequence 滑动平均去噪 + SmoothMotionBertSkeletonDataset 继承父类）+ `train_skeleton_motionbert.py --smooth`（默认 1=关可复位）+ `scripts/skeleton_smooth.sbatch`（repr3d smooth5 3折，对照 skeleton_3d ~0.45）。本地验证：平滑去噪生效（std 0.498→0.475）、window=1 完全等同、Smooth 数据集端到端 OK。**判读：mean > 0.45 则平滑有效。骨架平滑实测 mean 0.4592（std 0.0111），对比无平滑 0.4513 仅 +0.8pt → 已近 cross-subject 天花板**。
- **⭐ 伪标签自训练官方允许（2026-08-24，Kaggle 讨论区确认）**：测试数据伪标签 + 自训练/半监督**不违规**（组织方 Gvine15 确认）。405 无标签测试 clip 可用 0.73 模型打伪标签加入训练。⚠️ Caveat：Stage 2/3 有 unseen subjects，勿过拟合当前测试分布，需 held-out 验证。Leakage 已修复 + 反作弊（勿用泄漏）。**这是当前最高 EV 方向（合法大杠杆，可能是 80+ 队伍方法）。**
- **B 线报错修复（2026-08-24）**：train_backbone_compare 统一 permute 导致 r2plus1d34 双重 permute（R2Plus1D34.forward 内部已 permute(0,2,1,3,4)，期望 [B,T,C,H,W]；swin3d/x3d 期望 [B,C,T,H,W]）。修复：`_to_model_input(x, backbone)` 按骨干分发（r2plus1d34 不 permute）。本地验证：R2Plus1D34 通过 stem/layer 到 layer4（原在 stem 崩）、swin3d_t permute 正确。**服务器重跑 backbone_compare.sbatch。**
- **伪标签自训练已实现（2026-08-24，用户策略，官方允许）**：`scripts/pseudo_label_selftrain.py` 两阶段：`--stage pseudo`（全量 main+thermal seed42 对 405 测试推理 → **Co-training：top1 一致 + 保守置信度 min(max_prob) → 动态 top-K（默认 120）** → pseudo_labels.json）；`--stage selftrain`（fold0 真标签 + 伪标签测试 clip，**伪标签样本低权重 pseudo_weight=0.5 可靠性加权 CE**，fold0 val 早停防过拟合）。`scripts/pseudo_label_selftrain.sbatch`（pseudo → selftrain thermal fold0 对照 0.6339 → selftrain main fold0 对照 0.6609）。本地自测：Co-training+top-K 筛选逻辑验证通过、py_compile OK。
- **❓ 伪标签 fold0 结果（job 44054，2026-08-25）**：伪标签采纳 top-120 置信度 0.907（质量高）；**thermal selftrain fold0=0.6275（-0.64pt vs 0.6339）；main selftrain fold0=0.6566（-0.43pt vs 0.6609）**——微降但在 thermal std 0.027 噪声内。**关键洞察（用户）：fold0 val 是训练分布，伪标签样本来自测试分布（12 unseen subjects），自训练针对测试分布 → fold0 val 测不出伪标签价值（甚至反向）；且 thermal fold0(0.6339) 是较好折，fold1 仅 0.5697，单折评价噪声大。结论：伪标签 fold0 微降≠无效，**正确评价=全量自训练+烧 LB**（决定性实验）。**
- **B 线 x3d 排除（2026-08-25）**：服务器 torchvision `models.video` 无 `X3D_M_Weights` 导入 → backbone_compare.sbatch 去掉 x3d_m，只跑 r2plus1d34 + swin3d_t。s3d/mvit 需 224 也已排除。**swin3d_t（27.9M）参数比 R2Plus1D-34 少一半，大概率非更强（用户质疑正确），B 线 fold0 确认后关闭。**
- **VideoMAE-S 下载源（2026-08-25）**：官方 MODEL_ZOO，**VideoMAE-S K400 1600ep 预训练** Google Drive：`https://drive.google.com/file/d/1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww/view`（本地 VPN 下载 → 传服务器 `weights/videomae_s_k400_pretrain.pth`）。**关键：官方权重是裸 state_dict（非 transformers 格式）→ cuhk_x 无 transformers 也不需要！**手写 ViT-S 结构 + 加载 key 映射即可。大小账：VideoMAE-S int5 ~13.8MB；**2×Base(86M) int5=107.6MB 超**，但 main(R2Plus1D 40MB)+thermal(VideoMAE-S 14MB)=54MB 或 thermal 单换均可行。main 单 seed42=0.70646（实测）。
- **提交批次规划（2026-08-25，用户要求 seed 融合 + 伪标签全量都烧 LB）**：
  - **批次1 seed 配比探索**（`submit_fusion_seed_sweep.sbatch`，零训练）：main42/777 × thermal42/777，**m42_t42=0.73 已交不重跑**，只跑 3 个新组合（m42_t777 / m777_t42 / m777_t777）→ 3 个 csv，找 >0.73 组合。
  - **VideoMAE-S 接入澄清（2026-08-25）**：手写 ViT-S + key 映射 = transformers 效果**完全一样**（结构确定+权重相同→输出相同；手写风险在 key 映射，用加载覆盖率验证杜绝）。官方权重裸 state_dict **就是预训练好的权重**（encoder 全层参数），选 **pretrain**（自监督更域鲁棒）非 finetune。
  - **✅ VideoMAE-S 接入完成（2026-08-25）**：inspect 实测权重结构（encoder.patch_embed Conv3d(3→384,(2,16,16)) + 12 blocks(qkv 384→1152 bias=False + q_bias/v_bias 分离) + norm；**无 pos_embed/cls_token → 随机初始化**；mask_token/decoder 丢弃）。代码：`src/videomae.py`（VideoMAES 结构 + `load_videomae_pretrained` 加载覆盖率验证，**实测 160/160=100%**）+ `src/dataset.py` ThermalVideoDataset 加 `mean_std`（VideoMAE 用 ImageNet 0.485/0.229 归一化）+ `scripts/train_videomae.py` + `scripts/videomae.sbatch`（fold0 thermal 60ep）。本地自测：forward 形状、加载覆盖率 100% 全过。**判读：fold0 > 0.65（基线 0.6339）→ thermal 全量换 VideoMAE-S。**
  - **批次2 伪标签全量**（`submit_pseudo_full.sbatch`）：pseudo → selftrain --full main/thermal(80ep 含 120 伪标签 weight 0.5) → 打包 int5 → 融合提交（80MB）。`pseudo_label_selftrain.py` 已加 `--full` 模式（无 fold val，每 ep 覆盖保存）。
  - **批次3 VideoMAE-S**（等用户下载权重）：thermal 换 VideoMAE-S → 融合（等接入）。
  - 提交顺序：先批次1（零成本）→ 批次2（伪标签决定性）→ 批次3（VideoMAE）。
