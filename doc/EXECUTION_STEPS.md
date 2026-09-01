# CUHK-X 训练步骤执行手册（2026-08-26）

## 📌 当前状态快照

| 状态 | 配置 | 数字 |
|---|---|---|
| 🟢 **保底（绝不动）** | main_s42 + thermal_s42（各 int5，80MB）prob 平均 + flip TTA | **0.73 LB** |
| 🔵 main 本体真实主力 | 全量 seed42+777 | 0.71144 LB |
| 🔴 已关闭（结果确认） | dual_fusion 特征级融合 | 0.6114（修复后真实水平）|
| 🔴 已关闭（结果确认） | main-2fold 复现公开 0.715 | 0.6055（公开训练细节从未公开，不可复现）|
| 🔴 已关闭（历史） | VideoMAE / 伪标签 / 门控 / 高分辨率 / auxpose / 对比 / IMU / NTU-微调 | — |

**关键认知**：公开 0.711 的 parent_val_acc [0.7156, 0.7187] 来源不明、训练细节从未公开 → **不是可复现目标**。我们的真实参照 = 0.71144 LB（全量）+ 0.73（融合）。

---

## 🎯 训练步骤（按投入产出比排序，每步都有判据和终止条件）

### 步骤 0：保底
✅ 已提交 0.73，不动。

### 步骤 1：零训练成本（只烧 spare 提交，不碰保底）

**1A. v3 bounded-prior（类偏置修正，最高优先）**
```bash
python scripts/ensemble_inference.py --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth --quantize --flip_tta --prob_avg \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --prior_experiment --output submission_main.csv
```
- **判据**：输出 `soft-count imbalance` ≥2 → 模型类偏置 → 烧 spare 提交测 `submission_prior.csv`；<2 → 跳过
- **依据**：40 类长尾 29.5 倍 = 极度 class-biased；更新版测量典型 +2.3%，最差 −1.7%

**1B. Depth-fallback 检查（crop 质量）**
- 检查 `bbox_test.json` 里 fallback（全帧）clip 数量；有 → 用 Depth_Color 重探测（UPDATE 1）

**1C. YOLO 精细部位特征（零训练，GPU 检测）**
```bash
sbatch scripts/yolo_aux.sbatch
```
- **判据**：两个 GBDT mean acc 显著 >2.5%（随机基线）→ 部位信息有信号；≥0.3 且与 main 互补 → 决策级融合

### 步骤 2：骨架线（独立于 main，先修数据再加权）

**2A. 诊断 + 清洗**（对照 0.5447）
```bash
sbatch scripts/skeleton_augmented_clean.sbatch
```
- **判据**：全量诊断坏 clip 比例 + fold0 >0.5447 → 清洗有效

**2B. 清洗 + part-aware 部位权重**（对照 clean-only）
```bash
sbatch scripts/skeleton_augmented_part.sbatch
```
- **判据**：fold0 >0.5447 且 >clean-only → 部位加权有效

### 步骤 3：融合线（低预期，跑一次定生死）
```bash
sbatch scripts/misa_dual.sbatch
```
- **判据**：fold0 >0.66 留 / <0.66 关（dual_fusion 已 0.6114，MISA 正则可能不同但别抱期待）

### 步骤 4：决策汇总

| 结果 | 动作 |
|---|---|
| 1A imbalance ≥2 且 prior 提交 LB 升 | prior 并入最终提交 |
| 1C GBDT 有信号 | 部位特征做决策级融合 |
| 2A/2B 骨架提升显著 | 骨架评估参与融合（警惕历史拖累）|
| 全部边际 | **0.73 定稿** |

---

## ⚠️ 铁律
- 0.73 保底绝不动；所有实验 fold0/val 判据，不烧主 LB
- StepLR（不用 CosineAnnealing 衰减到 0）；sbatch 必须 LF
- 花哨方法全败教训：新方法必须 fold0 对照，失败即关
