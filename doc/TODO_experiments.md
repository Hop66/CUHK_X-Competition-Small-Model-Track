# CUHK-X 待做实验清单（2026-08-26 更新）

> 原则：全部 fold0 / val 判据，不烧 LB；0.73 保底绝不动；StepLR 铁律；LF 铁律。
> 所有新实验都是"代码就绪 + 外部文献背书或零成本确定性"，失败即关闭，不做无谓投入。

## 优先级 0：伪标签全量自训练（最高 EV，唯一合法大杠杆）

**背景**：405 无标签测试数据是**同分布**（比任何外部数据集相关），官方允许（Gvine15 确认，可能是 80+ 队伍方法）。伪标签质量高（top-120 置信度 0.907）。
**现状**：fold0 验证**已做**（job 44054：thermal 0.6275 / main 0.6566 微降）；**全量自训练 + 烧 LB 唯一没做**（代码 `--full` 已就绪）。
**依据**：fold0 val 是训练分布，测不出伪标签价值（伪标签针对测试分布）——决定性实验必须全量 + 烧 LB。

| # | 实验 | 文件 | 判据 | 状态 |
|---|------|------|------|------|
| 0 | 伪标签全量自训练（main+thermal --full）+ 与 0.73 决策级融合 | `scripts/pseudo_label_selftrain_full.sbatch` | 融合后 LB >0.73 → 伪标签有效 | ✅ 已实现待跑 |

```bash
sbatch scripts/pseudo_label_selftrain_full.sbatch
# 产物 outputs/selftrain_full/{main,thermal}_selftrain_full.pth → 与 0.73 ensemble 融合 → 烧 LB
```

## 优先级 1：main 本体（有公开证据链）

| # | 实验 | 文件 | 结果 | 状态 |
|---|------|------|------|------|
| 1 | main 2 折 subject-CV + best-val 早停（对齐公开 0.711） | `scripts/main_2fold_earlystop.sbatch` | fold0 best=**0.6055**（ep53）<< 公开 0.7156 | ❌ 关闭（公开 2 折 recipe 复现不成立；2 折用一半数据反而弱于全量 0.706 LB） |

## 优先级 2：融合（外部文献背书）

| # | 实验 | 文件 | 结果 | 状态 |
|---|------|------|------|------|
| 2 | main+thermal 早融合（RGBPoseConv3D 背书） | `scripts/dual_fusion.sbatch` | fold0 best=**0.6114**（修复 StepLR 后）< main 0.6609 | ❌ 关闭（用户质疑正确：特征级 concat+MLP 在我们小数据上 < 决策级 0.73；0.6114 是真实水平非 bug）|
| 3 | MISA 式多任务正则 | `scripts/misa_dual.sbatch` | 待跑（同类特征级，预期低；但 diff/sim/recon 正则可能不同，跑一次定生死：>0.66 留 / <0.66 关） | ⏳ 可跑一次 |

## 优先级 3：骨架数据质量（用户质疑"为什么不针对性解决"）

| # | 实验 | 文件 | 判据 | 状态 |
|---|------|------|------|------|
| 4 | 骨架全量质量诊断 | `scripts/diag_skeleton_quality.py` | 坏 clip 比例/vel_p99/肩宽 std → 定清洗强度 | ✅ 已实现 |
| 5 | 骨架清洗对照（零帧插值+坏帧修复+平滑） | `scripts/skeleton_augmented_clean.sbatch` | fold0 >0.5447（不清洗基线）→ 清洗有效 | ✅ 已实现待跑 |
| 5b | 骨架 part-aware 部位权重（可学习每关节权重，ST-GCN/CTR-GCN 同思想） | `scripts/skeleton_augmented_part.sbatch` | fold0 >0.5447 且 >clean-only → 部位加权有效 | ✅ 已实现待跑 |

## 优先级 4：YOLO 精细部位 / 运动特征（新方向，用户提出）

**洞察**：现在 YOLO 只检测 person bbox 且缓存只存合并窗口，逐帧"人怎么动"全丢。动作识别缺的就是运动线索。

| # | 实验 | 文件 | 判据 | 状态 |
|---|------|------|------|------|
| 6 | bbox 运动特征（逐帧 YOLO → 中心轨迹/面积/宽高比/方向）→ 辅助分类 | `scripts/yolo_motion_feats.py` | 特征单独 GBDT acc 显著 >随机；与 main 决策级融合 | 🔨 落地中 |
| 7 | YOLO pose 关键点（YOLO11n-pose 17 COCO 关键点+置信度）→ 部位运动特征，替代/对照 MMPose 3D | `scripts/yolo_pose_kpts.py` | 关键点特征 acc；或作为新骨架来源对照 0.54 | 🔨 落地中 |
| 8 | 骨架先验手工特征（H3.6M-17 关节语义 × 40 类动作模式，如手-嘴距离/腿摆幅）| 依赖诊断 | 先跑 #4 诊断，按坏 clip 分布决定 | ⏸ 待诊断 |

## 运行命令汇总（全部 fold0/val 判据，不烧 LB）

```bash
# P1 main 本体：2 折 + best-val 早停（对齐公开 0.7156/0.7187）
sbatch scripts/main_2fold_earlystop.sbatch

# P2 融合（外部文献背书）：早融合 + MISA 多任务正则
sbatch scripts/dual_fusion.sbatch
sbatch scripts/misa_dual.sbatch

# P3 骨架：诊断→清洗→part-aware（顺序做，先修数据再加权）
sbatch scripts/skeleton_augmented_clean.sbatch    # 内含全量诊断 + 清洗版（对照 0.5447）
sbatch scripts/skeleton_augmented_part.sbatch     # 清洗 + 部位权重（对照 clean-only）

# P4 YOLO 精细部位：bbox 运动特征 + YOLO-pose 关键点 + GBDT 评估
sbatch scripts/yolo_aux.sbatch
```

## 优先级 5：推理侧零训练优化（来自 yolo-for-cuhk-x.ipynb 更新版，2026-08-26）

**情报**：公开 0.711 notebook 的 fork 更新版确认——训练细节从未公开；以下全是推理侧零训练 trick：

| # | 实验 | 文件 | 判据 | 状态 |
|---|------|------|------|------|
| 9 | v3 bounded-prior（朝均匀先验调 logit，只翻低置信度<0.55 clip，cap 6%；类偏置修正，典型 +2.3%）| `scripts/ensemble_inference.py --prior_experiment` | 跑 0.73 ensemble 看 soft-count imbalance（≥2 才值得烧 spare 提交）| ✅ 已实现单测过 |
| 10 | Depth-fallback crop（原队伍 10/405 IR probe 失败走全帧；用 Depth_Color 重探测）| 检查 bbox_test.json fallback clip → 重探测 | 有 fallback clip 且修正后不降 → 并入 test crop | ⏳ 待服务器查 |
| 11 | 骨架 torso-scaled 归一化（更新版建议 pelvis+torso；我们目前 shoulder-scaled）| 骨架线加 torso 对照 | fold0 对照 clean/part | ⏳ 低优先 |

**已确认**：v2 damped-Sinkhorn balancing 丢 LB → 我们 prob 平均 + argmax 正确，不做 score balancing。
**已确认**：Thermal 是论文最强单模态、物理独立于 Depth/IR → 印证 0.73 融合方向对。

## 优先级 6：损失函数层面（0.7→0.8 跳板，用户情报 2026-08-27）

**情报**：很多队伍刚从 0.7 大幅升到 0.8+ → 0.8 可达到。关键空白：所有实验用 CE，从未试 metric learning。cross-subject 本质=人脸/ReID 跨身份泛化，ArcFace/CosFace/Center Loss 是核心方法。

| # | 实验 | 文件 | 判据 | 状态 |
|---|------|------|------|------|
| 12 | main + ArcFace（跨身份类内紧凑+类间分离）| `scripts/main_arcface.sbatch` | fold0 >0.6609（CE 基线）→ 全量重训冲 0.8 | ✅ 已实现待跑 |

**其他空白**：Radar 模态（唯一没碰）、伪标签多轮、全帧/多段利用、Center Loss。

## 优先级 7：新调研方向（用户指定，需先评估合规）
跨模态迁移、提示学习（visual prompt）、MLLM 适配、VTFusion（IEEE TCyb few-shot anomaly）。
⚠️ 竞赛禁 LLM/VLM 基础模型——MLLM/CLIP 类需合规评估；视觉提示（VPT）可能合规。

## 已关闭（不重试）
VideoMAE(0.31)、伪标签(0.69651)、类感知/门控/动态融合(0.64-0.66)、高分辨率(0.6448)、
auxpose 骨架监督(main -0.54pt / thermal -1.3~-3.7pt)、对比学习(s2 -1.7%)、NTU 预训练-微调(换汤不换药)、
IMU(0.3547)、2D 热像骨架(0.24)。

## 诚实预期
- #6/#7 是弱特征（bbox/关键点运动对 40 类区分度有限），预期单独 acc 0.3-0.45，**价值在决策级互补**（+0-1pt）
- #7 YOLO pose 关键点带真实置信度 + 2D 无 3D lift 深度噪声，可能比 MMPose 3D 骨架（0.54）质量高
- 最终 0.8 仍受数据量/域差硬约束；这些是边际优化，不是银弹
