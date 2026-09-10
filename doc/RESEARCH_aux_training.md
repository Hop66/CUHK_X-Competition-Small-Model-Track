# 辅助主模态训练方法调研（LUPI / Privileged Distillation 家族）
> 场景: CUHK Multimodal HAR 40类 LOSO（train 18 被试 / test 4 全新被试）
> 主模态: Depth+IR 视频(R2+1D34,16f) fold0=0.6695 / thermal(32f)=0.6478 → 推理纯相机
> 辅助: 骨架(17×3D, 神经~0.53) / IMU(30ch,~0.33) —— 单模弱, test 也被提供
> 硬约束: 模型≤100MB(int5≈39MB,int4 掉3pt), 推理不依赖骨架/IMU(孤立弱成员 LB -0.5)

## 0. 已判负 / 持平（勿重复）——本地实证
| 方法 | fold0 | 结果 |
|---|---|---|
| 同模态 KD(teacher=main×3+th×3 软标签→student) | 持平 | **=train_distill_stream --modality main, 已训(main_ds)未胜 seed42** |
| transductive 软蒸馏(test soft 标签) | -1.3 / +3.7 | fold 矛盾,关 |
| 端到端特征双流融合(main+骨架GRU gate) | 0.6114 | -5.8 判负 |
| main 输入侧加骨架 heatmap(7ch) | 0.6383 | -3.1 判负 |
| 弱模态 prob-avg 推理成员 | LB | 每次 -0.5 |
| L2 aux-head(骨架/IMU 监督 main) | 0.634-0.645 | -2.5~-3.6 判负 |
| 难对推理 override(手工特征换头) | fold+1.7~2.6 | LB -0.5 判负 |
| KD 蒸馏骨架(teacher=main+th→skel student, 56140) | 0.4984 | -3.3 判负 |

**共同模式**: main16f 训练侧任何"加东西"(输入通道/aux头/域增强/mixup)系统性 -2~-7pt；
推理侧融合是唯一正杠杆(anchor 0.75124)。且"强教师蒸馏 main"(main×3+th×3 soft→main
student) 已试过 = train_distill_stream, 与 seed42 持平未胜。原因: IG65M 预训练主干在
2931 样本上已近饱和, 训练侧任何激进改动破坏已学好的 Kinetics 先验。

## 1. LUPI 家族谱系（辅助主模态训练的理论框架）
- **SVM+ / LUPI** (Vapnik & Vashist, 2009): 特权信息 X* 只在训练期出现, 通过"修正函数"
  (correcting function) 放宽 SVM 间隔, 测试不用 X*。核心=用特权模态给"难样本/误分类样本"
  提供额外松弛。
- **Generalized Distillation** (Lopez-Paz et al. ICLR 2016): 把 LUPI 统一为"teacher(X*)→
  soft label, student(X)→ 模仿", 即 >KL(teacher||student)。→ **特权蒸馏**。
- **FitNets / Hint-Based** (arXiv:1412.6550, ICLR 2015): 不止 output, 还迁移 teacher
  **中间特征**(hint), 学生加映射层对齐中间表征。→ 中间层对齐的雏形。
- **Relational KD (RKD)** (arXiv:1904.05068, CVPR 2019): 转移**样本间关系**(距离/角度),
  而非单样本输出。对本文→ 类内/难对结构关系。
- **Deep Mutual Learning (DML)** (Zhang et al. CVPR 2018): 无预训 teacher, N 个模型互教,
  用 KL 对齐。→ 可能让 main/th 双流互学(已部分内化为融合)。

## 2. 结合我们场景: 还没试过的增量在哪

### ❌ A. 特权软标签蒸馏 main —— **已试过=train_distill_stream, 持平**
- teacher=main×3+th×3 soft, distill_w=0.5, T=3, 80ep, IG65M init = 已有的
  main_distill/main_ds。结果未胜过 seed42(锚仍 main_s42)。
- **为什么持平**: 3 折 teacher 的 soft 已经"接近 one-hot + 小噪声", 对比原生训练
  的多样性几乎没增加; 且对 fold 内样本 teacher 是"未来信息"(同折训练暴露)。
  → LUPI 期望的"强监督"没兑现。

### 🥇 B. 难对区 reweight 蒸馏/训练 —— **真正新 + 直击难对（推荐）**
- 结合上轮难对判别数据: 神经各成员难对二分类 0.585~0.662、main 整体难对 0.662,
  main per-pair 弱区是 (26,24)0.39 / (22,21)0.55。
- **做法**: 用 main+th OOF 融合概率做 teacher（强、同域、train 折无泄漏时不透漏）
  蒸馏 main 单流(同 train_distill_stream), **但对难对区样本加权 ×(2~4)**。
  - 难对区 = HARD 5 对(13,12)(22,21)(8,10)(18,17)(26,24)
  - 或更细: 按 main per-pair 置信差(conflates) 动态加权
- 与已试 A 的唯一区别 = **难对 reweight**。这正是 LUPI "修正函数"精神:
  弱模态不参与推理, 只告诉我们"哪些样本难、该多花梯度"。
- 判据: fold0 ≥0.6695 且难对区 >50.96%(main16f 基线难对区)。

### 🥈 C. 硬对比扩充(对难对) —— 样本级, 不动架构
- 难对区样本做 **mixup/cutmix within-pair**(13↔12 互换正文), 让 main 见到的
  "类间像素混合"增多; 纯训练数据增强, 无新模态、无推理开销。
- 风险: 与 main 训练侧"加东西防负"冲突(历史 -2~-7), 排 B 后。

### D. Hint/中间特征对齐、cross-modal contrastive —— 排后
- FitNets hint=训练侧加投影+loss, 历史训练侧改动全负 → 排后。
- 对比 InfoNCE(视频↔骨架) 需负样本、主任务梯度稀释, 且骨架 test 域偏移 → 排后。

## 3. 立即可落地的最小实验（建议先测这一个）
**A 的 fold0 快速版**: 
- 已有资产: main_oof.pkl / thermal_oof.pkl(3折) / HARD 5 对。
- 训练 main16f 单流, 同 IG65M 初始化, 60ep:
  `Loss = CE(y) + λ · w_i · KL(softmax(teacher_i/τ), softmax(logit_i/τ))`
  - teacher_i = 0.5·softmax(main_oof)+0.5·softmax(thermal_oof)（train 折 → 无泄漏）
  - w_i = 1 全样本 或 难对区 ×3（对照看难对聚焦是否必要）
  - λ=0.5~1.0, τ=3
- 判据: fold0 acc ≥ 0.6695（不伤）且难对区 > 50.96%（有增益才算"辅助成功"）。
- 若 fold0 正 → 这才是 LUPI 的正统落地(teacher 强 + 无新输入 + 同域自蒸馏)。

## 4. 结论
- 调研的诚实结论: **"辅助主模态训练最好"不是把弱模态塞进 main, 更不是拿弱模态当
  teacher/aux**(都≤0.53, 判负); 也不是"强教师(main+th)蒸馏 main"(已试=train_distill_stream, 持平)。
- 弱模态(骨架 BiGRU 难对 0.585 < main 0.662) 不够格当 teacher, 只能当
  "难对样本选择器"(告诉 model 哪对/哪些样本难), 不能当标签来源/特征来源。
- **剩余最有希望**: B = 难对区 reweight 的蒸馏/CE(纯 loss 侧, 不动架构、无推理开销,
  LUPI"修正函数"精神)。它唯一不同于已试 A 的点正是"难对加权"——历史上没人做过。
- 落地优先级: **B(难对 reweight 蒸馏/CE)** > C(难对 mixup 增强) > D(Hint/对比)。
- 若 B 的 fold0 也负(难对区 ≤50.96% 或整体 <0.6695) → 骨架/IMU 辅助训练全链关闭,
  维持 0.75124 锚收官。
