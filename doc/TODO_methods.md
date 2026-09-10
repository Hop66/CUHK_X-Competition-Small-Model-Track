# 方法论 → 待做清单 (2026-09-08 用户驱动: 骨架/IMU 作为"训练期教练", 推理只靠相机)

原则: 骨架/IMU 不再做推理侧独立成员(fold/LB/部署三重复判负);
      而是**训练期辅助图像模型**(用骨架/IMU 加强图像数据/ROI/关键帧/监督), 部署只依赖相机流。

## 方法论池(按优先)
- [ ] **M1 PoseC3D 骨架热图 early fusion 进 main** (CVPR'22 mmaction2): 3D骨架--bbox近似投影-->每帧1ch heatmap, 并入 main 4ch→5ch; fold0 main16 vs 0.6695. 有跨数据集鲁棒背书(对症域差)
- [ ] **M2 手部/局部 ROI 视图 (hand-crop/HOI 家族)**: skeleton wrist 引导手部放大窗双视角, 补孪生手部对像素判别; fold0 main16
- [ ] **M3 骨架/IMU 动量选关键帧**(=nf32 帧数红利同源升级): 骨盆/手速度峰 or IMU gyro峰值 → 选动作相位高信息帧采样; 长空闲打包, LB 判
- [ ] **M4 Privileged Information / LUPI teacher-student**: 训练时把骨架/IMU 当"特权信息"监督 teacher(如对预测加 auxiliary 或 soft 对齐), 推理只用 main/th 图像(纯相机). 需再搜具体 SOTA(LUPI 视频识别)
- [ ] **M5 ARN/SMART 双分支**(appearance+relation) 显式"体动/局部"分流(可选,工程较大)
- [ ] **M6 域差伪域 CV**: 按日期/批量切伪域(训练 05-07~06-11 多天), 用 leave-batch-out 校准"跨域型"融合因子(替代裸 fold)
- [ ] **M7 main 亮度增强匹配 test(0.524→0.462 域差打根因)**: 训练亮度/对比度 drop, 纯相机; fold0
- [ ] **M8 IMU 域增强重训**(gyro 幅值 ×0.6~1.8 覆盖 1.63× 域差), 校准推理被证无效→改训练侧增强; 出后 LB 判

## 数据层自检结论(每次先跑 scripts/data_structure_probe.py)
- 待更新: test vs train 骨架/IMU 结构差异是否导致"假域差"

## 已关闭(勿再投入)
骨架/IMU 独立推理成员 / 条件换头 overlay / IMU 推理侧幅值校准(0.358→0.333 无效)
掩码 ir_mask(main -3.0) / RSC(thermal -7.2) / 弱模态条件纠错(≤52%) / 伪标签弱模态(LB -0.5)

## 0.8+ 冲刺路径(09-10, 骨架gate突破)
- **🌟 骨架-only gate LB=0.75621 (+0.5 over 锚0.75124) = 骨架参与首次正票** → 难对纠错方向被 LB 证实
  - 方法: main+th预测落难对+margin<0.05+conf[0.5,0.85]→纯骨架v3 GBDT(13维去IMU)换头, 只翻3个
  - 第3成员(多成员加权)已否决: 弱模态加权3次 fold正→LB负(-0.5~-0.38); 100MB余23MB但骨架/IMU均负
- **混淆审计(09-10)**: 发现大量语义外难对(9↔10倒/搅 err22, 8↔9 err21, 6↔37 err19, 6↔7 err16, 17↔18 err14) — 但骨架v3只对(17,18)键盘↔写字有判别(0.603), **该对已在基础5对已覆盖**; 其余因样本少训不了判别器=长尾难对
- **长尾诊断**: 类25看TV(11样本 acc0.00)/类18写字(0.105)/类26游戏(0.10)/类37服药(0.292)/类8餐具(0.298); balanced sampler已有但仍不足; CB loss(beta0.9999)判负, beta0.99未试
- 下一步: ①骨架gate加固(**per-pair开关**: fold净正仅(13,12)+4/(22,21)+2, (8,10)-2/(18,17)-1/(26,24)-1坏对拉后腿→禁坏对只留好对) ②增加GBDT置信门控 ③方向C长尾
已盘点: 训练配方/弱模态/TTA 全员负; 100MB 极限=2×int5(77MB), 第3成员需 int8 级小骨干(GRU 已负)。
| 方向 | 状态 | 证据 |
|---|---|---|
| D4 推理侧 geomean/T | 已实现 --geomean/--temp | OOF 3折 0.6777→0.6819(+0.42, 3折正); fold0 0.6575→0.6652(+0.77); **待 test AB** |
| D2 SWA 单成员化 | main_avg(3seed参数平均) 已打包38.45MB | 雷=重TTA收缩30类; 轻flip未测; 可替换 main_s42 进链 |
| D1 主链 NoisyStudent | 教师 test_teacher_avg_probs.pkl 就绪(conf0.65, 0.9以上仅11%) | 机制同 IMU伪标签(-0.5); main 曾朴素 pseudo -0.4 → 期望有限, 排后 |
| D3 第3 int8 成员 | GRU 负; 需新骨干有 fold 正证据 | 小骨干未做 |

优先级: D4(test 最快零预算) → D2(SWA 免预算) → else 收官 0.75124 转报告。

## 已判定(09-09, 数据证实)
- M1 骨架heatmap输入通道(7ch 3view): fold0=0.6383 (-3.1) ❌
- M7 亮度匹配: fold0=0.6416 (-2.8) ❌
- L2 aux-skel: fold0=0.6448 (-2.5) ❌ / aux-imu: 0.6340 (-3.6) ❌ / aux-hardpairs: 58741(NFP崩)→58821 已修重提, 待 fold
- **骨架手工特征(v3/IMU GBDT) 全40类LOSO极弱**(v3=0.184 / IMU=0.103 / v3+IMU=0.199 ≪ BiGRU raw[T,17,6] 0.5318) → 手工特征只对难对二分类有效(62-65%); 骨架自身acc只能靠神经时序(gate融合用BiGRU=方向正确), 不再试手工特征进40类单模态
- **难对判别力(09-09)**: 神经各成员难对二分类 BiGRU=0.585 / IMU=0.623 / main=0.662 / thermal=0.619 (n=677 全量难对 OOF) → BiGRU 对混淆对无补益(<main); per-pair 弱点: (26,24)0.39
- **骨架特征门控融合(gate-skel 58742) fold0=0.6566 vs 0.6695 = -1.29 判负**; 三路 bigate(58743) 跑中 → BiGRU 门控低整体 + 无难对增益
- 调研结论(见 doc/RESEARCH_aux_training.md): 弱模态只当难对"样本选择器"不当 teacher; 强教师蒸馏 main 已试(train_distill_stream)持平; 剩余唯一未试 = **B 难对区 reweight 蒸馏/CE** (纯 loss 侧, LUPI 修正函数精神)
- 特征门控融合: 58742 gate-skel / 58743 bigate(main+skel+imu) 60ep RUNNING/PENDING, 判据 fold0 vs 0.6695
