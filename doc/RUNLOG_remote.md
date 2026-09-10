# RUNLOG_remote.md — 重建版（⚠️ 原 95KB 于 09-09 22:24 被误覆盖，本文件为从会话记忆/片段/相关文档重建的精华版，非全量）

> 恢复说明: 完整历史实验日志不慎丢失(仅保留本精华 + memory + TODO/HANDOFF/SUBMIT)。关键判决与数字已尽量恢复。

## 锚 / 提交
- **主锚 (实证最优可提交)**: `sub_chain_nf32_flip.csv` = main_s42(16f) + th_nf32full(32f), prob_avg 0.5/0.5 + flip → **LB 0.75124**
- main_s42 单模态 LB 0.70646; main+th_s42(16f) 基座 0.73134
- 100MB 硬上限: int5≈38.45MB → 2×int5=77MB 极限; int4 掉~3pt/成员; int6=46MB(1+1=84.5 可行)

## 判决汇总（数据说话）
### 骨架/IMU 弱模态 → 反复判负（09-07 起全链关闭）
- 骨架三路链(main16+th32+skel α0.075) LB = 0.74626 (-0.38 vs 锚) → 骨架线正式关闭
- IMU 双流 sub_chain_imu2 / imu_pseudo / hardpair_gate → 均 LB 0.74626 (-0.5)
- 难对推理 override(手工特征换头) fold +1.7~2.6 → LB -0.5
### 训练侧配方全负（main/thermal 增强）
- main: CB -3.0 / aug4 -1.72 / nf24≈0 / MixUp+EMA / EMA / RSC -7 / 掩码 -3 / 骨架heatmap输入 -3.1 / 亮度 -2.8
- thermal: nf48 到顶(32f=0.6478 最佳) / EMA-4.5 / Erase-4.3 / freeze-3.5 / MixUp-2.7 / CutMix-5.0 / RSC-7.2
- → **训练配方已达瓶颈, 只信: 推理侧融合(prob_avg+flip) + thermal 帧数(nf32)**
### 框架/融合架构负
- 端到端特征双流(dual_fusion) fold0=0.6114 / VideoMAE / BiGRU(0.339) / GRU+Attn(0.3575) / dual(SM) LB 0.637
### TTA 负
- 时间偏移 TTA4 -4~-6pt / 空间多裁剪 -4~-6pt / prior 调整 0.43 → 主链维持 flip-only

## 2026-09-09 骨架/IMU 训练期辅助「收官判决」(本会话核心)
| 实验 | 机制 | fold0 | 判定 |
|---|---|---|---|
| L2 aux-skel | main encoder→aux 预测骨架 51 维 | 0.6448 | -2.5 ❌ |
| L2 aux-imu (58740) | main aux 监督 IMU | 0.6340 | -3.55 ❌ |
| M1 骨架heatmap输入(7ch) | 输入通道 | 0.6383 | -3.1 ❌ |
| M7 亮度匹配 test | 输入亮度 | 0.6416 | -2.8 ❌ |
| gate-skel (58742) | main+骨架 BiGRU 特征门控 | 0.6566 | **-1.29 ❌** |
| bigate 三路 (58743) | main+骨架+IMU 门控 | 0.6674 | **-0.21 ❌** |
| aux-hardpairs (58821) | 难对 GBDT 软目标 aux | 0.6394 | **-3.0 ❌** (难对区46.09%≈0增益) |

→ **骨架/IMU 训练期辅助全链关闭（5 连判负）**。永久结论: 弱模态(骨架 BiGRU 难对 0.585 < main 0.662) 不配当 teacher/监督/特征源, 只能当"难对样本选择器"。

## 2026-09-09 骨架自身 acc 探针（CPU, 数据证实）
- v3(13d)=0.184 / IMU(43d)=0.103 / v3+IMU(56d)=0.199 GBDT 40类LOSO ≪ BiGRU raw[T,17,6] 0.5318
- → 手工特征只对难对二分类有效(62-65%); 骨架自身 acc 靠神经时序
- 难对判别力: biGRU 0.585 < main 0.662 <per-pair 弱点 (26,24)0.39

## 0.8+ 冲刺(09-09, 用户要求"都做一个个来")
- **D4 推理侧 geomean**: OOF 3折 arith0.6777→geomean0.6819(+0.42 3折正), fold0 +0.77; test 翻转25/405(高置信主导, 行为与fold低置信不一致→跨域漂移风险); arith T3 翻转10/405. → **高风险需用户决策是否 LB 实测**
- **D2 SWA**: main_avg(3seed 参数平均 seed42+2024+777)38.45MB 已打包; 雷=重TTA收缩30类, 轻flip未测
- **D1 主链 NoisyStudent**: 教师 test_teacher_avg_probs.pkl(conf0.65,0.9仅11%) 就绪; 期望有限(同 IMU伪-0.5), 排后
- **D3 第3 int8 成员**: GRU 已负; 需新骨干, 排后

## 其他保持
- 域根因: IMU test 角速度=train×1.63、时长-40%; Depth test 更暗12%; 输入校准无效
- 同模态 KD(train_distill_stream --modality main) 与 seed42 持平; transductive-soft fold 矛盾关闭
- fold→LB 转移≈+8.8pt(锚 fold0 0.6630 → LB 0.75124); 温和 fold 正不一定转移(骨架+1.65→LB-0.38, IMU fold+1.72→LB-0.5)

## 2026-09-09 冲刺执行(用户: 冲到0.8, 骨架必须用, 不回避)
- **难对重权训练(59057 RUNNING)**: main nf16 fold0 + 难对区 CE 梯度加权×2(骨架难对选择器, LUPI修正函数; HARD 5 对=13/12,22/21,8/10,18/17,26/24) → train_step1.py 加 --hardpair_w(per-sample weighted CE, 冒烟通过)
  - 判据: fold0 ≥0.6695 且难对区 >50.96% (main16f 基线难对区)
  - 理由: 骨架唯一强项=难对判别(手工特征62-65%>main), 纯 loss 侧不动输入/架构 → 区别于已负的输入/监督/融合三条路
- **D4 geomean(59027)**: 复现锚✓flips0 + geomean25flips(高置信主导, fold低置信→test高置信漂移=高风险) + arith T3 10flips(温和) → 不主动烧名额
- **D2 SWA(59040)**: main_avg+th_nf32full=flips84(39类不收缩✓) / 双SWA收缩33类(雷) → main_avg单替换候选但无fold证据(full SWA)
