# CUHK-X 多模态小型 HAR —— 方法论与实验总账 (2026-09-11)

> 任务: 40 类 Accuracy@1，LOSO（train: user 1-9/16-24，test: user 10-11/25-26），模型 ≤100MB，禁止大预训练骨干。
> 主链锚 LB = 0.75124（main_s42 + th_nf32full, prob_avg + flip_tta）；当前最优提交 = 0.75621（骨架 gate）。
> 本文件盘点所有已试方法，标注其数据、可靠性与证伪状态。**可靠性分级**: ✅LB 正 / ⚠️fold 正但未 LB / ❌已证伪 / 🔬未完全证伪（保留可能性）。

> ## ⚠️⚠️ 2026-09-11 实验体系污染声明（重要, 见 doc/idea.md）
> 下列历史结论**有效性降级**，原因是实验体系存在协议污染（独立审查 `idea.md` 确认）:
> | 问题 | 影响 | 状态 |
> |---|---|---|
> | **B. full 模型被当 fold0 OOF** (`check_th_main_inprotocol`, main_full 全量训练却在训练集 fold0 上评"OOF") | 该实验泄漏, 结论作废 | ❌ 判废 |
> | **C. gate fold 二分类器 a-vs-其余 ≠ test 的 a-vs-b** (`gate_fold_check_skel_only`) | 折叠验证与 test 机制不一致 + selection-bias | 🔧 已修过滤 + 标注 |
> | **D. double-softmax** (`hardpair_dual_fold`) | 折叠融合用了 softmax(softmax(logits)), 与提交协议不符 | 🔧 已修 exactly-once |
> | **E. +5.18 的 baseline 是坏融合**(aug2 融合 < 单 main) | 增益虚高, 真实约 +2pt 且在 wrong-host(full 不匹配难对表) 上转移失败(LB-2.49) | ❌ 不采信 |
> | **F. 512d vs 40d 不同协议**(hook 只抓 flip 一遍) | "512d 无可利用信息"结论撤回 | 🔧 已修两遍平均 |
> | **G. GRL 未可信验证**(monkey-patch 无效 + λ²) | GRL 方向从未有干净实验 | 🔧 已修(包装+去λ²) |
>
> **现状**: LB 0.75124 / 0.75621 作为"真实提交分数"保留; 但所有"方法增益/永久证伪"结论需在新协议下重验。
> **新协议基建已就绪**: `outputs/locked_split.json`(locked 4人: Env-A user4/7 + Env-B user18/23) + `scripts/fair_eval.py` + `scripts/make_locked_split.py`。

---

## 0. 一句话总结当前状态

- **推理侧 post-hoc 修正（偏移/重学/特征/logits）在 LOSO 下系统性无利可图**——原生 softmax 头已用尽可迁移信息（512d 特征跨折可分性 0.473 < 40d-softmax 0.717 为决定性证据）。
- **唯一被 LB 证实的方法 = 骨架难对门控（sub_hp_gate_skel, +0.5pt）**："约束难对内 + 高置信 + 真判别器"是唯一可迁移形态。
- 剩余全部"大突破"候选（残差偏移/LogitAdj/NCM/LR/512d-LDA/神经 gate）均已无泄漏证伪。
- **🚀 09-11 新增: 双塔同步难对训练（训练侧）大突破** —— 见 §12：fold0 融合 +5.18pt（难对区 -24 错 / 非难对 -23 错），th 难对重权 +1.76pt 是主引擎，之前全负根因="只改 main 单塔 + 旧噪声难对表 + 单模评估"。

---

## 1. 主链与基线（保留，不可动）

| 项目 | 配置 | 数据 |
|---|---|---|
| main 主线 | R2+1D-34，IG65M 预训练，Depth+IR 4ch，16f，seed42 | fold0 val ≈ 0.663→0.6695 |
| thermal | R2+1D-34 3ch，**32 帧**（nf32full） | fold0 ≈ 0.62-0.64 |
| 融合 | `prob_avg`（softmax 后加权）+ `flip_tta`（时间翻转） | **LB = 0.75124** |

关键事实:
- thermal 帧数 16→32 是少数正方向（nf24 负、nf32 正、nf48 未定→见 f 表）。
- 锚模型 `main_s42_fold0_int5.pth` 是**单 seed 单折**（无 3 折 OOF）——与 fold 实验用的 `baseline_aug2` 3 折**不同源**，是所有 "fold 正 → LB 不符" 的潜在总根源之一。
- 100MB 预算: int5≈38.5MB；极限 2×int5=77MB；第 3 成员需 int8 级小骨干。

---

## 2. 训练侧配方 —— 全部判负 ❌（30+ 连负）

| 方法 | fold0 数据 | 结论 |
|---|---|---|
| 掩码 ir_mask | main -3.0 | ❌ |
| RSC (thermal) | -7.2 | ❌ |
| 骨架 heatmap 输入通道 (7ch 3view) | 0.6383 (-3.1) | ❌ |
| 亮度匹配 (M7) | 0.6416 (-2.8) | ❌ |
| MixUp / EMA / CutMix / erase / mask | 全负 | ❌ |
| CB loss (beta 0.9999) | 判负 | ❌（beta 0.99 未试 🔬） |
| 难对重权 CE (59057; hardpair_w) | fold0=0.6631 < 0.6695 (-0.64) | ❌ |
| ArcFace / CenterLoss (未完成训练) | — | 🔬 未完全证伪（但伴随 30+ 负，期望低） |

**判定**: 训练/输入侧配方系统性全负 → 只信推理侧融合 + 帧数。但注意这些都是在"0.66 模型上打补丁"，不是架构级提升（big-base 未真正穷尽，见 §6）。

---

## 3. 弱模态（骨架/IMU/Radar）辅助 —— 全链判负 ❌

### 3.1 推理侧独立成员

| 方法 | fold | LB | 结论 |
|---|---|---|---|
| 弱模态 prob_avg 加权成员 (3 次) | fold 正 | **LB -0.5 ~ -0.38** | ❌ 成员加权作废 |
| 弱模态条件纠错 (孪生对) | ≤52% | — | ❌ 无互补 |

### 3.2 训练期辅助（aux/gate/蒸馏）

| 方法 (job) | fold0 | 结论 |
|---|---|---|
| aux-skel | 0.6448 (-2.5) | ❌ |
| aux-imu (58740) | 0.6340 (-3.55) | ❌ |
| aux-hardpairs (58821) | -3.0 | ❌ |
| gate-skel 特征门控 (58742) | 0.6566 (-1.29) | ❌ |
| bigate 三路 (58743) | 0.6674 (-0.21) | ❌ |
| 强教师蒸馏 main (distill stream/main_ds) | 持平 | ❌ |
| transductive-soft | 判负 | ❌ |
| 伪标签弱模态 | LB -0.5 | ❌ |

**判定**: 骨架/IMU 训练期辅助全链收官（5 连判负）。核心原因（数据）: 弱模态难对判别 BiGRU=0.585 / IMU=0.623 / thermal=0.619，均 < main=0.662 —— 弱模态"学不会" main 已经会的，当不了 teacher。

---

## 4. 🌟 骨架难对门控 —— 唯一 LB 正票 ✅（当前最优)

- **方法**: `main+th 预测落难对 + margin<0.05 + conf∈[0.5,0.85] → 纯骨架 v3 特征 GBDT(13维, 去 IMU 偏移) 换头`
- **产物**: `sub_hp_gate_skel.csv` — 只翻 3 个 (21→22 / 18→17 / 26→24)
- **数据**: fold +0.94pt (std 0.0005) → **LB = 0.75621 (+0.5 over 锚)**
- **per-pair 分解**: (13,12) 净+4 / (22,21) 净+2 / (8,10) -2 / (18,17) -1 / (26,24) -1
- **最优 variant**: `sub_hp_gate_keep.csv`（--pairs_keep "13-12,22-21" 只留好对，flip=1）fold +0.026（3.8×现版）但从未上 LB 🔬（候选）

**可靠性**: ✅ 已 LB 证实一次。⚠️ 后续继续扩大翻转数 → 风险递增（高置信难对翻转 test 行为漂移风险同 D4）。

---

## 5. 推理侧 logits 修正 —— 全部证伪 ❌（本轮核心)

### 5.1 logit 残差偏移（用户方向"融合让特征值偏移"）—— LB 证伪

- **方法**: `fused_logit = main + α·(weak_logit − mean)`, pred 落难对时启用。
- **fold（无泄漏 OOF）**: 热残差 α=0.5 = +1.15pt；+骨架 = +1.36pt；+IMU = +1.51pt（三折全正！看似最强）
- **LB = 0.74129 (-0.995)** ❌
- **真因**: 残差后 `fused.argmax()` 自由落任何类 → 30 flip 中 **28 个"跑偏到非难对类"**（38→4, 7→2, 17→9）。偏置增益是 train fold 偶然对，LOSO test 失效。
- **约束版**（只在难对内部调整）：fold 仅 +0.24pt（fold1 -0.11）→ 难对内部无真增益。
- **可靠性**: ❌ 完全证伪（非约束版跑偏；约束版无增益）。文件 `sub_logit_resid_*.csv` 冗余可删。

### 5.2 Logit Adjustment (Menon ICLR'21)

- **方法**: `adjusted = logits − τ·log(p_y)`，类频率 post-hoc。
- **OOF**: 最优 τ=0.0（=不调整），0 增益 ❌
- **原因**: 任务类别不平**28:1**，但 LOSO test 偏移是"被试级"而非"类频率"，train 先验不代表性。
- **可靠性**: ❌ 方向正确但不适用本任务（若 test 类分布与 train 相同才有效）。

### 5.3 Temperature Scaling (Guo ICML'17)

- 单参数置信校准；改变置信不改变 argmax，对 Accuracy@1 无直接作用。
- **可靠性**: 🔬 未深测；理论上对 Accuracy@1 收益极有限（只在温度非 1 时改最终预测）。

### 5.4 NCM / Logistic Regression 重分类（40d 特征空间）

- 用其他折 OOF logits 训 LR/NCM，测当前折（无泄漏）：
  - LR(40d): fold0=0.573 / 0.520 / 0.549 **大输** softmax(0.66/0.65/0.72)
  - NCM 欧氏(40d): 0.663/0.640/0.697（fold0 略高 +0.5pt，fold1/2 弱）
- **可靠性**: ❌ 40d logits 空间已被 softmax 头用尽，无额外可分信息。

---

## 6. 512d encoder 特征（本轮新提取，决定性证伪）❌

- **动机**: main_oof 只存 40d logits；"信息可能在 head 之前 512d"（few-shot/域迁移标准 NCM/linear-probe）。
- **提取**: 改造 `extract_oof_logits.py`（forward_hook 抓 `model.encoder` 输出，512d）；aug2 3 折 OOF + test → `main_feats512.pkl`。
- **难对跨折可分性**（train fold≠test fold, LDA）:

| | 512d LDA | 40d softmax |
|---|---|---|
| fold0 | 0.479 | 0.726 |
| fold1 | 0.552 | 0.666 |
| fold2 | 0.388 | 0.757 |
| **均值** | **0.473（<随机 0.5!）** | **0.717** |

- **结论**: softmax 头**没有压扁信息，反而把难对判别力浓缩进 logits**。原始 512d 特征跨被试更漂移（subject-specific 记忆）。
- **可靠性**: ❌ 完全证伪。"信息在 512d"前提错误。

---

## 7. 20% 无法分辨样本的本质（数据解剖）

- 总 894 错 (fold acc 0.679) = ①难对内 219 (25%) ②跨难对 256 (29%) ③**非难对间 419 (47%!)**
- 高置信错 (conf>0.8 仍错) = 216
- 难对样本在锚上绝对 acc = 0.454/0.438/0.558 ≪ 总 acc 0.64/0.62/0.68（**难对=最难20%**）但二分类 40d-softmax=0.717（判别力已在头内）
- 难对内部错**不到彼此**，而错到第三类（→ "难对门控"只能救 25% 内的一部）
- **LOSO 本质**: 难对模式 subject-specific，跨被试不迁移（跨折 LDA 0.473≈随机）。

---

## 8. 方法可靠性总表

| 方法 | fold | LB | 状态 |
|---|---|---|---|
| main+th prob_avg+flip（锚） | 0.663→0.6695 | 0.75124 | ✅ 主链基线 |
| **骨架难对门控 (skel gate)** | +0.94pt | **0.75621** | ✅ **正票** |
| 骨架 gate 只留好对 (keep) | +0.026 | 未测 | 🔬 候选 |
| geomean / 温度 (D4) | +0.42 (3折正) | 未提交 | ⚠️ 高置信翻转漂移风险 |
| SWA 单成员化 (main_avg) | 无 fold 证据 | 未测 | 🔬 高风险 |
| NoisyStudent 主链 | — | — | 🔬 期望低 |
| logit 残差偏移 | +1.15pt (幻觉) | 0.74129 | ❌ 证伪 |
| Logit Adjustment | 0 | — | ❌ 不适用 |
| LR / NCM on 40d | ≤softmax | — | ❌ 证伪 |
| 512d 特征 LDA/probe | 0.473 | — | ❌ 证伪 |
| 神经 BiGRU gate | -0.010~-0.016 | — | ❌ 证伪 |
| 训练侧配方 / 弱模态 aux | 全负 | 全负 | ❌ 收官 |
| IMU 积分姿态角 (6,37)=0.832 | 判别强 | 未集成 | 🔬 未完全证伪（判别强但 40 类太弱 0.34） |

---

## 9. 未完全证伪（保留可能性的死角）🔬

1. **骨架 gate 加强（keep 版）**: fold +0.026 (3.8×现版)，只 flip 1，从未上 LB —— 低风险正向候选。
2. **IMU 姿态积分特征进判别器**: 难对二分类 (6,37)=0.832（全模态该对最强）、(7,19)=0.712、(8,10)=0.687 —— 可作为难对判别器的强候选，但 IMU 40 类 acc 仅 0.34，需"约束难对内 + 真判别器"形态才可能有效。
3. **bigger-base（架构级）**: 所有 30+ 负都是"在 0.66 模型上打补丁"；真正更大的骨干/更长训练未穷尽（但 100MB 约束卡死 int4-2×int5）。
4. **CB loss beta=0.99**（未试的尾巴）。
5. **SWA main_avg 轻 flip**（免预算，but 高收缩雷）。

> **09-10 更新 —— 方向 A 已裁决（🔬→❌）**: 用户文献调研建议"IMU 姿态积分进难对判别器"。实测（无泄漏跨折）:
> - 全难对 fold +0.0032（触发 6/4/2 太少）；好对版 +0.004±0.012（fold1 转负）
> - 跨折难对二分类 GBDT: **skel+IMU(60d pose agg) = 0.574/0.538/0.636 vs 纯 skel = 0.584/0.543/0.621 → IMU 姿态聚合无稳定增益**
> - 与判别 bench (6,37)=0.832 矛盾解释: bench 是 per-pair CNN 非跨折；跨被试下 IMU 姿态判别力同样衰减（LOSO 不迁移的又一体现）
> - **方向 A 关闭，仅剩骨架 keep 版 (B) 值得 LB 试水**。

---

## 10. 硬教训（防再犯）

1. **任何 OOF 用前必须先查单模 acc**，>0.9 即泄漏（skel_full=0.98 假象）。
2. **"从 OOF 学模式 + val gate"式方法全部关闭**（main LDA +0.032 泄漏 / 无泄漏 -0.018）。
3. **残差式 logits 融合必须约束在难对内**，禁止自由 argmax（跑偏到无关类）。
4. **fold 与 LB 行为可完全相反**（D4/ds-likelihood 教训）；fold 正 ≠ LB 正。
5. **IMU 列名 bug**: test down 英文列名曾让 test 只剩 3 设备——数据管线 bug 是"假域差"根因。
6. **同源问题**: fold 实验模型 (baseline_aug2) ≠ 锚 (main_s42)；慎把 fold 结论外推到锚。
7. OOF 是 per-fold val 折（fold f 的 OOF 即 fold f 的 val）；"用 train 折 OOF"天然 n=0，需用其他折。
8. 文档先 git add 再改（RUNLOG 曾被覆盖丢失）。

---

## 12. 🚀 双塔同步难对训练 —— 训练侧大突破（09-11）

### 12.1 一句话
**训练侧难对重权翻盘**: main+th 双塔**同步**做难对重权 → fold0 融合 **+5.18pt**（0.6355→0.6872），难对区错 -24 / 非难对错 -23。此前所有训练侧难对实验全负的根因 = **只改 main 单塔 + 旧噪声难对表 + 单模评估**。

### 12.2 关键数据（fold0, baseline_aug2 双塔对照）

| 方案 | 融合 acc | 难对区错 | 非难对错 | 单模 main | 单模 th |
|---|---|---|---|---|---|
| 基线双塔 | 0.6355 | 165 | 166 | 0.6674 | 0.6068 |
| **难对双塔** | **0.6872** | **141** | **143** | 0.6542 | **0.6244** |

- **th 塔 +1.76pt 是主引擎**（th 从未做过难对训练，难对重权对它是纯新正信号）
- main 塔 -1.32pt（可接受，融合完全吸收）

### 12.3 为什么以前全负、现在翻盘？（三重遗漏修正）

| 对比 | 历史 (58821/59057) | 现在 (59788) |
|---|---|---|
| **难对表** | 旧 5 弱对（含 3 对纯噪声 12↔13/22↔21/26↔24） | 新 11 对可学难对（判力矩阵 0.65-0.85 精选） |
| **塔** | 只 main 单塔 | **main+th 双塔同步**（同一 HARD_PAIRS 模块级常量天然同步） |
| **评估** | 只看 main 单模 | 看 **main+th 融合**（锚真实形态） |

**难对表定案（train_step1.py HARD_PAIRS, 11 对）**:
`(0,1)(6,37)(7,37)(8,9)(8,10)(8,15)(8,18)(11,14)(17,18)(18,20)(38,39)` — 类簇16/40, 难对样本45%
- 来源: OOF 训练集逐 clip 报错（总≥6次且≥2折跨折稳定）, 100% 训练集带标签（非测试推测）
- 排除已会 (2c>0.85) + 纯噪声 (2c<0.65≤抛硬币: 12↔13/21↔22/26↔24 数据不可分)
- 判力矩阵: main 几乎全面 ≥ 骨架/IMU → 弱模态无独立信息可借, 纯靠 main 自身提高难对区判别

### 12.4 训练配方
- `hardpair_w=2.0`: 难对区样本 CE 梯度 ×2（per-sample, 非难对保持1）
- full(seed42): main 16f/80ep + th 32f/80ep（保留帧数红利）; 快照按 ep 保留（防过拟合选点）
- int5 打包: hpdual_main_fold0_int5 40.2MB + hpdual_th_fold0_int5 40.2MB（共80.4MB ≤100MB 合规）

### 12.5 full 组链（60340）
- `sub_hpdual_full_flip.csv`: **flips=82/405 (20%), 去重类=40(不收缩)**
- 难对相关类改动 65/82 (79%) 专心改难对 ✓ 方向正确
- ⚠️ 但 82 flips 偏大 — 高置信区行为漂移风险（同 D4/残差教训）, 需锚同源仔细核对再定是否替代 0.75 锚

### 12.6 衍生方向（09-11 已实现, 待 GPU 验证）
- **方向1 难对区 Focal**: `--hp_focal_gamma` (train_step1) — 难对类 (1-p)^γ 逐样本加权, 非难对保持1, 自洽不漂移（修 1.59× loss 尺度漂移）
- **方向2 难对区定向融合**: `hpzone_fusion.py` — fold 难对区 main 单(0.589) > 融合0.5(0.560) > th(0.515), 难对区应抬高 main 权重(α≥0.6); 保守 α_hp=0.7, 需锚同源 test 验证（最优 α 折间不一致: f0 0.8/f1 0.6/f2 0.6）
- **方向3 骨架→spatial attention**: `_skeleton_heatmap` [T,3,S,S] 可作乘法门控（比 M1 concat 更软), 小数据高风险, 等难对定案后做

### 12.7 纪律（09-11 新增）
- 验证推理必须与基线同口径（flip 对齐）— 曾因 infer 无 flip 造成 +5.18 疑点
- fold(aug2) α 最优 ≠ test(s42) 最优（铁律）; α 扫描只作诊断
- 登录节点 tradmin-02 无 GPU! 前台跑 ensemble 是 CPU 慢卡死 → 必须 sbatch 到 GPU 节点

---

## 11. 清理备注

- 已证伪脚本/产物（logit 残差、feature-resid、无效融合）标记为冗余可删（见删除清单）。
- 待保留: `sub_hp_gate_skel.csv`(最优提交)、`sub_hp_gate_keep.csv`(候选)、`ensemble_inference.py`(推理链)、`src/`(全部)、`extract_oof_logits.py`/`make_test_gate_csv.py`(可复用工具)、`main_feats512.pkl`(虽证伪但重提取贵，留)。
- 幂等: 所有探索脚本可 git 追溯（仓库已 init，勿 cat> 覆盖）。
- **09-11 清理**: 
  - 归档 `scripts/archive_judged/`（判负脚本: gate_fold_imu_pose / fold_int / train_contrastive / twin_v3 / skel_twin_correct / twin_audit_overlay / e2e_overlay / gate_imu.sbatch）— 保留作为"为什么负"的方法学资产
  - 删 `dual_hp_full/*_ep80.pth`（=裸名 dup, md5 相同, 省 508MB）；保留 ep10-70 快照供选点
  - 删空目录 `hand_roi` / `thermal_rebuild`
  - 保留: `dual_hp_f0`(flip复核输入) + `dual_hp_full` ep10-70 + hpdual int5 打包 + `sub_hpdual_full_flip.csv`
