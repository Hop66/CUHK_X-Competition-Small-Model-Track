# SUBMIT 清单 — CUHK-X 冲刺 0.8（2026-09-06 更新#5 · 新锚 0.75!)

## 新基准(0.75 链 = 最优可提交)
- **`sub_chain_nf32_flip.csv` = main_s42(16f) + th_nf32full(32f)** → **LB 0.75** ✅（+1.9pt vs 旧链 0.73134）
- fold +0.55 → LB +1.9: thermal 帧数红利实锤; fold→LB 转移成立
- 这是**新锚**; 旧链(0.73134)是回滚基准

## 待 LB 候选(09-06)
| 优先级 | 文件 | 依据/风险 |
|---|---|---|
| 1(推荐) | `sub_chain_nf32_flip.csv` | **已验 0.75**(锚), 可重复提交校准 |
| 2 | 未来: th nf48 / main nf32(完整训) / 集成变体 | 基于帧数红利继续扩展 |
| 3 | SWA / 旧链 | 78MB 温和回退 |
| - | mega/H/J/Kint5/int4 系列 | ⛔ 超限或 int4 掉点 |
| - | HF 门控数据集(Kevin-Pal/CUHK-X_Small_Model) | ⛔ 疑似测试标签/泄露, 违规风险, 勿用 |

## ⚠️ 100MB pack 红线(09-05 核实, 全候选过筛; 每 int5=39MB / int4≈31MB)
- **关键实测(同 fold0 模型)**: int5=0.6685 / **int4=0.6372(掉 3.1pt)** / int6=0.6674(≈int5)
  → **int4 不可用; int5 是量化甜点**; A 形态(2main1th)必须 int4=97MB 但掉点致命 → **撤销**
- **可提交(≤100MB)只有 2×int5(78MB)形态**:
  | 文件 | 成员 | 大小 | 状态 |
  |---|---|---|---|
  | sub_B_mains42_ths42 | main_s42+th_s42 int5 | 78MB | ✅ LB **0.73134**(锚) |
  | sub_B_mainavg_thavg 系 | 2×int5 SWA | 78MB | ✅ |
  | sub_E_main/thermal_bn | BN-TTA | 78MB | ✅ 弱 |
  | ~~sub_K_2m1th_int4~~ | 3×int4 | 97MB | ❌ **int4 掉 3.1pt, 撤销** |
- **超限不可提交(⛔)**: sub_I_mega3/megamax(6/9成员), sub_J_mega4/8, sub_H_ds(4), sub_K_3m1th/4m1th, sub_C2_*
- **推论**: 多成员路线在 100MB 内**短路** → 唯一杠杆 = 提升单成员质量(训练配方 aug4/nf24, 伪标签) 

## 待 LB 候选(09-06 更新)
| 优先级 | 文件 | 依据/风险 |
|---|---|---|
| **1(推荐)** | `sub_chain_nf32_flip.csv` | **新链 = main_s42(16f) + th_nf32full(32f)**; fold +0.55pt 已验证; vs旧链改动 57/405(较大但 fold 支持) → 用 1 次 LB 检验 fold→LB 转移 |
| 2 | `sub_B_mains42_ths42`(旧链) | 0.73134 锚, 回滚基准 |
| 3 | SWA 双成员(re-提交) | 78MB 温和 |
| - | mega/H/J/Kint5/int4 系列 | ⛔ 超限或 int4 掉点, 永不提交 |

## 已定结论(09-06 增补)
- **thermal 帧数是正杠杆**: th_nf32 fold 0.6478(+5.1pt vs 16f), main+th_nf32 fold +0.55
  (同帧数不同 seed = -1.41 负; 不同帧数=时间尺度多样性 = 正!)→ thermal 升级为 nf32
- GRU+Attn 判负(0.3575); IMU(0.33-0.39)/Radar(47%空) 证伪
- **官方允许外部数据**(公开+披露); NTU RGB+D 骨架 35G 已备(两阶段管线, 待跑)
- 主链不再动结构; 增长靠帧数多样性/外部数据/伪标签(已知低频)

## 判定链(已全 fold 关)
cb_loss❌ / gray_norm❌ / BiGRU❌(0.339) / TENT❌ / MLP&logistic stacking❌
→ 主链不再动结构；增长只信 flip-only 概率平均的"成员多样性"

## 最新判读(09-05 晚)
- fold 证据: 2main+1th=+0.83✓, 1main+2th=-1.41❌, 2main+2th=+1.43✓(不一致,慎),
  transductive-soft 两折矛盾(证据不足;fold2 数据点 54360 出结果中)
- 换框架线全关: dual_fusion(0.6114)❌ / VideoMAE-S(0.31)❌ / 骨架❌ → 唯一大杠杆=成员正交多样性(帧数/增强/seed/范式)+训练配方(aug4/nf24, 54387 AB 中)
- 已产: sub_K_2m1th_int4_flip.csv(97MB, 唯一新形态探针) / K 系列 int5(超限仅参考)

## th nf32 3折验证 ✅ 完成(09-06 18:16)
- **mean = 0.6311 ± 0.0255**; fold0=0.6478 / fold1=0.5951 / fold2=0.6505
- 判读: fold0/fold2 双高, fold1 低点(subject 划分难) → 单折波动真实, 3折才是 unbiased
- **th_nf32 full 真实水平 ≈0.631(±2.6pt)**; LB 0.75 锚不受影响(已 LB 实锤链整体)
- 决策规则(已生效): nf48 fold0 与 0.6478 同折比(>则升, 否则帧数到顶); main nf32 fold0 与 0.6695 同折比
- **55719(th nf48 fold0)已自动接力 RUNNING(18:16)**

## 09-07 骨架三路链 LB 判负 → 最终链
- **sub_chain_skel.csv(main16+th32+skel α0.075) = LB 0.74626** < 0.75 锚(-0.38) → ❌ 关闭, 勿再投
- **0.75 锚(sub_chain_nf32_flip.csv = main_s42(16f)+th_nf32full(32f))** = 唯一推荐主链(已两次 LB 实证 0.75)
- 全骨架线(motion dual / MotionBERT 独立成员 / KD)全部关闭; 帧数红利 thermal 32f 封顶; main 保持 16f
- **提交纪律(教训)**: 折叠 <1pt 温和增益跨被试 LOSO 转移概率极低; 只信「fold 量级≥1pt + 多折同向」才花名额
