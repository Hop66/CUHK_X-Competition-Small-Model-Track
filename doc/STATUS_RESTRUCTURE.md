# CUHK-X 现状盘点与推倒重做梳理 (2026-09-12)

> 背景: 用户决定暂停实验、审代码、推倒重做(现 repository 冗余混乱, outputs 37G/107 目录)。
> 本文档 = 审代码的地图 + 重构清理清单。所有 job 已停, GPU 空闲。

---

## 1. 当前可信资产 (保留 / 重做时复用)

### 1.1 新协议基建 (✅ 干净, 重构保留)
| 文件 | 用途 |
|---|---|
| `outputs/locked_split.json` | locked 永久审计集 = Env-A user4/7 + Env-B user18/23 (4人) |
| `outputs/dev_folds.json` | dev 14 人环境平衡 3 折 (每折 A/B 各≥2) |
| `scripts/make_locked_split.py` | 生成 locked split |
| `scripts/make_dev_folds.py` | 生成环境平衡 dev_folds |
| `scripts/fair_eval.py` | 统一评估(overall+per-subject+macroF1+worst, locked 优先) |

### 1.2 新可信基线 (P1 baseline, ✅ dev-3折 融合 0.8001)
- `outputs/p1_base/main/r2plus1d34_depthir_fold{0,1,2}.pth`
- `outputs/p1_base/th/r2plus1d34_thermal_fold{0,1,2}.pth`
- `outputs/oof/p1_main_oof.pkl`, `outputs/oof/p1_th_oof.pkl`
- **这是修复评估体系后的新权威参照** (旧污染协议折叠才 0.64)

### 1.3 核心 code (保留, 重构后继续用)
- `src/model.py` (R2+1D34 / TSM-MobileNetV2 / MixStyle 接入)
- `src/dataset.py`, `src/split.py`, `src/quantize.py`
- `scripts/train_step1.py` (支持 --fold_split 环境平衡)

---

## 2. 已确认修复/判负的实验 bug (审代码注意对应文件)

| bug / 结论 | 文件 | 状态 |
|---|---|---|
| double-softmax (评价协议不符) | `scripts/hardpair_dual_fold.py` | ✅ 已修 exactly-once |
| 512d 只抓 flip 一遍 (≠40d) | `scripts/extract_oof_logits.py` | ✅ 已修两遍平均 |
| GRL monkey-patch 无效 + λ² | `scripts/train_grl.py` | ✅ 已修+验证可训练 |
| gate 折叠二分类 a-vs-其余 | `scripts/gate_fold_check_skel_only.py` | ✅ 已修过滤+标注 |
| full 当 fold0 = 泄漏 | `scripts/check_th_main_inprotocol.py` | ❌ 判废标注 |
| gate_fuse best-checkpoint 混搭 | `scripts/train_gate_fuse.py` | ✅ 已修(五组件best深拷贝) |
| class-cond hard-pair correction 净 -9 | (离线) | ❌ 判负 |
| TSM-MobileNetV2 单折 0.45 | (实验) | ⚠️ 轻量骨干单打弱, 仅可作 ensemble |
| condres freeze=1.0 → 0.2372 | `scripts/train_cond_residual.py` | ⚠️ freeze 过头(bug)未判方法 |

---

## 3. outputs/ 混乱清单 (37G / 107 目录) → 清理建议

### 3.1 建议保留 (可信/可复用, 共 ~9G)
- `locked_split.json`, `dev_folds.json`
- `p1_base/` (新基线), `oof/p1_*.pkl`
- `pack/` (int5 打包, 含旧锚但体积大, 待判)
- `cond_resid/` (仅保留最终 ckpt)

### 3.2 建议删除/归档 (旧污染产物, ~28G)
- 旧难对训练: `dual_hp_f0`(0.7G), `dual_hp_full`(3.8G)
- 旧骨干重训: `main_dual`(1.5G), `main_baseline`(1.5G), `main_full`(0.7G),
  `main_34`(0.7G), `d_trans`(1.5G), `th_nf32`(0.7G), `th_folds*`(1.2G),
  `thermal_*`(2.5G), `selftrain*`(1.5G), `aug_search`(1G), `ab_*`(2.5G),
  `misa_dual`(0.5G), `thermal_notebook14`(1G), `main_8ch_aug_2fold`(0.5G) 等
- ⚠️ 因体积大+污染, 建议先 `mv outputs/ ~/outputs_archive/` 整盘移走而非 `rm`
  (审代码期可能还要回看某 dir; 移走最安全, 空间立即释放)

---

## 4. 推倒重做建议 (待你定夺)

### 4.1 代码层重构方向
- **保留**: `src/`(数据/模型/量化核心) + `scripts/train_step1.py` + 新协议 4 件套
- **收敛**: 321 个 scripts 大多是一次性探索, 建议按 `P0修复/实验/判负/工具` 归档
  (已有 `scripts/archive_judged/`)
- **统一入口**: 未来只用 train_step1.py(训练) + fair_eval(评估) + 一套 inference,
  禁散落的一次性调参脚本

### 4.2 结果文件夹重构
```
outputs/
  base/       # 可信基线 (p1_base 迁入)
  oof/        # OOF logits (只保留 dev 协议)
  models/     # 唯一 ckpt 落点 (按 模型/seed 分)
  暂存/archive/  # 需回看但非产出的东西
```
## 5. 审代码入口 (按重要性)
1. `scripts/train_step1.py` (主训练, 含 --fold_split 新协议)
2. `scripts/train_cond_residual.py` (residual fusion, freeze 需审)
3. `scripts/train_grl.py` (GRL 修复后首次可验)
4. `scripts/fair_eval.py` (新评估协议)
5. `src/model.py` (TSM/MixStyle 接入点)

---

> 待用户审计意见 → 本轮 push 方向由用户拍板 (推倒结构 / 具体重构范围 / 删哪些)。
