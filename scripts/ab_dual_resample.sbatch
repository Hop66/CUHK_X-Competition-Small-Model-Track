#!/bin/bash
# ============================================================
# CUHK-X —— 待办 A 关键裁决:骨架「真·重采样」能否让被判负的 SM-dual 翻盘
#
# 背景(dom 诊断 2026-09-03):
#   测试每帧速度幅值 = 训练 2.09×,但原帧数相近(26.3 vs 29.0)
#   → kp 级真·重采样扫描:speed_scale=1 时,kp 升采样×2.0 → 速度比 1.02 ✓
#      (物理解读:测试每帧跨真实时间 ≈ 训练 2×,要按物理 cadence 对齐需升采样)
#   标量 speed_scale=0.5 已试过(LB 不变) → 本脚本验证"kp 级重采样"是否真不同
#
# 裁决:
#   - 各 cell CSV 与基线(ss0.5)一字不差 → MotionNet 首层 BN 对输入 scale/形态鲁棒
#     → 域对齐不是 dual 崩的开关 → 按 handoff 结论「dual 线彻底关闭」
#   - CSV 有差异 → 真·重采样改变了 motion 通路输出 → 交 LB 裁决是否翻盘(≥0.73 采纳)
#
# 用法: sbatch scripts/ab_dual_resample.sbatch
# ============================================================
#SBATCH -J cuhkx-ab-resample
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TH=outputs/pack/thermal_s42_fold0_int5.pth
FULL_SM=outputs/pack/main_dual_s42_fold0_int5.pth

INF="python scripts/ensemble_inference.py --prob_avg --flip_tta"
CROPS="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"
COMMON="$INF --quantize --w_main_dual 1.0 --thermal $TH --w_thermal 1.0 $CROPS"

# ============ [A] 自验基准(GPU):fold0 SM 应复现 learned=0.6652 ============
echo "==== [A] selftest_main_dual_infer (golden 0.6652) ===="
python scripts/selftest_main_dual_infer.py \
  --ckpt outputs/main_dual/main_SM_fold0.pth --flip_tta || echo "⚠️ selftest 非零退出"

# ============ [B] 全量-SM dual + thermal AB(0.637 线)============
echo "==== [B1] full-SM 基线 ss0.5 (复现旧线 LB≈0.637) ===="
$COMMON --main_dual $FULL_SM --test_speed_scale 0.5 \
  --output outputs/sub_ab_fullSM_base_ss05.csv || echo "⚠️ B1 失败"

echo "==== [B2] full-SM 真·重采样 kp×2.0 (speed_scale=1) ===="
$COMMON --main_dual $FULL_SM --test_speed_scale 1.0 --test_resample 2.0 \
  --output outputs/sub_ab_fullSM_rs20.csv || echo "⚠️ B2 失败"

echo "==== [B3] full-SM 重采样×2.0 + scale 0.479 (双对齐) ===="
$COMMON --main_dual $FULL_SM --test_speed_scale 0.0 --test_resample 2.0 \
  --output outputs/sub_ab_fullSM_rs20_ss0479.csv || echo "⚠️ B3 失败"

echo "==== [B4] full-SM static-only 参照 (α=1.0, 即 R2+1D 单流 ≈ 0.70646 链) ===="
$COMMON --main_dual $FULL_SM --dual_alpha 1.0 --test_speed_scale 1.0 \
  --output outputs/sub_ab_fullSM_static.csv || echo "⚠️ B4 失败"

# ============ [C] 三折 dual + thermal AB(0.62189 线, foldavg 推荐)============
FOLD0=outputs/pack/main_dual_f0_fold0_int5.pth
FOLD1=outputs/pack/main_dual_f1_fold0_int5.pth
FOLD2=outputs/pack/main_dual_f2_fold0_int5.pth
FOLDS="$FOLD0 $FOLD1 $FOLD2"

echo "==== [C1] foldavg 基线 ss0.5 (复现 submission_foldavg, LB≈0.62189) ===="
$COMMON --main_dual $FOLDS --test_speed_scale 0.5 \
  --output outputs/sub_ab_foldavg_base_ss05.csv || echo "⚠️ C1 失败"

echo "==== [C2] foldavg 真·重采样 kp×2.0 ===="
$COMMON --main_dual $FOLDS --test_speed_scale 1.0 --test_resample 2.0 \
  --output outputs/sub_ab_foldavg_rs20.csv || echo "⚠️ C2 失败"

echo "==== [D] CSV 一致性对比(是否一字不差)===="
echo "  full-SM : base vs rs20   -> $(cmp -s outputs/sub_ab_fullSM_base_ss05.csv outputs/sub_ab_fullSM_rs20.csv && echo 相同 || echo 不同)"
echo "  full-SM : base vs rs20_ss(双对齐) -> $(cmp -s outputs/sub_ab_fullSM_base_ss05.csv outputs/sub_ab_fullSM_rs20_ss0479.csv && echo 相同 || echo 不同)"
echo "  foldavg : base vs rs20   -> $(cmp -s outputs/sub_ab_foldavg_base_ss05.csv outputs/sub_ab_foldavg_rs20.csv && echo 相同 || echo 不同)"

echo ""
echo "==== AB DONE ===="
echo "提交候选(待 LB):"
echo "  sub_ab_fullSM_rs20*.csv / sub_ab_foldavg_rs20.csv (真·重采样线)"
echo "  参照: sub_ab_fullSM_static.csv (static 单流+thermal)"
echo "判读:"
echo "  rs20 与 base 一字不差  -> motion 通路对输入域不敏感 -> dual 线关闭"
echo "  rs20 与 base 有差异     -> 交 LB, ≥0.73 采纳重采样线"
