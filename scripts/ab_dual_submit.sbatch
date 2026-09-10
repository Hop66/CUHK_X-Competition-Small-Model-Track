#!/bin/bash
# ============================================================
# CUHK-X —— SM dual 提交 0.56 分层 A/B（只推理，不训练）
# 目的：一刀切定位根因
#   A) ab_main_crop0   : 0.73 基线复现（main_s42+thermal，无 thermal_crop，
#                         与当年 0.73 链命令一致）→ 验证共享代码未回归
#   B) ab_main_crop1   : 同 A 但 thermal 带 bbox_thermal_train.json crop
#                        → 隔离 thermal_crop 因素（我新链加了 crop，旧链没有）
#   C) ab_dual_crop0   : dual(learned α)+thermal 无 crop → 复现/对照 0.56
#   D) ab_dual_crop1   : dual(learned α)+thermal + crop（复现 0.56 参考）
#   E) ab_dual_static0 : dual 强制纯 static(α=1)+thermal 无 crop
#                        → static 分支单独水平（若 ~0.7+ → motion 是元凶；
#                          若 ~0.56 → static 分支本身/联合训练问题）
#   F) ab_dual_half0   : dual 强制 α=0.5+thermal 无 crop（中间档）
# 判定: 交 outputs/ab/*.csv 逐个看 LB。
# 用法: sbatch scripts/ab_dual_submit.sbatch
# ============================================================
#SBATCH -J cuhkx-ab-dual
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/ab
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

MAIN=outputs/pack/main_s42_fold0_int5.pth       # 0.73 纯 main（int5）
DUAL=outputs/pack/main_dual_s42_fold0_int5.pth  # SM dual（int5）
TH=outputs/pack/thermal_s42_fold0_int5.pth      # thermal（int5）
OUT=outputs/ab
CROP="--main_crop bbox_train.json"
THCROP="--thermal_crop bbox_thermal_train.json"

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$DUAL" ] || { echo "❌ 缺 $DUAL"; exit 1; }

if [ -f "$MAIN" ]; then
  echo "==== [A] 0.73 基线复现：main+thermal 无 crop ===="
  python scripts/ensemble_inference.py --prob_avg --flip_tta \
    --main "$MAIN" --quantize --thermal "$TH" --quantize $CROP \
    --output "$OUT/ab_main_crop0.csv" || echo "⚠️ [A] 失败"
  echo "==== [B] main+thermal 带 thermal crop（隔离 crop 因素）===="
  python scripts/ensemble_inference.py --prob_avg --flip_tta \
    --main "$MAIN" --quantize --thermal "$TH" --quantize $CROP $THCROP \
    --output "$OUT/ab_main_crop1.csv" || echo "⚠️ [B] 失败"
else
  echo "⚠️ 缺 $MAIN → 跳过 [A][B]（0.73 链 ckpt 须先 pack）"
fi

echo "==== [C] dual(learned)+thermal 无 crop ===="
python scripts/ensemble_inference.py --prob_avg --flip_tta \
  --main_dual "$DUAL" --quantize --w_main_dual 1.0 \
  --thermal "$TH" --quantize --w_thermal 1.0 $CROP \
  --output "$OUT/ab_dual_crop0.csv" || echo "⚠️ [C] 失败"

echo "==== [D] dual(learned)+thermal + crop（复现 0.56 参考）===="
python scripts/ensemble_inference.py --prob_avg --flip_tta \
  --main_dual "$DUAL" --quantize --w_main_dual 1.0 \
  --thermal "$TH" --quantize --w_thermal 1.0 $CROP $THCROP \
  --output "$OUT/ab_dual_crop1.csv" || echo "⚠️ [D] 失败"

echo "==== [E] dual 强制纯 static(α=1)+thermal 无 crop ===="
python scripts/ensemble_inference.py --prob_avg --flip_tta --dual_alpha 1.0 \
  --main_dual "$DUAL" --quantize --w_main_dual 1.0 \
  --thermal "$TH" --quantize --w_thermal 1.0 $CROP \
  --output "$OUT/ab_dual_static0.csv" || echo "⚠️ [E] 失败"

echo "==== [F] dual 强制 α=0.5+thermal 无 crop ===="
python scripts/ensemble_inference.py --prob_avg --flip_tta --dual_alpha 0.5 \
  --main_dual "$DUAL" --quantize --w_main_dual 1.0 \
  --thermal "$TH" --quantize --w_thermal 1.0 $CROP \
  --output "$OUT/ab_dual_half0.csv" || echo "⚠️ [F] 失败"

echo ""
echo "==== AB DONE — 提交 outputs/ab/*.csv 逐个对拍 ===="
echo "判读表:"
echo "  A≈0.73                → 共享代码未回归（问题只在 dual 侧/crop）"
echo "  A≈0.73 且 B≈0.56      → thermal_crop 不一致是元凶（新链加的 crop 错了）"
echo "  E≈0.73+/A≈0.73        → static 分支好 → motion(learned α) 拖累 → 用 α=1 或重训"
echo "  E≈0.56                → SM 联合训练 static 本身对测试差 → 需对拍/回退纯 main"
echo "  C vs D (0.56)         → 复现确认 crop 因素；C≈0.73 则 crop 因素"
