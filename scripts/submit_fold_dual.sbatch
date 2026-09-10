#!/bin/bash
# ============================================================
# CUHK-X —— fold 三折 dual 提交（替代 full-SM 提交）
# 背景：full SM_full 对 LB 陌生 subject 泛化崩(0.55)；fold 划分版对陌生
#       subject 有 0.6652（已自验复现）。本脚本用 fold0/1/2 划分 dual +
#       thermal_s42 融合出提交：
#         submission_fold{F}.csv       单折提交
#         submission_foldavg.csv       三折 logit 平均（≈全量数据+泛化受控）
# 前置：outputs/main_dual/main_SM_fold{0,1,2}.pth（main_dual_folds 产物）
#       + outputs/pack/thermal_s42_fold0_int5.pth
# 用法: sbatch scripts/submit_fold_dual.sbatch
# ============================================================
#SBATCH -J cuhkx-submit-fold
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/pack
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

TH=outputs/pack/thermal_s42_fold0_int5.pth
[ -f "$TH" ] || { echo "❌ 缺 $TH"; exit 1; }
DMD="${DM_DIR:-outputs/main_dual}"          # fold ckpt 目录（默认 outputs/main_dual）

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 每折：pack int5 → 单独推理（dual fold + thermal_s42 prob_avg + flip）
FOLD_CKPTS=""
for F in 0 1 2; do
  CK="$DMD/main_SM_fold${F}.pth"
  if [ ! -f "$CK" ]; then
    echo "⚠️ 缺 $CK（尝试 S-only）"
    CK="$DMD/main_S_fold${F}.pth"
    [ -f "$CK" ] || { echo "❌ 缺 fold$F dual ckpt → 跳过"; continue; }
  fi
  FOLD_CKPTS="$FOLD_CKPTS $CK"
  P5="outputs/pack/main_dual_f${F}_fold0_int5.pth"
  if [ ! -f "$P5" ]; then
    echo "[fold$F] int5 打包 $CK"
    python scripts/quantize_pack.py --name main_dual_f${F} --checkpoints "$CK" \
      --bits 5 --out_dir outputs/pack || echo "⚠️ fold$F pack 失败"
  fi
  echo "[fold$F] 推理提交 → outputs/submission_fold${F}.csv"
  python scripts/ensemble_inference.py --prob_avg --flip_tta \
    --main_dual "$P5" --quantize --test_speed_scale 0.5 --w_main_dual 1.0 \
    --thermal "$TH" --w_thermal 1.0 \
    --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
    --output outputs/submission_fold${F}.csv || echo "⚠️ fold$F 推理失败"
done

# 三折 logit 平均（一个 --main_dual 传多 ckpt，infer_main_dual 自动 /=n）
echo "=== 三折平均推理 → outputs/submission_foldavg.csv ==="
if [ -n "$FOLD_CKPTS" ]; then
  AVG=()
  for CK in $FOLD_CKPTS; do
    F=$(basename "$CK" | sed -n 's/.*fold\([0-9]\)\.pth/\1/p')
    AVG+=("outputs/pack/main_dual_f${F}_fold0_int5.pth")
  done
  python scripts/ensemble_inference.py --prob_avg --flip_tta \
    --main_dual "${AVG[@]}" --quantize --dual_alpha -1 --test_speed_scale 0.5 --w_main_dual 1.0 \
    --thermal "$TH" --w_thermal 1.0 \
    --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
    --output outputs/submission_foldavg.csv || echo "⚠️ 三折平均失败"
else
  echo "⚠️ 无 fold ckpt → 三折平均跳过"
fi

echo ""
echo "==== SUBMIT FOLD DONE ===="
echo "提交候选:"
echo "  outputs/submission_fold{0,1,2}.csv  （单折）"
echo "  outputs/submission_foldavg.csv       （三折平均，推荐先测）"
echo "判读: 任一 > 0.73 → 采纳；foldavg 大概率 > full-SM(0.55)"
