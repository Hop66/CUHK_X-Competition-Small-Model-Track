#!/bin/bash
# ============================================================
# CUHK-X —— 跨模态对比学习（main Depth+IR ↔ thermal，InfoNCE）
# 验证：对比学习是否提升 main 的 val（vs 基线 4ch+增强 fold0=0.6192）
# 用法: sbatch scripts/train_contrastive.sbatch
# 产物: outputs/contrastive/r2plus1d34_contrastive_fold{0,1,2}.pth
# ============================================================
#SBATCH -J cuhkx-contrast
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f ig65m_r2plus1d34.pth ]; then
  echo "❌ ig65m_r2plus1d34.pth 不存在"; exit 1
fi

echo "==== 跨模态对比学习（main↔thermal, 3折, 60ep, λ=0.1, ls=0.1）===="
python scripts/train_contrastive.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --folds 3 \
  --lambda_contrast 0.1 --temperature 0.07 --label_smoothing 0.1 \
  --aug_strength 2 \
  --save_dir outputs/contrastive --crop_cache bbox_train.json --thermal_crop bbox_thermal.json

echo "==== ALL DONE ===="
