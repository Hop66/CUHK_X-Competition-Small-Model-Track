#!/bin/bash
# ============================================================
# CUHK-X Iter12: thermal 归一化 AB（他队 notebook 发现: 14th 基线与 BiGRU 都
#   用灰度归一 (x-0.5)/0.25；现役 thermal 是 Kinetics 归一 → 可能不匹配单通道域）
#   T0 现役(Kinetics)       对照 0.5966
#   T1 gray_norm            (x-0.5)/0.25
#   T2 gray_norm + cb_loss
# 判读: T1/T2 明显 > T0 → thermal 全量重训换 gray_norm
# 用法: sbatch scripts/ab_thermal_norm.sbatch
# ============================================================
#SBATCH -J cuhkx-ab-th-norm
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/ab_th_norm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter12 thermal norm AB ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }

COMMON="--backbone r2plus1d34 --weights ig65m_r2plus1d34.pth --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json --modality thermal"

echo "===== [T0] 现役 Kinetics 归一(对照 0.5966) ====="
mkdir -p outputs/ab_th_norm/kin
python scripts/train_step1.py $COMMON --save_dir outputs/ab_th_norm/kin \
  2>&1 | tee logs/ab_th_norm_kin.log | tail -3 || echo "⚠️ T0"

echo "===== [T1] gray_norm (x-0.5)/0.25 ====="
mkdir -p outputs/ab_th_norm/gray
python scripts/train_step1.py $COMMON --gray_norm --save_dir outputs/ab_th_norm/gray \
  2>&1 | tee logs/ab_th_norm_gray.log | tail -3 || echo "⚠️ T1"

echo "===== [T2] gray_norm + cb_loss ====="
mkdir -p outputs/ab_th_norm/graycb
python scripts/train_step1.py $COMMON --gray_norm --cb_loss --save_dir outputs/ab_th_norm/graycb \
  2>&1 | tee logs/ab_th_norm_graycb.log | tail -3 || echo "⚠️ T2"

echo ""
echo "==== Iter12 thermal norm AB DONE ===="
grep -h "fold 0 best" logs/ab_th_norm_*.log 2>/dev/null
