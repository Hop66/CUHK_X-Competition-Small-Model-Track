#!/bin/bash
# ============================================================
# CUHK-X Iter6: thermal 重训 fold0 AB（配合 main AB 一起定重训配方）
# 基线 thermal fold0 ≈ 0.59-0.61（backbone_cmp 0.5923 / vote3d 0.6105）
# 变量(thermal 无 IR，ir_mask 不适用): cb_loss / num_frames 24 / 基线复核
# 用法: sbatch scripts/ab_fold0_thermal.sbatch
# ============================================================
#SBATCH -J cuhkx-ab-th-f0
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
mkdir -p logs outputs/ab_th_fold0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter6 thermal fold0 AB ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }

COMMON="--backbone r2plus1d34 --weights ig65m_r2plus1d34.pth --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json --modality thermal"

echo "===== [T0] 基线复核 thermal fold0 ====="
mkdir -p outputs/ab_th_fold0/base
python scripts/train_step1.py $COMMON --save_dir outputs/ab_th_fold0/base \
  2>&1 | tee logs/ab_th_fold0_base.log | tail -3 || echo "⚠️ T0"

echo "===== [T1] thermal + cb_loss ====="
mkdir -p outputs/ab_th_fold0/cb
python scripts/train_step1.py $COMMON --cb_loss --save_dir outputs/ab_th_fold0/cb \
  2>&1 | tee logs/ab_th_fold0_cb.log | tail -3 || echo "⚠️ T1"

echo "===== [T2] thermal + num_frames 24 ====="
mkdir -p outputs/ab_th_fold0/nf24
python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 24 --size 128 --batch_size 24 --workers 8 --lr 1e-4 --epochs 60 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json --modality thermal \
  --save_dir outputs/ab_th_fold0/nf24 2>&1 | tee logs/ab_th_fold0_nf24.log | tail -3 || echo "⚠️ T2"

echo ""
echo "==== Iter6 thermal fold0 AB DONE（判读: 与约0.59-0.61 基线比）===="
