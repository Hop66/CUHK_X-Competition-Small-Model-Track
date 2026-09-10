#!/bin/bash
# ============================================================
# CUHK-X Iter4: 重训战役 fold0 单变量 AB（为全量重训选配方）
# 基线: baseline_aug2 fold0 val = 0.6695（已有）
# 变量(各自单独试, 每个 ~60ep, fold0 only):
#   AB1 cb_loss        (长尾 29.5× 类平衡)
#   AB2 ir_mask        (IR Otsu 全身掩码抑制背景)
#   AB3 cb_loss+ir_mask
#   AB4 num_frames 24  (更长时序)
# 判定: fold0 val 超过 0.6695 的变量进全量重训组合
# 用法: sbatch scripts/ab_fold0_retrain.sbatch
# ============================================================
#SBATCH -J cuhkx-ab-fold0
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
mkdir -p logs outputs/ab_fold0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter4 fold0 AB ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }

COMMON="--backbone r2plus1d34 --weights ig65m_r2plus1d34.pth --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_train.json --modality depthir"

echo "===== [AB1] cb_loss ====="
mkdir -p outputs/ab_fold0/cb
python scripts/train_step1.py $COMMON --cb_loss \
  --save_dir outputs/ab_fold0/cb 2>&1 | tee logs/ab_fold0_cb.log | tail -3 || echo "⚠️ AB1"

echo "===== [AB2] ir_mask ====="
mkdir -p outputs/ab_fold0/ir
python scripts/train_step1.py $COMMON --ir_mask \
  --save_dir outputs/ab_fold0/ir 2>&1 | tee logs/ab_fold0_ir.log | tail -3 || echo "⚠️ AB2"

echo "===== [AB3] cb_loss+ir_mask ====="
mkdir -p outputs/ab_fold0/cbir
python scripts/train_step1.py $COMMON --cb_loss --ir_mask \
  --save_dir outputs/ab_fold0/cbir 2>&1 | tee logs/ab_fold0_cbir.log | tail -3 || echo "⚠️ AB3"

echo "===== [AB4] num_frames 24 ====="
mkdir -p outputs/ab_fold0/nf24
python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 24 --size 128 --batch_size 24 --workers 8 --lr 1e-4 --epochs 60 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_train.json --modality depthir \
  --save_dir outputs/ab_fold0/nf24 2>&1 | tee logs/ab_fold0_nf24.log | tail -3 || echo "⚠️ AB4"

echo ""
echo "==== Iter4 fold0 AB DONE（判读: 与 baseline_aug2 fold0=0.6695 比较）===="
grep -hE "best|val acc|fold0" logs/ab_fold0_*.log 2>/dev/null | tail -30
