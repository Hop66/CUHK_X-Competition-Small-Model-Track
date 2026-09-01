#!/bin/bash
# ============================================================
# CUHK-X —— 高分辨率 fold0 诊断（main r2plus1d34，224 vs 128 基线 0.6609）
# 理由：128 偏离 IG65M 预训练域(224)；224 捕捉手部细微运动（打字/玩手机/写字等）
# 判读：fold0 > 0.6609 且显著 → 分辨率是瓶颈 → 全量 224 重训 main
#       ~0.6609 → 分辨率非瓶颈 → 0.73 为当前框架上限，收尾
# 前置：ig65m_r2plus1d34.pth + bbox_train.json
# 用法: sbatch scripts/res224_diag.sbatch
# ============================================================
#SBATCH -J cuhkx-res224
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/res224
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
[ -f bbox_train.json ] || { echo "❌ 缺 bbox_train.json"; exit 1; }

echo "######## main r2plus1d34 224 分辨率 fold0（对照 128 基线 0.6609）########"
python scripts/train_step1.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --num_frames 16 --size 224 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_train.json --lr 1e-4 --batch_size 16 \
  --save_dir outputs/res224
echo "######## 224 分辨率完成 ########"

echo "==== RES224 DONE（判读：fold0 > 0.6609 显著 → 分辨率是瓶颈，全量 224 重训）===="
