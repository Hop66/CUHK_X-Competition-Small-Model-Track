#!/bin/bash
# ============================================================
# CUHK-X —— 全量数据冲刺：全部 18 被试训练（不分折），3 个 seed 顺序跑（适配 1 个 GPU）
# 配置：4ch 无帧差 + crop + 数据增强 + label smoothing 0.1 + 80ep（实测 ~ep70 达峰，100ep 后段冗余）
# 用法: sbatch scripts/run_full_main.sbatch
# 产物: outputs/main_full/r2plus1d34_depthir_full_seed{42,2024,777}.pth
# ============================================================
#SBATCH -J cuhkx-full
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

# 3 个 seed 顺序训练（1 个 GPU），每个 seed 独立日志
for SEED in 42 2024 777; do
  echo "==== FULL seed=$SEED（4ch 无帧差 + crop + 增强 + ls0.1 + 80ep）===="
  python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
    --lr 1e-4 --num_frames 16 --epochs 80 --label_smoothing 0.1 \
    --full --seed "$SEED" --full_save_every 10 \
    --save_dir outputs/main_full --crop_cache bbox_train.json 2>&1 | tee "logs/cuhkx-full-$SEED.log"
  echo "==== seed $SEED DONE ===="
done

echo "==== ALL 3 SEEDS DONE ===="
