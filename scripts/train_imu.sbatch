#!/bin/bash
# ============================================================
# CUHK-X —— IMU 单模态摸底（P2，异源惯性 1D CNN，3 折 30ep）
# 前置：先跑 explore_skeleton_2d.py（CPU）看骨架投影可行性
# 用法: sbatch scripts/train_imu.sbatch
# 判据: mean val >0.5 且与 main 互补 → 值得加入动态集成
# ============================================================
#SBATCH -J cuhkx-imu
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/imu
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==== IMU 单模态摸底（1D CNN, 3折 30ep, balanced）===="
python scripts/train_imu.py --epochs 30 --folds 3 --balanced --save_dir outputs/imu

echo "==== IMU 摸底完成（看 mean val）===="
