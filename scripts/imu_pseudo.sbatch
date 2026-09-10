#!/bin/bash
#SBATCH -J cuhkx-impse
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== Noisy-Student 式 IMU 伪标签微调(0.75锚 soft 拉齐 test 域) =="
python -u scripts/train_imu_pseudo.py --balanced 2>&1
echo DONE
