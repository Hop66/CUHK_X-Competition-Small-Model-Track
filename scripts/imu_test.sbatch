#!/bin/bash
#SBATCH -J cuhkx-imute
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== IMU test logits 提取(3折 IMUCNN 平均) =="
python -u scripts/extract_imu_test.py --model gru --ckpts outputs/imu_gru/imu_gru_fold0.pth outputs/imu_gru/imu_gru_fold1.pth outputs/imu_gru/imu_gru_fold2.pth --out outputs/oof/imu_gru_test_logits.npy 2>&1
echo DONE
