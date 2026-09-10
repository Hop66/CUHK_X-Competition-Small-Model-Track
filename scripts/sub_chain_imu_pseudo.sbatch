#!/bin/bash
#SBATCH -J cuhkx-impsub
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
# main+th(0.75锚, flip) + IMU_pseudo(伪标签对齐test域后, α=0.2)
python -u scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --thermal_frames 32 \
  --imu_logits outputs/oof/imu_pseudo_test_logits.npy \
  --w_main 0.40 --w_thermal 0.40 --w_imu 0.20 \
  --prob_avg --flip_tta --quantize \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_chain_imu_pseudo.csv 2>&1
echo DONE
