#!/bin/bash
#SBATCH -J cuhkx-imu2
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
# a) 0.75 锚链复现(应 100% == sub_chain_nf32_flip.csv, 验证 crop/flip/成员)
python -u scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --thermal_frames 32 \
  --w_main 1.0 --w_thermal 1.0 \
  --prob_avg --flip_tta --quantize \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_chain_anchor_re.csv 2>&1
# b) + 双 IMU(α0.30: main/th=0.35/0.35, cnn/gru 各 0.15)
python -u scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --thermal_frames 32 \
  --imu_logits outputs/oof/imu_test_logits.npy outputs/oof/imu_gru_test_logits.npy \
  --w_main 0.35 --w_thermal 0.35 --w_imu 0.30 \
  --prob_avg --flip_tta --quantize \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_chain_imu2.csv 2>&1
echo DONE
