#!/bin/bash
#SBATCH -J cuhkx-anchsoft
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
# 产 0.75 锚 fused soft(test) -> outputs/oof/anchor_softprobs.pkl
python -u scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --thermal_frames 32 \
  --w_main 1.0 --w_thermal 1.0 \
  --prob_avg --flip_tta --quantize \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_anchor_soft_check.csv \
  --save_avg_probs outputs/oof/anchor_softprobs.pkl 2>&1 | tail -4
echo DONE
