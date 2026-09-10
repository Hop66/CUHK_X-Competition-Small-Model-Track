#!/bin/bash
#SBATCH -J cuhkx-srclg
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
# 与 save_anchor_soft.sbatch 同款命令, 额外导出 main/thermal 各自 test logits (同源: flip_tta/quantize/crop)
python -u scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --thermal_frames 32 \
  --w_main 1.0 --w_thermal 1.0 \
  --prob_avg --flip_tta --quantize \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_src_logits_check.csv \
  --save_main_logits outputs/oof/main_src_test.pkl \
  --save_thermal_logits outputs/oof/thermal_src_test.pkl 2>&1 | tail -6
echo DONE
