#!/bin/bash
#SBATCH -J cuhkx-Ctest
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
CR="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"
M2="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth"
echo "== C1: 2main(s42,s777)+2th(s42,s777) flip (=fold C 形态) =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $M2 --thermal outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth \
  $CR --output outputs/sub_C2_2m2th_flip.csv || echo ⚠C1
echo "== C2: 2main+th(s42,ds) flip =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $M2 --thermal outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_ds_fold0_int5.pth \
  $CR --output outputs/sub_C2_2m2th_ds_flip.csv || echo ⚠C2
echo "== C3: 3main+2th(s42,s777) flip =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth \
  $CR --output outputs/sub_C2_3m2th_flip.csv || echo ⚠C3
echo "==== C DONE ===="
