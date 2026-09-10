#!/bin/bash
#SBATCH -J cuhkx-mega4
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
MAIN4="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth outputs/pack/main_ds_fold0_int5.pth"
TH4="outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth outputs/pack/thermal_avg_int5.pth outputs/pack/thermal_ds_fold0_int5.pth"
echo "== MEGA4: 3seed+distill × 双流, flip-only =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MAIN4 --thermal $TH4 $CR --output outputs/sub_J_mega4_flip.csv || echo "⚠J"
echo "== J2: mega4 + megamax(6main) =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MAIN4 outputs/pack/main_fold0_int5.pth outputs/pack/main_fold1_int5.pth outputs/pack/main_ps_fold0_int5.pth \
  --thermal $TH4 $CR --output outputs/sub_J_mega8_flip.csv || echo "⚠J2"
echo "==== MEGA4 DONE ===="
ls -la outputs/sub_J*.csv 2>/dev/null
