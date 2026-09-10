#!/bin/bash
#SBATCH -J cuhkx-manyMain
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
M3="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth"
M4="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth outputs/pack/main_ps_fold0_int5.pth"
TS=outputs/pack/thermal_s42_fold0_int5.pth
echo "== K1: 3main(m3)+1th(链th) flip =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $M3 --thermal $TS $CR --output outputs/sub_K_3m1th_flip.csv || echo ⚠K1
echo "== K2: 4main(m4)+1th flip =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $M4 --thermal $TS $CR --output outputs/sub_K_4m1th_flip.csv || echo ⚠K2
echo "== K3: 2main(aug1+aug2 全量近似→ 用 s42+s777)+1th =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth --thermal $TS $CR \
  --output outputs/sub_K_2m1th_flip.csv || echo ⚠K3
echo "==== K DONE ===="
ls -la outputs/sub_K*.csv 2>/dev/null
