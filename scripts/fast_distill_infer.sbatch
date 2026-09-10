#!/bin/bash
#SBATCH -J cuhkx-ds-inf
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) | distill flip-only 推理 ===="
CR="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"
MD=outputs/pack/main_ds_fold0_int5.pth
TD=outputs/pack/thermal_ds_fold0_int5.pth
MS=outputs/pack/main_s42_fold0_int5.pth
TS=outputs/pack/thermal_s42_fold0_int5.pth
echo "== H1: ds(双) + seed42(双) 纯翻flip 集成 =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MD $MS --thermal $TD $TS $CR --output outputs/sub_H_ds_flip.csv || echo ⚠H1
echo "== H2: 纯蒸馏双流 =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MD --thermal $TD $CR --output outputs/sub_H_dsonly_flip.csv || echo ⚠H2
echo "== H3: main 换蒸馏, thermal 保持 seed42 =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MD $MS --thermal $TS $CR --output outputs/sub_H_mainds_th42_flip.csv || echo ⚠H3
echo "==== distill infer DONE ===="
ls -la outputs/sub_H*.csv 2>/dev/null
