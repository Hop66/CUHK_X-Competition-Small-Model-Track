#!/bin/bash
# ============================================================
# CUHK-X —— main+thermal 动态融合提交（样本级动态加权，零训练零拟合）
# 对照：朴素平均 0.73 保底（submission_fusion_mT.csv）
# 判读：LB > 0.73 → 动态融合有效
# 前置：outputs/pack/main_fold0_int5.pth + thermal_fold0_int5.pth + 4 bbox
# 用法: sbatch scripts/dynamic_fusion.sbatch
# ============================================================
#SBATCH -J cuhkx-dynfuse
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for f in outputs/pack/main_fold0_int5.pth outputs/pack/thermal_fold0_int5.pth \
         bbox_train.json bbox_test.json bbox_thermal_train.json bbox_thermal_test.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

echo "######## 动态融合推理（样本级加权，不拟合）########"
python scripts/dynamic_fusion.py \
  --main outputs/pack/main_fold0_int5.pth \
  --thermal outputs/pack/thermal_fold0_int5.pth \
  --quantize --flip_tta \
  --main_crop bbox_train.json --main_test_crop bbox_test.json \
  --thermal_crop bbox_thermal_train.json --thermal_test_crop bbox_thermal_test.json \
  --output submission_dynamic.csv

echo "==== DYNAMIC FUSION DONE（对比 0.73 保底）===="
