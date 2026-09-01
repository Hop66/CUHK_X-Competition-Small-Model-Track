#!/bin/bash
# ============================================================
# CUHK-X —— 类感知 / 置信门控融合提交（main+thermal，对比 0.73 朴素平均）
# 在 fold0 val 上用同模型学融合超参（τ / 类权重）→ 应用 test → csv
# 判读：LB > 0.73 则融合方式比朴素平均强 → 定为新保底
# 前置：outputs/pack/main_fold0_int5.pth + thermal_fold0_int5.pth（0.73 同款）
# 用法: sbatch scripts/submit_gated.sbatch
# ============================================================
#SBATCH -J cuhkx-gated
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

echo "######## 类感知/置信门控融合推理（val 学 τ/类权重 → test 应用）########"
python scripts/fusion_class_gated.py \
  --main outputs/pack/main_fold0_int5.pth \
  --thermal outputs/pack/thermal_fold0_int5.pth \
  --quantize --flip_tta \
  --test_root "${TEST_ROOT:-$HOME/Multimodal/data/Testing/data/small_model_track_test}" \
  --main_crop bbox_train.json --main_test_crop bbox_test.json \
  --thermal_crop bbox_thermal_train.json --thermal_test_crop bbox_thermal_test.json \
  --output submission_gated.csv

echo "==== GATED FUSION DONE（对比 0.73 保底 = submission_fusion_mT.csv）===="
python -c "
import pandas as pd
df = pd.read_csv('submission_gated.csv')
print(f'rows={len(df)} 类数={df.prediction.nunique()} 类0={int((df.prediction==0).sum())}')
"
