#!/bin/bash
# ============================================================
# CUHK-X —— 回答用户质疑: fold val 三折骨架运动提升到底多大、是否全正
# 用与提交完全一致的推理链路在 fold1/fold2 val split 上复演 static vs learned
# (fold0 已测: static=0.6663, learned(α=0.548)=0.6674 → +0.0011)
# 用法: sbatch scripts/diag_fold_motion_margin.sbatch
# ============================================================
#SBATCH -J cuhkx-fold-margin
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for F in 1 2; do
  CK="outputs/main_dual/main_SM_fold${F}.pth"
  [ -f "$CK" ] || { echo "❌ 缺 $CK"; continue; }
  echo "===== fold${F} SM 自验(与 fold0 同链路, 必须 --fold $F) ====="
  python scripts/selftest_main_dual_infer.py --ckpt "$CK" --fold "$F" --flip_tta || echo "⚠️ fold$F selftest 失败"
done

echo ""
echo "==== DONE: 对照 fold0 static=0.6663 learned(0.548)=0.6674 (+0.0011) ===="
echo "ckpt 自带 best_acc: SM fold0=0.6652 fold1=0.6138 fold2=0.7058 | S fold0=0.6598 fold1=0.5905 fold2=0.6782"
