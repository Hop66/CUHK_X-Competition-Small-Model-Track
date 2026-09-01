#!/bin/bash
# ============================================================
# CUHK-X —— 类级门控融合对拍（fold0 内留一被试验证）
# 三模态 per-class 门控 vs uniform（prob_avg）：
#   soft_gate(T=1) / hard_gate(每类取最强模态) / uniform
# 回答：
#   A. per-class 门控是否有泛化增益（>+1pt 且有非单折证据）→ 上 0.73 测试线 A/B
#   B. 骨架是否被 gate 稳定选中为某类 winner → 骨架有效用法=仅这类提供软证据
# 前置：main/thermal fold0 ckpt + 可选 skeleton fold0 ckpt + bbox json
# 用法: sbatch scripts/eval_fusion_perclass.sbatch
# ============================================================
#SBATCH -J cuhkx-fusion-perclass
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
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

MAIN_CKPT=${MAIN_CKPT:-outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth}
TH_CKPT=${TH_CKPT:-outputs/thermal_34/r2plus1d34_thermal_fold0.pth}
SKEL_CKPT=${SKEL_CKPT:-outputs/skeleton_aug/augmented_fold0.pth}
[ -f "$SKEL_CKPT" ] || SKEL_CKPT=""

if [ -n "$SKEL_CKPT" ]; then
  python scripts/eval_fusion_perclass.py --fold 0 \
    --main_ckpt "$MAIN_CKPT" --thermal_ckpt "$TH_CKPT" \
    --skeleton_ckpt "$SKEL_CKPT" \
    || echo "⚠️ 三模态 per-class 对拍失败"
else
  echo "⚠️ 跳过 skeleton（缺 ckpt），仅 main+thermal"
  python scripts/eval_fusion_perclass.py --fold 0 \
    --main_ckpt "$MAIN_CKPT" --thermal_ckpt "$TH_CKPT" \
    || echo "⚠️ 双模态 per-class 对拍失败"
fi

echo "==== FUSION PERCLASS DONE（判读：soft/hard_gate - uniform ≥ +1pt → 上测试线 A/B）===="
