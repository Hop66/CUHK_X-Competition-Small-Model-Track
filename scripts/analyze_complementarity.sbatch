#!/bin/bash
# ============================================================
# CUHK-X —— 模态互补性分析（main vs skeleton/thermal 的错误重叠）
# 关键指标：互补纠正率（辅助模态能救 main 的错的比例）
#   >0.3 → 互补，集成值得试；<0.1 → 拖累（已实测 thermal/skeleton 拖累）
# 用法: sbatch scripts/analyze_complementarity.sbatch
# 依赖: 增强后的骨架 checkpoint outputs/skeleton_aug/motionbert_fold0.pth
# ============================================================
#SBATCH -J cuhkx-complement
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

# 骨架 checkpoint：优先增强后的，否则用旧的
SKEL_CKPT=outputs/skeleton_aug/motionbert_fold0.pth
[ -f "$SKEL_CKPT" ] || SKEL_CKPT=outputs/skeleton_mb/motionbert_fold0.pth
echo "使用 skeleton checkpoint: $SKEL_CKPT"

# thermal checkpoint（round3 训练的 3ch 无帧差模型）
TH_CKPT=outputs/thermal_34/r2plus1d34_thermal_fold0.pth
[ -f "$TH_CKPT" ] || { echo "⚠️ 缺 thermal checkpoint: $TH_CKPT，仅分析 skeleton"; TH_CKPT=""; }

echo "==== main vs skeleton/thermal 互补性分析（fold0）===="
if [ -n "$TH_CKPT" ]; then
  python scripts/analyze_complementarity.py --fold 0 --per_class --skeleton_ckpt "$SKEL_CKPT" --thermal_ckpt "$TH_CKPT"
else
  python scripts/analyze_complementarity.py --fold 0 --per_class --skeleton_ckpt "$SKEL_CKPT"
fi

echo "==== ALL DONE ===="
