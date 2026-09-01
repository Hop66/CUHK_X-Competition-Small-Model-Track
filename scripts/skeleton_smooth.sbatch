#!/bin/bash
# ============================================================
# CUHK-X —— 骨架质量处理重训（用户强调"骨架有用但处理不对"）
# 假设：NYX 骨架（MMPose 提取）有噪声 → 时间平滑（滑动平均 0.5s）去抖后
#       MotionBERT 3D 重训可能超过无平滑的 skeleton_3d（~0.45）。
# 对照：skeleton_3d.sbatch 的 repr=3d 结果（无平滑基线）
# 注意：--smooth 默认 1=关（行为不变，可复位）；此脚本用 5。
# 用法: sbatch scripts/skeleton_smooth.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-sm
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
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

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 weights/mb_lite_latest_epoch.bin"; exit 1; }

echo "######## 骨架 3D + 时间平滑(w=5) 3折 30ep（对照无平滑 skeleton_3d ~0.45）########"
python scripts/train_skeleton_motionbert.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --repr 3d --smooth 5 --epochs 30 --folds 3 --balanced \
  --label_smoothing 0 --patience 999 \
  --save_dir outputs/skeleton_smooth --workers 8
echo "######## 骨架平滑重训完成（判读：mean > 0.45 则平滑有效）########"

echo "==== SKELETON SMOOTH DONE ===="
