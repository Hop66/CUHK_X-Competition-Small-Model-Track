#!/bin/bash
# ============================================================
# CUHK-X —— 骨架输入表示对照实验（验证"3D 表示 + 每 epoch 随机增强"的增益）
# 两个修复：
#   [1] MotionBertSkeletonDataset 默认喂原生 3D [水平,垂直,深度]（匹配 model_pos 3D 预训练，
#       不丢深度；旧 [x,y,conf] 的 conf 恒 1.0 是退化输入）
#   [2] train_skeleton_motionbert.py 修 bug：无 --balanced 时 sampler=None + shuffle=False
#       → 同一样本增强每 epoch 确定性（过拟合固定扰动）。现在无 sampler 时 shuffle=True。
# 对照（同一设置：强化增强 + balanced + 无 LS + 无早停，3 折 30ep）：
#   [1] repr=2d（旧表示 + 修 shuffle）→ 应 ≥ 旧 0.4513（shuffle 修复本身应有增益）
#   [2] repr=3d（原生 3D）→ 对比 [1]，验证 3D 表示增益
# 用法: sbatch scripts/skeleton_3d.sbatch
# ============================================================
#SBATCH -J cuhkx-skel3d
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

# ============ [1/2] 对照：repr=2d（旧表示 + 修 shuffle bug）============
echo "######## [1/2] 骨架 repr=2d（对照，修 shuffle 后应 ≥ 0.4513）########"
python scripts/train_skeleton_motionbert.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --repr 2d --epochs 30 --folds 3 --balanced \
  --label_smoothing 0 --patience 999 \
  --save_dir outputs/skeleton_3d/2d --workers 8
echo "######## [1/2] repr=2d 完成 ########"

# ============ [2/2] 实验：repr=3d（原生 3D）============
echo "######## [2/2] 骨架 repr=3d（原生 3D，匹配 model_pos 预训练）########"
python scripts/train_skeleton_motionbert.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --repr 3d --epochs 30 --folds 3 --balanced \
  --label_smoothing 0 --patience 999 \
  --save_dir outputs/skeleton_3d/3d --workers 8
echo "######## [2/2] repr=3d 完成 ########"

echo "==== SKELETON 3D 对照完成（对比 2d vs 3d vs 旧 0.4513）===="
