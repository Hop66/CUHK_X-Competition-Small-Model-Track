#!/bin/bash
# ============================================================
# CUHK-X —— Step 3：双视角骨架双分支训练（共享 backbone + 双输入头）
# 热像姿态桥（2026-08-23 顶会方法论定稿）：
#   分支1 = 热像 2D [x,y,conf]（Step 2 提取，outputs/thermal_skeleton/*.npz）
#   分支2 = NYX 3D [x,y,z]（Skeleton predictions）
#   共享 DSTformer encoder + 双输入头（预训练 joints_embed 初始化）+ 双分类头
#   + logits 加权融合（可学习）+ 3D 分支绕竖直轴旋转增强（补相机视点差）
#
# 前置: Step 2 热像骨架提取完成（done=2873）→ outputs/thermal_skeleton
#       weights/mb_lite_latest_epoch.bin 存在
# 用法: sbatch scripts/dual_branch.sbatch
# ============================================================
#SBATCH -J cuhkx-dualbranch
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
N_NPZ=$(find outputs/thermal_skeleton -name "*.npz" | wc -l)
echo "==== 热像 2D 骨架 npz 数: $N_NPZ（应 ≈2873）===="
[ "$N_NPZ" -ge 2000 ] || { echo "❌ 热像骨架不足（<2000）"; exit 1; }

# ============ 主实验：sigmoid conf + 绕竖直轴旋转 30° + fused loss + 可学习融合 ============
echo "######## 双分支（fused/learn/sigmoid/rot30）########"
python scripts/train_dual_branch.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --epochs 30 --folds 3 --balanced \
  --conf_norm sigmoid --rot_angle 30.0 \
  --loss_mode fused --fusion learn \
  --save_dir outputs/dual_branch --workers 8
echo "######## 双分支主实验完成 ########"

echo "==== DUAL BRANCH 训练完成（对照基线：3D 单分支 skeleton_3d ~0.45）===="
