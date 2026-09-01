#!/bin/bash
# ============================================================
# CUHK-X —— thermal 2D 预训练域匹配 A/B（修复 gray_norm 域矛盾）
# 背景：resnet18/34 ImageNet 预训练权重却被喂 (0.5,0.25) 灰度归一化 = 线外输入 → 0.40 的可能元凶
#   [1] resnet18 + ImageNet 归一化（匹配预训练域）
#   [2] resnet18 + gray_norm（现状同机对照，历史≈0.386-0.44）
# 判读：①[1] > [2]+3pt → 域矛盾是 2D 弱的原因 → 预训练 2D 线值得重构（重抽权重/重训）
#       ②[1] ≈ [2] ~0.40 → 2D 静态 0.4 就是天花板（与 norm 无关，2D 线终判）
# 用法: sbatch scripts/thermal_2d_norm.sbatch
# ============================================================
#SBATCH -J cuhkx-th-2d-norm
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
mkdir -p logs outputs/thermal_2d_norm
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "⚠️ 缺 $CROP"; exit 1; }
EP=40

echo "########## [1/2] resnet18 + ImageNet norm（匹配预训练域）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --imagenet_norm \
  --epochs $EP --crop_cache "$CROP" --save_dir outputs/thermal_2d_norm --workers 8 \
  || echo "⚠️ [1] ImageNet-norm 失败"

echo "########## [2/2] resnet18 + gray_norm（现状同机对照）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs $EP --crop_cache "$CROP" --save_dir outputs/thermal_2d_norm --workers 8 \
  || echo "⚠️ [2] gray-norm 失败"

echo "==== THERMAL 2D NORM A/B DONE ===="
echo "判读：[1](ImageNet-norm) - [2](gray-norm) ≥ +3pt → 域矛盾是元凶；≈0.40 → 2D 静态即天花板"
