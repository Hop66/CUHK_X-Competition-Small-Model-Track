#!/bin/bash
# ============================================================
# CUHK-X —— thermal 2D-per-frame（0.8 组方法强化版）：ImageNet 预训练 2D CNN + per-frame logits mean
# 修正 FrameNet(从零 0.21) 判负过早：0.8 组方法 = 逐帧 2D 分类 + logits 时间平均 + 灰度拉伸 + 分段采样，
#   并不绑定"从零小 CNN"。这里用 torchvision ImageNet 预训练 resnet18/34 复刻同一聚合，
#   加 gray_norm + segment + YOLO 人物 crop（减背景、聚焦人物）。
# 对照：thermal 基线 fold0 = 0.6339（R2+1D-34 3D 时序）
#   [1] resnet18 预训练 + per-frame logits（gray+segment+crop1.4）
#   [2] resnet34 预训练 + 同上
#   [3] resnet18 + 紧框 crop(margin1.2)：减背景 × 2D per-frame 叠加
# 判读：任一 fold0 > 0.6339 → 2D-per-frame 确为 thermal 强路径 → 全量+入 0.73 融合（体积小）
#        <0.55 → 2D-per-frame 即使强化也不及 3D 时序 → thermal 走 3D rebuild 线
# 前置：bbox_thermal_train.json（[3] 需 bbox_thermal_m12.json，缺则跳过该档）
# 用法: sbatch scripts/thermal_2d_framevote.sbatch
# ============================================================
#SBATCH -J cuhkx-th-2d-framevote
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
mkdir -p logs outputs/thermal_2dframevote
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "⚠️ 缺 $CROP（[1][2] 将失败，可先跑 detect --modality thermal）"; exit 1; }
EP=64

echo "########## [1/3] resnet18(ImageNet) + per-frame logits mean ##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --crop_cache "$CROP" --epochs $EP --save_dir outputs/thermal_2dframevote --workers 8 \
  || echo "⚠️ [1] resnet18 失败"

echo "########## [2/3] resnet34(ImageNet) + per-frame logits mean ##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet34 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --crop_cache "$CROP" --epochs $EP --save_dir outputs/thermal_2dframevote --workers 8 \
  || echo "⚠️ [2] resnet34 失败"

echo "########## [3/3] resnet18 + 紧框 crop(margin1.2)（减背景 × 2D per-frame）##########"
CROP_TIGHT=bbox_thermal_m12.json
if [ ! -f "$CROP_TIGHT" ]; then
  python -m src.detect --mode train --out "$CROP_TIGHT" --modality thermal --margin 1.2 \
    || echo "⚠️ detect margin1.2 失败"
fi
if [ -f "$CROP_TIGHT" ]; then
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --crop_cache "$CROP_TIGHT" --epochs $EP --save_dir outputs/thermal_2dframevote --workers 8 \
  || echo "⚠️ [3] resnet18+紧框失败"
else
  echo "⚠️ [3] 跳过（缺紧框）"
fi

echo "==== THERMAL 2D FRAMEVOTE DONE ===="
echo "判读：任一 fold0 > 0.6339 → 2D-per-frame 为 thermal 强路径 → 全量+融合；<0.55 → 走 3D rebuild 线"
