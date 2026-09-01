#!/bin/bash
# ============================================================
# CUHK-X —— thermal 2D-per-frame 整帧 A/B（补 14th 精髓缺失档）
# 背景：thermal_2d_framevote 全用 YOLO crop（resnet18 只有 0.3863）; 但 14th(0.8) 精髓 = 整帧无裁剪 + 帧级 logits mean。
#        → 我们从未测过"无 crop 整帧"，本脚本补上，隔离 crop 对帧级投票的效应。
# 对照基线：resnet18+固定crop = 0.3863 / +紧框 = 0.3981（thermal_2d_framevote）
#   [1] resnet18 + 无crop 整帧 + seg/gray + 8帧112 + 40ep    → 隔离 crop 效应
#   [2] FrameNet 从零 + 无crop + 严格复刻 notebook(8ep, 无LS, lr3e-4) → 复刻精髓
#   [3] FrameNet 从零 + 无crop + 长训(40ep, LS0.1)            → 从零模型增强版
# 判读：
#   [1] vs 0.3863 → crop 是帮 2D-per-frame 还是害它（整帧>crop → 回滚 thermal 的 crop pipeline）
#   [2] vs 我们以前从零先验 0.21 → 复刻 vs 加料差异；逼近 0.5+ → 帧级+整帧在 thermal 是真路
#   任一显著 >0.5 → thermal 2D 起死回生，进 0.73 融合冲 0.8；全部 <0.45 → 2D 线终判负、thermal 回 3D
# 用法: sbatch scripts/thermal_2d_ab.sbatch
# ============================================================
#SBATCH -J cuhkx-th-2d-ab
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/thermal_2d_ab
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "########## [1/3] resnet18 无crop 整帧（隔离 crop 效应）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs 40 --save_dir outputs/thermal_2d_ab --workers 8 \
  || echo "⚠️ [1] resnet18 无crop 失败"

echo "########## [2/3] FrameNet 从零 无crop 严格复刻（8ep 无LS lr3e-4）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch framenet \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs 8 --no_smooth --lr 3e-4 --save_dir outputs/thermal_2d_ab --workers 8 \
  || echo "⚠️ [2] 严格复刻 失败"

echo "########## [3/3] FrameNet 从零 无crop 长训（40ep LS0.1）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch framenet \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs 40 --save_dir outputs/thermal_2d_ab --workers 8 \
  || echo "⚠️ [3] 从零长训 失败"

echo "==== THERMAL 2D 整帧 A/B DONE ===="
echo "结果: outputs/thermal_2d_ab/*.pth"
echo "判读: ①[1] vs 0.3863 → crop 是否帮 2D；②[2] vs 0.21 → 严格复刻；③ 任一 >0.5 → 2D 起死回生"
