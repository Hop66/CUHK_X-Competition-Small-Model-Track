#!/bin/bash
# ============================================================
# CUHK-X —— B 线：骨干对比 fold0 验证（换骨干能否质变？）
# 目的：更强视频骨干（torchvision Kinetics 预训练）在 cross-subject 下
#       是否显著超 R2Plus1D-34（IG-65M）？
#   基线：R2Plus1D-34 thermal fold0 = 0.6339 / main fold0 = 0.6609
#   thermal 3ch 先验证（直接匹配预训练域，最公平）
# 候选：r2plus1d34(基线) / swin3d_t
#   （x3d_m 服务器 torchvision 无 X3D_M_Weights 导入 → 排除；
#     s3d/mvit_v2_s 需 224 输入，128 下 avgpool/patch 不兼容 → 排除）
# 判读：swin3d_t fold0 > 0.6339 → 换骨干全量重训 thermal → 融合冲更高
# 注意：新增文件独立，不触碰现有主线代码（可复位）。
# 用法: sbatch scripts/backbone_compare.sbatch
# ============================================================
#SBATCH -J cuhkx-bbcompare
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
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f bbox_thermal_train.json ] || { echo "❌ 缺 bbox_thermal_train.json"; exit 1; }

MODALITY=thermal
CROP=bbox_thermal_train.json
EPOCHS=60

# ============ [1/2] 基线 r2plus1d34（IG-65M）============
echo "######## [1/2] 基线 r2plus1d34 thermal fold0 ########"
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }
python scripts/train_backbone_compare.py \
  --backbone r2plus1d34 --modality $MODALITY --crop_cache $CROP \
  --epochs $EPOCHS --fold 0 --save_dir outputs/backbone_cmp --workers 8
echo "######## [1/2] 基线完成（对照 0.6339）########"

# ============ [2/2] swin3d_t ============
echo "######## [2/2] swin3d_t thermal fold0 ########"
python scripts/train_backbone_compare.py \
  --backbone swin3d_t --modality $MODALITY --crop_cache $CROP \
  --epochs $EPOCHS --fold 0 --save_dir outputs/backbone_cmp --workers 8
echo "######## [2/2] swin3d_t 完成 ########"

echo "==== BACKBONE COMPARE DONE（判读：swin3d_t fold0 > 0.6339？）===="
