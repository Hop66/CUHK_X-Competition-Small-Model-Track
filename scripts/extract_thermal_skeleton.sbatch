#!/bin/bash
# ============================================================
# CUHK-X —— Step 2：全量热像 2D 骨架提取（KeypointRCNN 零样本迁移）
# 门禁已过（job 42110：人检测 1.00 / 骨架完整 0.95）→ 全量跑 2931 clip
# 产物: outputs/thermal_skeleton/<Action>/<Subject>/<sample>.npz
#       （kp [N,17,2] 归一化 + conf [N,17]）
# 用法: sbatch scripts/extract_thermal_skeleton.sbatch
# ============================================================
#SBATCH -J cuhkx-thskel
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/thermal_skeleton
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -d ~/Multimodal/data/Training/HAR/Thermal ] || { echo "❌ 缺 Thermal 数据"; exit 1; }

# 若权重已缓存则跳过下载
[ -f "$HOME/.cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth" ] \
  && echo "KeypointRCNN 权重已缓存" || echo "KeypointRCNN 权重将自动下载（~160MB）"

echo "==== 全量热像 2D 骨架提取（2931 clip）===="
python scripts/extract_thermal_skeleton.py \
  --root ~/Multimodal/data/Training/HAR \
  --out outputs/thermal_skeleton \
  --step 1

echo "==== THERMAL SKELETON EXTRACT DONE ===="
ls outputs/thermal_skeleton | head -20
find outputs/thermal_skeleton -name "*.npz" | wc -l
