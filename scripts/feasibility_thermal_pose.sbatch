#!/bin/bash
# ============================================================
# CUHK-X —— 热像姿态可行性门禁（"双视角骨架桥"第一步）
# 问题：KeypointRCNN（COCO 预训练）在铁火伪彩热像（320x240）上能否检出人+关键点？
# 这是整条线命门：通过 → 建 NYX 3D + 热像 2D 双视角骨架模型；否则 → 不投入。
# 用法: sbatch scripts/feasibility_thermal_pose.sbatch
# 产物: outputs/thermal_pose_feas/*_pose.jpg（可视化）+ 终端统计
# ============================================================
#SBATCH -J cuhkx-posfeas
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/thermal_pose_feas
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 确认热像数据存在
[ -d ~/Multimodal/data/Training/HAR/Thermal ] || { echo "❌ 缺 Thermal 数据"; exit 1; }

# 关键诊断：torchvision 版本决定归一化约定。
#   >=0.15 的 GeneralizedRCNNTransform.normalize 期望输入 [0,1] float（脚本 img/255 正确）
#   <0.15  内部会再除 255 → 双重归一化 → 检测全失败（门禁不可信）
python -c "import torchvision; print('torchvision', torchvision.__version__)"

# 若 torchvision 缺 KeypointRCNN 权重则自动下载（首次联网 ~160MB）
echo "==== 热像姿态可行性门禁（KeypointRCNN COCO 预训练）===="
python scripts/feasibility_thermal_pose.py \
  --root ~/Multimodal/data/Training/HAR \
  --n_clips 8 \
  --out outputs/thermal_pose_feas

echo "==== POSE FEASIBILITY DONE（看判读结论）===="
