#!/bin/bash
# ============================================================
# CUHK-X —— YOLO 精细部位辅助（新方向）
#   [1] 逐帧 bbox 运动特征（"人怎么动"：轨迹/面积/宽高比/位置）
#   [2] YOLO11n-pose 17 关键点部位特征（带真实置信度，对照 MMPose 恒 1.0）
#   [3] GBDT 评估特征单独分类能力（规则允许非 DL）
# 判读：GBDT mean acc 显著 > 随机(2.5%) 且有信号(≥0.2)；≥0.3 且与 main 互补 → 决策级融合
# 前置：yolo11n.pt / yolo11n-pose.pt（HF 镜像自动下载）+ 训练数据
# 用法: sbatch scripts/yolo_aux.sbatch
# ============================================================
#SBATCH -J cuhkx-yolo-aux
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/feats
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "######## [1] bbox 运动特征 ########"
python scripts/yolo_motion_feats.py --train_root ~/Multimodal/data/Training/HAR \
  --out outputs/feats/bbox_feats_train.csv --stride 2

echo "######## [2] YOLO-pose 关键点部位特征 ########"
python scripts/yolo_pose_kpts.py --train_root ~/Multimodal/data/Training/HAR \
  --out outputs/feats/kpts_feats_train.csv --stride 2

echo "######## [3] GBDT 评估 ########"
python scripts/eval_feats_gbdt.py --csv outputs/feats/bbox_feats_train.csv
python scripts/eval_feats_gbdt.py --csv outputs/feats/kpts_feats_train.csv

echo "==== YOLO AUX DONE（判读：GBDT acc 显著 > 2.5% 随机基线 → 特征有信号）===="
