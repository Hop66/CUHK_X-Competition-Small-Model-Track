#!/bin/bash
# ============================================================
# CUHK-X —— 批次3b：VideoMAE-S A/B 诊断（fold0 thermal）
# 已跑（ImageNet 归一化 lr=1e-4 60ep）→ fold0=0.3127，远低于 R2+1D 基线 0.6339
# A: Kinetics 归一化（与 R2+1D 基线完全同预处理）lr=1e-4 60ep → 隔离"归一化"
# B: Kinetics + lr=5e-4 + warmup5 + 100ep + patience30       → 归一化+训练配置都修
# 判读: A>0.45 → 归一化是主因，直接换 Kinetics 跑 thermal 全量
#       B>0.50 → 训练配置可救，用 B 配置全量
#       都~0.3 → 自监督先验/结构不敌 IG65M 监督 → 关闭 VideoMAE-S 线（同 swin3d_t）
# 前置: weights/videomae_s_k400_pretrain.pth + bbox_thermal_train.json
# 用法: sbatch scripts/videomae_ab.sbatch
# ============================================================
#SBATCH -J cuhkx-videomae-ab
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

[ -f weights/videomae_s_k400_pretrain.pth ] || { echo "❌ 缺 VideoMAE-S 权重"; exit 1; }
[ -f bbox_thermal_train.json ] || { echo "❌ 缺 bbox_thermal_train.json"; exit 1; }

echo "######## A: Kinetics 归一化 lr=1e-4 60ep（隔离预处理变量，对照 0.3127 / 0.6339）########"
python scripts/train_videomae.py \
  --ckpt weights/videomae_s_k400_pretrain.pth \
  --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --lr 1e-4 \
  --normalize kinetics --patience 15 --warmup_epochs 0 \
  --save_dir outputs/videomae_ab_a --workers 8

echo "######## B: Kinetics + lr=5e-4 + warmup5 + 100ep + patience30（修训练配置）########"
python scripts/train_videomae.py \
  --ckpt weights/videomae_s_k400_pretrain.pth \
  --epochs 100 --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --lr 5e-4 \
  --normalize kinetics --patience 30 --warmup_epochs 5 \
  --save_dir outputs/videomae_ab_b --workers 8

echo "==== VIDEO MAE A/B DONE（判读：A>0.45→归一化主因换Kinetics全量；B>0.5→配置修复；都~0.3→关闭）===="
