#!/bin/bash
# ============================================================
# CUHK-X —— 骨架两阶段：NTU 域对齐预训练 → 我们微调（大改，用对 NTU 数据）
# 阶段1: NTU A001-A040 (37,920, 17关节域对齐) 训练 backbone → 跨40被试运动表征
# 阶段2: 换我们 40 类头，微调 fold0
# 对照：混合增强 0.5447
# 判读：fold0 > 0.55 显著 → 预训练-微调有效 → 全量微调 + 参与融合
# 用法: sbatch scripts/ntu_pretrain_finetune.sbatch
# ============================================================
#SBATCH -J cuhkx-ntu-pretrain
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
mkdir -p logs outputs/ntu_pretrain outputs/skeleton_finetune
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 MotionBERT 权重"; exit 1; }
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
[ -d "$NTU_DIR" ] || { echo "❌ 缺 NTU 骨架"; exit 1; }

echo "######## 阶段1: NTU 骨架预训练（37,920 样本，17关节域对齐）########"
python scripts/train_ntu_pretrain.py \
  --ntu_root "$NTU_DIR" --pretrained weights/mb_lite_latest_epoch.bin \
  --epochs 40 --batch_size 64 --lr 1e-3 --num_frames 24 \
  --save_dir outputs/ntu_pretrain --workers 8
echo "######## 阶段1 完成 ########"

echo "######## 阶段2: NTU 预训练 backbone → 我们 fold0 微调 ########"
python scripts/train_ntu_finetune.py \
  --fold 0 --folds 3 \
  --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth \
  --epochs 60 --batch_size 64 --num_frames 24 \
  --save_dir outputs/skeleton_finetune --workers 8
echo "######## 阶段2 完成 ########"

echo "==== NTU PRETRAIN-FINETUNE DONE（判读：fold0 > 0.55 → 预训练-微调有效）===="
