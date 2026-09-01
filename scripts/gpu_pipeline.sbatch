#!/bin/bash
# ============================================================
# CUHK-X —— GPU 统一排队（自动跑完，无需手动干预）
# 顺序：
#   [1] 骨架强化训练（带正则 30ep×3fold）→ outputs/skeleton_aug
#   [2] 互补分析（oracle 修复）→ 看互补纠正率 + oracle 上限
#   [3] 对比学习 3 折（aug_strength=2 定版，60ep）→ 对比 aug_search s2 fold0=0.6609
# 注意：已证明对比学习 60ep 收敛（续跑 0.6663 < 0.6771），不再续跑
# 用法: sbatch scripts/gpu_pipeline.sbatch
# ============================================================
#SBATCH -J cuhkx-gpu-pipeline
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
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

# ============ [1/3] 骨架强化训练（MotionBERT + 强化 CG 增强 + ls0.1 + 早停，3折 30ep）============
echo "######## [1/3] 骨架强化训练 ########"
[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 weights/mb_lite_latest_epoch.bin"; exit 1; }
python scripts/train_skeleton_motionbert.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --epochs 30 --folds 3 --balanced \
  --label_smoothing 0.1 --patience 8 \
  --save_dir outputs/skeleton_aug
echo "######## [1/3] 骨架训练完成（对比：无增强 0.4177 / 温和 0.4380 / 强化 0.4513 / +正则 待看）########"

# ============ [2/3] 互补分析（依赖 [1] 骨架 checkpoint，快）============
echo "######## [2/3] 互补分析 ########"
SKEL_CKPT=outputs/skeleton_aug/motionbert_fold0.pth
[ -f "$SKEL_CKPT" ] || SKEL_CKPT=outputs/skeleton_mb/motionbert_fold0.pth
python scripts/analyze_complementarity.py --fold 0 --skeleton_ckpt "$SKEL_CKPT"
echo "######## [2/3] 互补分析完成（看互补纠正率 + oracle 完美选择上限）########"

# ============ [3/3] 跨模态对比学习 3 折（aug_strength=2 定版，60ep）============
echo "######## [3/3] 跨模态对比学习（main↔thermal, aug_strength=2, 3折60ep）########"
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }
python scripts/train_contrastive.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --folds 3 \
  --lambda_contrast 0.1 --temperature 0.07 --label_smoothing 0.1 \
  --aug_strength 2 \
  --save_dir outputs/contrastive --crop_cache bbox_train.json --thermal_crop bbox_thermal.json
echo "######## [3/3] 对比学习完成（看各 fold main_val vs 基线 0.6609）########"

echo "==== ALL GPU PIPELINE DONE ===="
