#!/bin/bash
# ============================================================
# CUHK-X —— main + ArcFace 度量学习（跨身份泛化，从未试过的损失层面）
# 背景：0.7→0.8+ 队伍情报 → 我们所有实验用 CE，从未试 ArcFace（人脸/ReID 跨身份核心方法）
# 评估：--fold -1 跑全部 3 折，输出 mean±std（可靠评估协议，避免单折噪声误判）
# 判据：ArcFace 3 折 mean ± std 与 CE 基线 3 折（同协议）对比，均值差 > std 才有把握
# 前置：ig65m_r2plus1d34.pth + bbox_train.json
# 用法: sbatch scripts/main_arcface.sbatch
# ============================================================
#SBATCH -J cuhkx-arcface
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_arcface
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
[ -f bbox_train.json ] || { echo "❌ 缺 bbox_train.json"; exit 1; }

echo "######## main + ArcFace(s=30,m=0.5) 全 3 折（mean±std 可靠评估）########"
python scripts/train_main_arcface.py \
  --fold -1 --folds 3 --weights ig65m_r2plus1d34.pth \
  --epochs 60 --batch_size 32 --lr 1e-4 --s 30.0 --m 0.5 \
  --crop_cache bbox_train.json --save_dir outputs/main_arcface --workers 8
echo "######## ArcFace 3 折完成 ########"

echo "==== MAIN ARCFACE DONE ===="
echo "判读: ArcFace 3折 mean±std 与 CE 基线 3折（同协议）对比；均值差 > std 才有把握判定有效"
