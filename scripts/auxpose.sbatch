#!/bin/bash
# ============================================================
# CUHK-X —— 辅助 3D 位姿头（"3D 骨架加强 2D 视频"，研究推荐 Top1）
# 机制：视频模型动作 CE + λ·MPJPE(回归同相机对齐的原生 3D 骨架) → 学人体几何、
#       把动作从外观解耦 → 提升 cross-subject。测试摘头零开销、模型大小不变。
# 对照：4ch 基线 fold0 = 0.6609；λ 从 0.1 起，可选扫 0.05/0.2
# 用法: sbatch scripts/auxpose.sbatch
# ============================================================
#SBATCH -J cuhkx-auxpose
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
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }

# ============ v2 位姿头（layer3 空间细节 + 小 λ），扫 λ=0.05 / 0.02 ============
echo "######## 辅助 3D 位姿头 v2 λ=0.05（layer3）fold0 60ep ########"
python scripts/train_auxpose.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_train.json --lambda_pose 0.05 --pose_layer 3 \
  --label_smoothing 0.1 --aug_strength 2 \
  --save_dir outputs/auxpose/v2_l05 --workers 8
echo "######## λ=0.05 完成 ########"

echo "######## 辅助 3D 位姿头 v2 λ=0.02（layer3）fold0 60ep ########"
python scripts/train_auxpose.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_train.json --lambda_pose 0.02 --pose_layer 3 \
  --label_smoothing 0.1 --aug_strength 2 \
  --save_dir outputs/auxpose/v2_l02 --workers 8
echo "######## λ=0.02 完成 ########"

echo "==== AUXPOSE v2 DONE（对比 4ch 基线 fold0 0.6609）===="
