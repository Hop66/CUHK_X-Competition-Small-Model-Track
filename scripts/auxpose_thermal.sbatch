#!/bin/bash
# ============================================================
# CUHK-X —— 辅助 3D 位姿头注入 thermal（"3D 骨架几何监督热像"）
# 机制：thermal 视频模型动作 CE + λ·MPJPE(回归跨相机比例对齐的原生 3D 骨架)
#   → 补 thermal 缺的 3D 几何（main 的 Depth 已含深度几何，骨架冗余 → 不注入 main）
#   测试摘头零开销、模型不变。
# 对照：thermal 基线 fold0 = 0.6339（Kinetics 归一化修复后）；3 折 mean 0.6053
# 对齐：跨相机按帧比例（global time 同步）；骨架缺失帧 mask=0
# 用法: sbatch scripts/auxpose_thermal.sbatch
# ============================================================
#SBATCH -J cuhkx-auxpose-th
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
[ -f bbox_thermal_train.json ] || { echo "❌ 缺 bbox_thermal_train.json"; exit 1; }

# ============ thermal 位姿头（layer3 空间细节 + 小 λ），扫 λ=0.05 / 0.02 ============
echo "######## thermal 辅助 3D 位姿头 λ=0.05（layer3）fold0 60ep ########"
python scripts/train_auxpose.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --modality thermal \
  --lambda_pose 0.05 --pose_layer 3 \
  --label_smoothing 0.1 --aug_strength 2 \
  --save_dir outputs/auxpose_th/v2_l05 --workers 8
echo "######## λ=0.05 完成 ########"

echo "######## thermal 辅助 3D 位姿头 λ=0.02（layer3）fold0 60ep ########"
python scripts/train_auxpose.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --modality thermal \
  --lambda_pose 0.02 --pose_layer 3 \
  --label_smoothing 0.1 --aug_strength 2 \
  --save_dir outputs/auxpose_th/v2_l02 --workers 8
echo "######## λ=0.02 完成 ########"

echo "==== AUXPOSE THERMAL v2 DONE（对比 thermal 基线 fold0 0.6339；main_val≥0.64 则骨架几何监督有效）===="
