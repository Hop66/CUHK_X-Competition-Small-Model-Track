#!/bin/bash
# ============================================================
# CUHK-X —— FrameNet 轻量逐帧基线（14th-place 移植）fold0
# 验证 per-frame logits mean 是否关键：从零 2D CNN + 帧级 logits 平均
#   vs thermal 基线 fold0=0.6339（R2+1D-34+IG65M+时序 GAP）
# 变体（同协议 subject fold，全部 gray_norm + segment，贴近 notebook）：
#   [1] 8 帧 112px（notebook 原样配置）
#   [2] 16 帧 112px（帧数对齐我们）

   # 判读：≥0.60 → per-frame 投票层信息接近强基线 → FrameNet(~5M) 可作轻量独立视角入 0.73 集成
   #        ≥0.6339 或接近 → 帧级池化确为关键载体 → 考虑移植逐帧池化到强模型
   #        <0.55 → 单帧外观不足，3D 时序不可替代，per-frame 非关键
   # 用法: sbatch scripts/thermal_framenet.sbatch
   # ============================================================
#SBATCH -J cuhkx-framenet
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/thermal_framenet
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "########## [1/2] FrameNet 8 帧 112px（notebook 配置）##########"
python scripts/train_framenet_thermal.py --fold 0 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --save_dir outputs/thermal_framenet --workers 8 \
  || echo "⚠️ [1] FrameNet(8) 失败"

echo "########## [2/2] FrameNet 16 帧 112px（帧数对齐我们，隔离帧数变量）##########"
python scripts/train_framenet_thermal.py --fold 0 \
  --num_frames 16 --size 112 --sample_mode segment --gray_norm \
  --save_dir outputs/thermal_framenet --workers 8 \
  || echo "⚠️ [2] FrameNet(16) 失败"

echo "==== FRAMENET DONE（判读：fold0 vs 0.6339；≥0.60 说明帧级投票层信息够 → 轻量视角入 0.73）===="
