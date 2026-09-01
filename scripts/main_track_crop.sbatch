#!/bin/bash
# ============================================================
# CUHK-X —— main(3D) 每帧跟人裁剪 + 位移/尺度轨迹通道（用户设想的主战场）
# 思想：固定 clip 框(1.4 square)高瘦人体 clamp 帧高→左右背景多；整 clip 单框不跟随时序位移
#       → 逐帧 YOLO 跟人剪到满框（姿态最大化、背景≈0、3D 时序不被位移稀释）
#       + 每帧 [cx,cy,bw,bh] 轨迹通道（位移/走近走远显式回补，R2Plus1D34 head 前 concat）
# 对照：main 基线 fold0 = 0.6609（R2+1D-34 + IG65M + 固定窗口 1.4）
#   [1] detect --per_frame --modality depthir → bbox_ir_perframe.json（IR 帧检测，cpu）
#   [2] r2plus1d34 + track_crop（逐帧跟人 + 轨迹） else 固定框
# 判读：[2] vs 0.6609 / [3] vs [2] = track_crop 净值
# 用法: sbatch scripts/main_track_crop.sbatch
# ============================================================
#SBATCH -J cuhkx-main-track
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_track
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CROP=bbox_train.json
[ -f "$CROP" ] || { echo "⚠️ 缺 $CROP（main 主线 YOLO 框）"; CROP=""; }
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }
EP=60

echo "########## [1/3] detect --per_frame --modality depthir（IR 帧逐帧 bbox）##########"
PF=bbox_ir_perframe.json
if [ ! -f "$PF" ]; then
  python -m src.detect --mode train --out "$PF" --modality depthir --margin 1.1 --per_frame \
    || echo "⚠️ detect per_frame(depthir) 失败"
fi
[ -f "$PF" ] || PF=""

echo "########## [2/3] r2plus1d34 + track_crop（逐帧跟人 + 轨迹通道，main 3D）##########"
if [ -n "$PF" ]; then
  python scripts/train_backbone_compare.py --backbone r2plus1d34 --modality depthir \
    --crop_cache "$CROP" --weights ig65m_r2plus1d34.pth \
    --track_crop --box_path "$PF" --sample_mode uniform \
    --epochs $EP --fold 0 --size 128 \
    --save_dir outputs/main_track --workers 8 \
    || echo "⚠️ [2] main track_crop 失败"
else
  echo "⚠️ [2] 跳过（逐帧 bbox 生成失败）"
fi

echo "########## [3/3] 对照：r2plus1d34 + 固定 clip 框（main 0.6609 协议复核）##########"
python scripts/train_backbone_compare.py --backbone r2plus1d34 --modality depthir \
  --crop_cache "$CROP" --weights ig65m_r2plus1d34.pth \
  --sample_mode uniform --epochs $EP --fold 0 --size 128 \
  --save_dir outputs/main_track --workers 8 \
  || echo "⚠️ [3] 固定框对照失败"

echo "==== MAIN TRACK CROP DONE ===="
echo "判读：[2] vs 0.6609 / [3] = 逐帧跟人+轨迹 在 main 3D 的净值"
echo "若 [2] > [3] 或 > 0.6609 → 方法有效 → 全量重训 main → 与 thermal(2D per-frame) 融合冲 0.8"
