#!/bin/bash
# ============================================================
# CUHK-X —— thermal 每帧跟人裁剪 + 位移/尺度轨迹通道（用户设想落地）
# 思想：把「人物姿态细节」与「人物位移/尺度」显式解耦——
#   逐帧 YOLO 跟人剪到满框（姿态最大化、背景≈0、不切肢体）→ 2D per-frame resnet18 学姿态；
#   每帧人物中心[ cx, cy ] + 框尺寸[ bw, bh ]作为显式轨迹通道喂模型（走动/走近走远不丢）。
# 对照基线 thermal fold0 = 0.6339；对照组 = thermal_2d_framevote[1]（resnet18 固定 clip 框）。
#   [1] detect --per_frame 生成逐帧 bbox（cpu 1-2h）
#   [2] resnet18 + track_crop（逐帧跟人 8帧112 gray segment）
#   [3] 对照：resnet18 + 固定 clip 框（无需新 detect）
# 判读：[2] vs [3] → 逐帧跟人+轨迹 的净值（若 > 0 且整体高 → 方法有效 → 全量/移植到 3D 主线）
# 用法: sbatch scripts/thermal_track_crop.sbatch
# ============================================================
#SBATCH -J cuhkx-th-track
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
mkdir -p logs outputs/thermal_track
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "⚠️ 缺 $CROP"; exit 1; }
EP=64

echo "########## [1/3] detect --per_frame 逐帧 bbox（thermal, margin 1.1 紧框）##########"
PF=bbox_thermal_perframe.json
if [ ! -f "$PF" ]; then
  python -m src.detect --mode train --out "$PF" --modality thermal --margin 1.1 --per_frame \
    || echo "⚠️ detect per_frame 失败"
fi
[ -f "$PF" ] || PF=""

echo "########## [2/3] resnet18 + track_crop（每帧跟人 + 轨迹通道）##########"
if [ -n "$PF" ]; then
  python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 --track_crop \
    --num_frames 8 --size 112 --sample_mode segment --gray_norm \
    --crop_cache "$CROP" --box_path "$PF" --epochs $EP \
    --save_dir outputs/thermal_track --workers 8 \
    || echo "⚠️ [2] track_crop 失败"
else
  echo "⚠️ [2] 跳过（逐帧 bbox 生成失败）"
fi

echo "########## [3/3] 对照：resnet18 + 固定 clip 框（复用 2dframevote 配置）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --crop_cache "$CROP" --epochs $EP \
  --save_dir outputs/thermal_track --workers 8 \
  || echo "⚠️ [3] 固定框对照失败"

echo "==== THERMAL TRACK CROP DONE ===="
echo "判读：[2] vs [3] = 逐帧跟人+轨迹 净值；两者 vs 0.6339 = 2D-per-frame 离线水平"
echo "若 [2] > [3] 且接近/超 0.6339 → 方法有效 → 全量/移植到 3D 主线（main 也可这么做）"
