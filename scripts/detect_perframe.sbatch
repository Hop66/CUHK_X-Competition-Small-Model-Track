#!/bin/bash
# ============================================================
# CUHK-X —— 逐帧 person bbox 检测（先跑这个，拿到 track_crop 数据源 + 可视化的真框）
#   [1] thermal:  bbox_thermal_perframe.json（供 thermal 2D 帧级 / viz_track）
#   [2] main/IR : bbox_ir_perframe.json （供 main 3D track_crop / viz_track）
# 产出后：
#   - 可视化: python scripts/viz_track_boxes.py --frame_dir <clip> --mode cache --box <json> --clip_key <k>
#   - 训练  : sbatch scripts/main_track_crop.sbatch / thermal_track_crop.sbatch / 2d_framevote
# margin=1.1 紧框（人物满框、背景≈0）；逐帧遍历（对每个 clip 的全部帧，较耗时）
# 用法: sbatch scripts/detect_perframe.sbatch
# ============================================================
#SBATCH -J cuhkx-detect-perframe
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
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "########## [1/2] thermal 逐帧 bbox（margin1.1 紧）##########"
python -m src.detect --mode train --out bbox_thermal_perframe.json \
  --modality thermal --margin 1.1 --per_frame \
  || echo "⚠️ thermal per_frame 失败"

echo "########## [2/2] main/IR 逐帧 bbox（margin1.1 紧）##########"
python -m src.detect --mode train --out bbox_ir_perframe.json \
  --modality depthir --margin 1.1 --per_frame \
  || echo "⚠️ IR per_frame 失败"

echo "==== DETECT PERFRAME DONE ===="
ls -lh bbox_thermal_perframe.json bbox_ir_perframe.json 2>/dev/null || true
echo "下一步: 1) viz_track_boxes.py 可视化验框 → 2) sbatch main_track_crop / thermal_track_crop / 2d_framevote"
