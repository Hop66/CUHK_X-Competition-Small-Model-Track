#!/bin/bash
# ============================================================
# CUHK-X —— P1 Thermal 全套 recipe 升级 + P2 main↔skeleton 对比学习（统一提交）
# 顺序：
#   [1] 生成 thermal YOLO crop（bbox_thermal_train.json）
#   [2] P1: Thermal 全套 recipe（crop + balanced + aug2 + ls0.1 + Kinetics 归一化修复）fold0 60ep
#       → 对比旧 thermal fold0 0.5646，目标 0.65+
#   [3] P2: main↔skeleton 对比学习 fold0 60ep（CE(main)+λInfoNCE，只提升 main）
#       → 对比 4ch 基线 fold0 0.6609
# 用法: sbatch scripts/multimodal_pipeline.sbatch
# ============================================================
#SBATCH -J cuhkx-multimodal
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

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }

# ============ [1/3] thermal YOLO crop（首次生成后缓存）============
echo "######## [1/3] 生成 thermal crop（bbox_thermal_train.json）########"
[ -f bbox_thermal_train.json ] || python -m src.detect \
  --train_root "$HOME/Multimodal/data/Training/HAR" \
  --out bbox_thermal_train.json --modality thermal --margin 1.4
echo "######## [1/3] thermal crop 完成 ########"

# ============ [2/3] P1 Thermal 全套 recipe fold0 ============
echo "######## [2/3] Thermal 全套 recipe（crop+balanced+aug2+ls0.1+Kinetics归一化）fold0 60ep ########"
python scripts/train_step1.py \
  --modality thermal --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --aug_strength 2 --label_smoothing 0.1 \
  --save_dir outputs/thermal_v2 --workers 8
echo "######## [2/3] Thermal 完成（对比旧 fold0 0.5646，目标 0.65+）########"

# ============ [3/3] P2 main↔skeleton 对比学习 fold0 ============
echo "######## [3/3] main↔skeleton 对比学习 fold0 60ep ########"
python scripts/train_contrastive.py \
  --aux skeleton --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --lambda_contrast 0.1 --temperature 0.07 --label_smoothing 0.1 --aug_strength 2 \
  --save_dir outputs/contrast_skel --crop_cache bbox_train.json --workers 8
echo "######## [3/3] 对比学习完成（对比 4ch 基线 fold0 0.6609）########"

echo "==== MULTIMODAL PIPELINE DONE ===="
