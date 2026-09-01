#!/bin/bash
# ============================================================
# CUHK-X —— Thermal 冲提交级（基于 P1 已验证 recipe：+5.4pt → fold0 0.6190）
# 顺序：
#   [1] Thermal 3 折 60ep（确认 recipe 稳定，mean 应 ~0.61-0.62）
#   [2] Thermal 全量 2-seed（seed42 + seed777）80ep → 提交级 thermal 模型
# 之后：quantize_pack 每 seed 独立 int5 打包 + ensemble_inference 与 main 动态集成
# 用法: sbatch scripts/thermal_push.sbatch
# ============================================================
#SBATCH -J cuhkx-thermal-push
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
[ -f bbox_thermal_train.json ] || { echo "❌ 缺 bbox_thermal_train.json（先跑 multimodal_pipeline.sbatch 或 detect.py）"; exit 1; }

# ============ [1/3] Thermal 3 折确认（fold0 已 0.6190）============
echo "######## [1/3] Thermal 全套 recipe 3 折 60ep ########"
python scripts/train_step1.py \
  --modality thermal --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --folds 3 \
  --crop_cache bbox_thermal_train.json --aug_strength 2 --label_smoothing 0.1 \
  --save_dir outputs/thermal_v2 --workers 8
echo "######## [1/3] Thermal 3 折完成（看 mean，预期 ~0.61-0.62）########"

# ============ [2/3] Thermal 全量 seed42 ============
echo "######## [2/3] Thermal 全量 seed42 80ep ########"
python scripts/train_step1.py \
  --modality thermal --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 80 --full --seed 42 \
  --crop_cache bbox_thermal_train.json --aug_strength 2 --label_smoothing 0.1 \
  --save_dir outputs/thermal_full --workers 8
echo "######## [2/3] seed42 完成 -> outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth ########"

# ============ [3/3] Thermal 全量 seed777 ============
echo "######## [3/3] Thermal 全量 seed777 80ep ########"
python scripts/train_step1.py \
  --modality thermal --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 80 --full --seed 777 \
  --crop_cache bbox_thermal_train.json --aug_strength 2 --label_smoothing 0.1 \
  --save_dir outputs/thermal_full --workers 8
echo "######## [3/3] seed777 完成 -> outputs/thermal_full/r2plus1d34_thermal_full_seed777.pth ########"

echo "==== THERMAL PUSH DONE（下一步：quantize_pack 每 seed int5 + 与 main 动态集成）===="
