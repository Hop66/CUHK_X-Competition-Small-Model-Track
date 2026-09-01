#!/bin/bash
# ============================================================
# CUHK-X —— main(教师)→thermal(学生) 软标签蒸馏（XTinyHAR 式跨模态蒸馏）
# 方向：强 main（冻结 0.66-0.71）教 弱 thermal（独立视角 0.62）
#   → thermal 学 main 的动作语义，突破自身 recipe 上限 → 独立视角更强
# 对照：thermal 全套 recipe fold0 = 0.6190；目标 0.65+
# 用法: sbatch scripts/distill_thermal.sbatch
# ============================================================
#SBATCH -J cuhkx-distill-th
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
TEACHER=outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth
[ -f "$TEACHER" ] || { echo "❌ 缺教师 ckpt $TEACHER（换 --teacher_ckpt）"; exit 1; }

# ============ main→thermal 蒸馏 fold0（λ_kl=1.0, T=4）============
echo "######## main→thermal 蒸馏 fold0 60ep（λ_kl=1.0 T=4）########"
python scripts/train_distill_thermal.py \
  --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --teacher_ckpt "$TEACHER" \
  --lr 1e-4 --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_train.json --thermal_crop bbox_thermal_train.json \
  --lambda_kl 1.0 --T 4.0 --label_smoothing 0.1 --aug_strength 2 \
  --save_dir outputs/distill_thermal --workers 8
echo "######## 蒸馏完成（对比 thermal recipe fold0 0.6190，目标 0.65+）########"

echo "==== DISTILL THERMAL DONE ===="
