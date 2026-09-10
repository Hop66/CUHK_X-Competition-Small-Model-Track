#!/bin/bash
# ============================================================
# CUHK-X Iter13: thermal BiGRU 复现（其他队 thermal-specialist-v2）
#   arch: per-frame ResNet ←→ BiGRU(2层) + attention; 灰度归一 (x-0.5)/0.25; 112px
#   同协议 subject-3 折验证, 供与 R2+1D-thermal 对比(R2+1D Kinet fold0=0.5966; gray 见53524)
# 用法: sbatch scripts/ab_thermal_bigru.sbatch
# ============================================================
#SBATCH -J cuhkx-bigru
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/th_bigru
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter13 BiGRU-thermal ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_thermal_bigru.py --folds 0 1 2 --epochs 60 --lr 3e-4 \
  --batch_size 32 --workers 8 --save_dir outputs/th_bigru 2>&1 | tee logs/th_bigru121.log

echo ""
echo "==== Iter13 DONE ===="
ls -la outputs/th_bigru/*.pth 2>/dev/null
echo "判读: mean fold best > R2+1D-thermal(Kinet 0.5966 / gray 53524-T1) → 采用 BiGRU 替换/集成 thermal"
