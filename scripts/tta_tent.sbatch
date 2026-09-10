#!/bin/bash
# ============================================================
# CUHK-X Iter5: TENT 完全测试时自适应(熵最小化)
#   fold 代理(main fold0/1/2): no-adapt vs TENT 200步 → 离线判定
#   真测试出 CSV: main / thermal TENT 后 TTA
# 用法: sbatch scripts/tta_tent.sbatch
# ============================================================
#SBATCH -J cuhkx-tent
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter5 TENT ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "===== [A] fold 代理 main ====="
for F in 0 1 2; do
  echo "----- fold$F -----"
  python scripts/tta_tent.py --modality main --mode fold --fold "$F" --flip --steps 200 || echo "⚠️ fold$F"
done

echo "===== [B] 真测试 TENT 出 CSV ====="
python scripts/tta_tent.py --modality main --mode test --flip --steps 300 \
  --output outputs/sub_G_main_tent.csv || echo "⚠️ test main"
python scripts/tta_tent.py --modality thermal --mode test --flip --steps 300 \
  --output outputs/sub_G_thermal_tent.csv || echo "⚠️ test thermal"

echo ""
echo "==== Iter5 DONE: 待 LB ===="
ls -la outputs/sub_G_*.csv 2>/dev/null
