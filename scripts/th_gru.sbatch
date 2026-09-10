#!/bin/bash
#SBATCH -J cuhkx-thgru
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/th_gru
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
echo "== ThermalGRU(ResNet2D+BiGRU+Attn) fold0 AB, 对照 R2+1D-th 0.5966 =="
python -u scripts/train_thermal_gru_attn.py --fold 0 --epochs 60 --batch_size 32 --workers 4 \
  --save_dir outputs/th_gru 2>&1 | tail -4
echo "== DONE =="
