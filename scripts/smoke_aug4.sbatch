#!/bin/bash
#SBATCH -J cuhkx-smokeaug
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== smoke: aug4 fold0 3ep =="
python scripts/train_main_baseline.py --fold 0 --weights ig65m_r2plus1d34.pth \
  --aug_strength 4 --epochs 3 --batch_size 32 --save_dir outputs/main_aug4_smoke 2>&1 | tail -6
echo SMOKE-DONE
