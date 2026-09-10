#!/bin/bash
#SBATCH -J cuhkx-aug4eval
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== 收数: main aug4 fold0 ckpt(对照 aug2=0.6695) =="
python -u scripts/eval_main_fold0.py --ckpt outputs/main_aug4/baseline_aug4_fold0.pth \
  --num_frames 16 --fold 0 --batch_size 16 --workers 4 2>&1 | tail -4
echo DONE
