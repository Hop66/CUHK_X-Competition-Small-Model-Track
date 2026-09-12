#!/bin/bash
#SBATCH -J cuhkx-p1frz
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
echo "== P0 conditional residual fusion fold0 (env-balanced) =="
python scripts/train_cond_residual.py --fold 0 --epochs 60 --freeze_enc 0.8 --mode concat 2>&1 | tail -6
echo DONE
