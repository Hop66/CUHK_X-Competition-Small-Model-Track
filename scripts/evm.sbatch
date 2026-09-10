#!/bin/bash
#SBATCH -J cuhkx-evmain
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
python -u scripts/eval_main_fold0.py --ckpt outputs/main_nf32/r2plus1d34_depthir_fold0.pth \
  --num_frames 32 --fold 0 --workers 4
echo DONE
