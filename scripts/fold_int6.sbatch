#!/bin/bash
#SBATCH -J cuhkx-foldint6
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
python -u scripts/eval_pack_fold0.py --workers 3 \
  --packs main_a2f0_i6_fold0_int6 \
  --out outputs/oof/foldint6_fold0.pkl
echo DONE
