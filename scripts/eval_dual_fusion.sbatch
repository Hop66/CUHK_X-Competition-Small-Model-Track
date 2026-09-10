#!/bin/bash
#SBATCH -J cuhkx-dualeval
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
[ -f outputs/dual_fusion/dualfusion_fold0.pth ] || { echo "缺 ckpt"; exit 1; }
python scripts/eval_dual_fusion.py --ckpt outputs/dual_fusion/dualfusion_fold0.pth --workers 2
echo "== DONE =="
