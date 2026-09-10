#!/bin/bash
#SBATCH -J cuhkx-auxhp
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/main_aux
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== L2-HARD: main nf16 fold0 + 难对骨架软判别aux(推理纯相机) 对照 main16f=0.6695 =="
python -u scripts/train_main_aux.py --fold 0 --epochs 60 --aux hardpairs --lambda_aux 1.0 2>&1 | tail -6
echo "== DONE =="
