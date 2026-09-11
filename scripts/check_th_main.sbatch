#!/bin/bash
#SBATCH -J cuhkx-thmain
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
# 检验: th 是否拖累 main —— 真锚协议(fold0 s42 + flip_tta + crop) fold0 val OOF
python -u scripts/check_th_main_inprotocol.py --flip 2>&1 | tail -12
echo DONE
