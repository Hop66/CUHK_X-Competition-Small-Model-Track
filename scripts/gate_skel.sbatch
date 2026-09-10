#!/bin/bash
#SBATCH -J cuhkx-gatesk
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/main_gate_skel
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== 骨架特征门控融合(main+skel-BiGRU+gate) fold0 对照 main16f=0.6695 =="
python -u scripts/train_gate_fuse.py --fold 0 --epochs 60 2>&1 | tail -4
echo "== DONE =="
