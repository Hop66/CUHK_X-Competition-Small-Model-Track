#!/bin/bash
#SBATCH -J cuhkx-skeloof
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== 骨架(MotionBert finetune, nf24) fold0 OOF logits 提取 =="
python -u scripts/extract_skel_oof.py --ckpt outputs/skeleton_finetune/finetune_fold0.pth \
  --out outputs/oof/skel_fold0.pkl --num_frames 24 2>&1 | tail -4
echo DONE
