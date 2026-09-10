#!/bin/bash
#SBATCH -J cuhkx-skel-fullcheck
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
for fi in 0 1 2; do
  echo "== FULL 骨架 fold$fi val OOF (核对 full 质量 vs fold模型) =="
  python -u scripts/extract_skel_oof.py --fold $fi \
    --ckpt outputs/skeleton_finetune_full/finetune_full.pth \
    --out outputs/oof/skel_full_fold$fi.pkl --num_frames 24 2>&1 | tail -1
done
echo DONE
