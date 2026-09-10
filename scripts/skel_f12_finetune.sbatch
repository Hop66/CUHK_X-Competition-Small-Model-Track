#!/bin/bash
#SBATCH -J cuhkx-skel-f12
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/skeleton_finetune
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
[ -f outputs/ntu_pretrain/ntu_pretrained_backbone.pth ] || { echo "缺 NTU ckpt"; exit 1; }
export HF_ENDPOINT=https://hf-mirror.com
for fi in 1 2; do
  echo "==== skeleton finetune fold$fi (同一 NTU backbone, 与 fold0 可比) ===="
  python -u scripts/train_ntu_finetune.py --fold $fi --folds 3 \
    --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth \
    --epochs 60 --batch_size 64 --num_frames 24 \
    --save_dir outputs/skeleton_finetune --workers 8 2>&1 | tail -2
done
echo DONE
