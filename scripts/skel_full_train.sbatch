#!/bin/bash
#SBATCH -J cuhkx-skel-full
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/skeleton_finetune_full
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
[ -f outputs/ntu_pretrain/ntu_pretrained_backbone.pth ] || { echo "缺 NTU ckpt"; exit 1; }
echo "== 骨架 full 全量训练 (nf24, 供 3成员链 test 推理) =="
python -u scripts/train_ntu_finetune_full.py \
  --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth \
  --epochs 40 --batch_size 64 --num_frames 24 --workers 8 \
  --save_dir outputs/skeleton_finetune_full 2>&1 | tail -3
echo DONE
