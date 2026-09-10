#!/bin/bash
#SBATCH -J cuhkx-mxema-m
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/main_mixema_f0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
echo "== main nf16 MixUp+EMA fold0 (对照 aug2 fold0=0.6695) =="
python -u scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 16 --size 128 --batch_size 24 --workers 8 --lr 1e-4 --epochs 55 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_train.json \
  --mixup_alpha 0.2 --ema_decay 0.999 \
  --modality depthir --save_dir outputs/main_mixema_f0 2>&1 | tail -3
echo DONE
