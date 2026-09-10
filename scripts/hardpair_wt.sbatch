#!/bin/bash
#SBATCH -J cuhkx-hpwt
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/main_hpwt
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== 难对重权训练: main nf16 fold0 + 难对区(骨架难对选择器) CE加权 ×2 | 对照 main16f=0.6695 / 难对区50.96% =="
python -u scripts/train_step1.py \
  --backbone r2plus1d --weights ig65m_r2plus1d34.pth \
  --num_frames 16 --epochs 60 --batch_size 16 --size 128 \
  --crop_cache bbox_train.json --aug_strength 2 --label_smoothing 0.1 \
  --hardpair_w 2.0 --fold 0 \
  --save_dir outputs/main_hpwt 2>&1 | tail -8
echo "== DONE =="
