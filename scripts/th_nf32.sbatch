#!/bin/bash
#SBATCH -J cuhkx-thnf32
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/th_nf32
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
echo "== thermal nf32 fold0 AB (对标 14th 0.8 thermal), 对照 16f=0.5966/24f=0.6030 =="
python -u scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 32 --size 128 --batch_size 16 --workers 8 --lr 1e-4 --epochs 60 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json \
  --modality thermal --save_dir outputs/th_nf32 2>&1 | tail -4
echo "== DONE =="
