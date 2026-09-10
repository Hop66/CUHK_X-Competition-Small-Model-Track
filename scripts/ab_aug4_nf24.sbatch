#!/bin/bash
#SBATCH -J cuhkx-aug4ab
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== A. main aug4(域鲁棒增强) fold0 AB, 对照 aug2=0.6695 ===="
python -u scripts/train_main_baseline.py --fold 0 --weights ig65m_r2plus1d34.pth \
  --aug_strength 4 --epochs 60 --batch_size 32 --workers 4 --save_dir outputs/main_aug4 \
  2>&1 | tail -3
echo "==== B. main nf24(24帧) fold0 AB, 对照 aug1=16帧 0.6695(时间尺度多样性) ===="
python -u scripts/train_main_baseline.py --fold 0 --weights ig65m_r2plus1d34.pth \
  --num_frames 24 --aug_strength 1 --epochs 60 --batch_size 32 --workers 4 --save_dir outputs/main_nf24 \
  2>&1 | tail -3
echo "== AB DONE =="
