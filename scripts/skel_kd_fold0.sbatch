#!/bin/bash
#SBATCH -J cuhkx-skelkd
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/skeleton_kd
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
[ -f outputs/ntu_pretrain/ntu_pretrained_backbone.pth ] || { echo "缺 NTU ckpt"; exit 1; }
echo "== 骨架 KD 蒸馏 fold0 (teacher=main+th soft, lam=0.5) =="
python -u scripts/train_ntu_kd.py --fold 0 \
  --ntu_ckpt outputs/ntu_pretrain/ntu_pretrained_backbone.pth \
  --teacher outputs/teacher/teacher_main_train.pkl \
  --teacher2 outputs/teacher/teacher_thermal_train.pkl \
  --kd_lambda 0.5 --epochs 60 --batch_size 64 --num_frames 24 \
  --out outputs/skeleton_kd/skeleton_kd_fold0.pth --workers 8 2>&1 | tail -3
echo DONE
