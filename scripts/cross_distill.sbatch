#!/bin/bash
#SBATCH -J cuhkx-cdistill
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/th_cross
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
echo "==== cross-modal 蒸馏 thermal student ← main teacher (fold0, 对照 thermal 0.5966) ===="
python -u scripts/train_distill_stream.py --modality thermal \
  --teacher outputs/teacher/teacher_main_train.pkl \
  --crop bbox_thermal_train.json --fold 0 --epochs 50 --batch_size 32 \
  --workers 4 --lr 1e-4 --distill_w 0.5 --T 3.0 \
  --save_dir outputs/th_cross --seed 42 2>&1 | tail -5
echo "== CROSS-DISTILL DONE =="
