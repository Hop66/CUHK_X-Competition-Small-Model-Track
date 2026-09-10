#!/bin/bash
#SBATCH -J cuhkx-long-th
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/th_long_f0
LOCAL=/tmp/th_${USER##*/}
if [ ! -d "$LOCAL/Thermal/0_Wash_face" ]; then
  echo "== 同步 Thermal 本地 $(date +%T) =="
  rm -rf "$LOCAL"; mkdir -p "$LOCAL"
  tar -C data/Training/HAR -cf - Thermal | tar -C "$LOCAL" -xf -
  echo "== 同步完成 $(date +%T) =="
fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== thermal nf32 配方对齐(100ep + grad_clip40, 对照 55ep 0.6478) =="
python -u scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 32 --size 128 --batch_size 12 --workers 8 --lr 1e-4 --epochs 100 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json \
  --grad_clip 40 \
  --modality thermal --train_root "$LOCAL" --save_dir outputs/th_long_f0 2>&1 | grep -E "best val|DONE" | tail -3
echo DONE
