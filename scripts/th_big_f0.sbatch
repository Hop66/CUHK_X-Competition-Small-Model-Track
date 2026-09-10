#!/bin/bash
#SBATCH -J cuhkx-big-th
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/th_big_f0
LOCAL=/tmp/th_${USER##*/}
if [ ! -d "$LOCAL/Thermal/0_Wash_face" ]; then
  echo "== 同步 Thermal 本地 $(date +%T) =="
  rm -rf "$LOCAL"; mkdir -p "$LOCAL"
  tar -C data/Training/HAR -cf - Thermal | tar -C "$LOCAL" -xf -
  echo "== 同步完成 $(date +%T) =="
fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== 🚀 BIG-1: thermal nf32 大增强(size160+segment+aug3+120ep+clip40) fold0 =="
python -u scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 32 --size 160 --batch_size 10 --workers 8 --lr 8e-5 --epochs 120 \
  --label_smoothing 0.1 --aug_strength 3 --crop_cache bbox_thermal_train.json \
  --sample_mode segment --grad_clip 40 \
  --modality thermal --train_root "$LOCAL" --save_dir outputs/th_big_f0 2>&1
echo DONE
