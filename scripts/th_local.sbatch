#!/bin/bash
#SBATCH -J cuhkx-thlocal
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
LOCAL=/tmp/th_${USER##*/}
if [ ! -d "$LOCAL/Thermal/0_Wash_face" ]; then
  echo "== 同步 Thermal 到本地 $(date +%T) =="
  rm -rf "$LOCAL"; mkdir -p "$LOCAL"
  tar -C data/Training/HAR -cf - Thermal | tar -C "$LOCAL" -xf -
  echo "== 同步完成 $(date +%T) =="
else
  echo "== Thermal 本地缓存已存在: $LOCAL =="
fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== A. thermal R2+1D nf32 fold0 (本地) =="
python -u scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --fold 0 --num_frames 32 --size 128 --batch_size 16 --workers 8 --lr 1e-4 --epochs 60 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json \
  --modality thermal --train_root "$LOCAL" --save_dir outputs/th_nf32 2>&1 | tail -4
echo "== B. thermal GRU+Attn fold0 (本地) =="
python -u scripts/train_thermal_gru_attn.py --fold 0 --epochs 60 --batch_size 32 --workers 8 \
  --train_root "$LOCAL" --save_dir outputs/th_gru 2>&1 | tail -4
echo "== TH-LOCAL DONE =="
