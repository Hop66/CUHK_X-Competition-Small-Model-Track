#!/bin/bash
#SBATCH -J cuhkx-aug4loc
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
LOCAL=/tmp/har_${USER##*/}
mkdir -p "$LOCAL"
if [ ! -d "$LOCAL/Training/HAR" ]; then
  echo "== 同步数据到本地盘(tar) $(date +%T) =="
  rm -rf "$LOCAL"; mkdir -p "$LOCAL"
  tar -C data -cf - Training | tar -C "$LOCAL" -xf -
  echo "== 同步完成 $(date +%T) =="
else
  echo "== 本地缓存已存在 $LOCAL =="
fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
echo "==== A. main aug4(域鲁棒增强) fold0 AB (本地数据) ===="
python -u scripts/train_main_baseline.py --fold 0 --weights ig65m_r2plus1d34.pth \
  --train_root "$LOCAL/Training/HAR" --aug_strength 4 --epochs 60 --batch_size 32 \
  --workers 4 --save_dir outputs/main_aug4 2>&1 | tail -4
echo "==== B. main nf24(24帧) fold0 AB (本地数据) ===="
python -u scripts/train_main_baseline.py --fold 0 --weights ig65m_r2plus1d34.pth \
  --train_root "$LOCAL/Training/HAR" --num_frames 24 --aug_strength 1 --epochs 60 \
  --batch_size 32 --workers 4 --save_dir outputs/main_nf24 2>&1 | tail -4
echo "== AB LOCAL DONE =="
