#!/bin/bash
#SBATCH -J cuhkx-fullaug
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/main_aug4 outputs/main_nf24 outputs/pack
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== A. aug4 全折(3-fold) ===="
python scripts/train_main_baseline.py --fold -1 --weights ig65m_r2plus1d34.pth \
  --aug_strength 4 --epochs 60 --batch_size 32 --save_dir outputs/main_aug4 2>&1 | tail -4
echo "==== B. nf24 全折 ===="
python scripts/train_main_baseline.py --fold -1 --weights ig65m_r2plus1d34.pth \
  --num_frames 24 --aug_strength 1 --epochs 60 --batch_size 32 --save_dir outputs/main_nf24 2>&1 | tail -4
echo "==== 打包 int4(fold0/1/2) ===="
for nm in main_aug4 main_nf24; do
  for fi in 0 1 2; do
    ck="outputs/$nm/baseline_aug4_fold${fi}.pth"
    [ "$nm" = main_nf24 ] && ck="outputs/$nm/baseline_aug1_fold${fi}.pth"
    [ -f "$ck" ] && python scripts/quantize_pack.py --name "${nm}_f${fi}" --checkpoints "$ck" --bits 4 --out_dir outputs/pack | tail -1
  done
done
echo "== FULL DONE =="
