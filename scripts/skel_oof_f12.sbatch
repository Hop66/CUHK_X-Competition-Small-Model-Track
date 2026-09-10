#!/bin/bash
#SBATCH -J cuhkx-skeloof12
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
for fi in 1 2; do
  echo "== 骨架 finetune fold$fi OOF (nf24) =="
  python -u scripts/extract_skel_oof.py --fold $fi \
    --ckpt outputs/skeleton_finetune/finetune_fold$fi.pth \
    --out outputs/oof/skel_fold$fi.pkl --num_frames 24 2>&1 | tail -1
done
echo "== KD fold0 OOF (nf24, 对比 native) =="
python -u scripts/extract_skel_oof.py --fold 0 \
  --ckpt outputs/skeleton_kd/skeleton_kd_fold0.pth \
  --out outputs/oof/skel_kd_fold0.pkl --num_frames 24 2>&1 | tail -1
echo DONE
