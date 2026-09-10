#!/bin/bash
#SBATCH -J cuhkx-aug1-oof
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== aug1 main OOF（2-member 折级集成验证用） =="
python scripts/extract_oof_logits.py --main --mode oof --flip \
  --fold_ckpts outputs/main_baseline/baseline_aug1_fold0.pth outputs/main_baseline/baseline_aug1_fold1.pth outputs/main_baseline/baseline_aug1_fold2.pth \
  --out outputs/oof/main_oof_aug1.pkl
echo DONE
