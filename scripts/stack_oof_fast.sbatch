#!/bin/bash
#SBATCH -J cuhkx-stack-oof
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) | fast-lane OOF prep ===="
python scripts/extract_oof_logits.py --main --mode oof --flip \
  --fold_ckpts outputs/main_baseline/baseline_aug2_fold0.pth outputs/main_baseline/baseline_aug2_fold1.pth outputs/main_baseline/baseline_aug2_fold2.pth \
  --out outputs/oof/main_oof.pkl
python scripts/extract_oof_logits.py --main --mode test --flip \
  --fold_ckpts outputs/main_full/r2plus1d34_depthir_full_seed42.pth --out outputs/oof/main_test.pkl
python scripts/extract_oof_logits.py --thermal --mode test --flip \
  --ckpts outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth --out outputs/oof/thermal_test.pkl 2>/dev/null || \
python scripts/extract_oof_logits.py --thermal --mode test --flip \
  --fold_ckpts outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth --out outputs/oof/thermal_test.pkl
echo "==== OOF fast-lane DONE ===="
ls -la outputs/oof/
