#!/bin/bash
#SBATCH -J cuhkx-feat512
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
# 提取 main encoder 512d 特征 (OOF 3折 + test) - 与 main_oof.pkl 同批模型 baseline_aug2
python -u scripts/extract_oof_logits.py --main --mode oof \
  --fold_ckpts outputs/main_baseline/baseline_aug2_fold0.pth outputs/main_baseline/baseline_aug2_fold1.pth outputs/main_baseline/baseline_aug2_fold2.pth \
  --num_frames 16 --size 128 --crop bbox_train.json \
  --out outputs/oof/main_oof.pkl \
  --save_feats outputs/oof/main_feats512.pkl 2>&1 | tail -8
echo OOF-DONE
# test 512d (用 fold0 模型, 同 save_anchor_soft 协议)
python -u scripts/extract_oof_logits.py --main --mode test \
  --fold_ckpts outputs/main_baseline/baseline_aug2_fold0.pth \
  --num_frames 16 --size 128 \
  --out outputs/oof/main_src_test.pkl \
  --save_feats outputs/oof/main_src_feats512.pkl 2>&1 | tail -6
echo TEST-DONE
