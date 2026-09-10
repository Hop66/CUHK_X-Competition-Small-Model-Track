#!/bin/bash
#SBATCH -J cuhkx-trans-f2
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/d_trans
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) | trans-soft fold2 第三数据点 ===="
TM=outputs/teacher/teacher_main_train.pkl
[ -f "$TM" ] || { echo "缺教师"; exit 1; }
for TAG in p t; do
  EXTRA=""; [ "$TAG" = p ] && EXTRA="--no_trans"
  echo "===== fold2 ${TAG}($EXTRA) ====="
  python scripts/train_distill_transductive.py --modality main --teacher "$TM" \
    --fold 2 --epochs 40 --batch_size 16 --tag "$TAG" --save_dir outputs/d_trans \
    $EXTRA 2>&1 | tail -2 || echo "⚠$TAG"
done
echo "==== DONE: fold2 对比 ===="
grep -E "fold2|best=" logs/cuhkx-trans-f2-*.out | tail -6
