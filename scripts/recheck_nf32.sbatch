#!/bin/bash
#SBATCH -J cuhkx-rcknf32
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
source "$HOME/Multimodal/scripts/selfcheck_io.sh" || exit 3
echo "== A. selftest 复核 nf32 =="
python -u scripts/selftest_thermal_val.py --ckpt outputs/th_nf32/r2plus1d34_thermal_fold0.pth \
  --num_frames 32 --fold 0 --workers 4 2>&1 | tail -3
echo "== B. collect th_nf32 OOF =="
python -u scripts/collect_th_oof.py --th_nf32 outputs/th_nf32/r2plus1d34_thermal_fold0.pth \
  --th_gru "" --fold 0 --workers 4 2>&1 | tail -4
echo "== C. 3路(暂2)融合折级 =="
python -u scripts/compare_th_fusion.py 2>&1 | tail -14
echo DONE
