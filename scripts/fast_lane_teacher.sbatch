#!/bin/bash
#SBATCH -J cuhkx-tech
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/teacher
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) | fast-lane teacher(prep v2 修复后) ===="
python -u scripts/prep_distill_teacher.py --modality main --root train --quantize --flip \
  --ckpts outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_ps_fold0_int5.pth \
  --out outputs/teacher/teacher_main_train.pkl
python -u scripts/prep_distill_teacher.py --modality thermal --root train --quantize --flip \
  --ckpts outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth outputs/pack/thermal_avg_int5.pth \
  --out outputs/teacher/teacher_thermal_train.pkl
echo "==== teacher fast-lane DONE ===="
ls -la outputs/teacher/
