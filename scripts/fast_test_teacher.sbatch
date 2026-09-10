#!/bin/bash
#SBATCH -J cuhkx-tteacher
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/teacher
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "== 测试集教师软标签(以备 transductive-soft 用) =="
python scripts/prep_distill_teacher.py --modality main --root test --quantize --flip \
  --ckpts outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth \
  --out outputs/teacher/teacher_main_test.pkl
python scripts/prep_distill_teacher.py --modality thermal --root test --quantize --flip \
  --ckpts outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_avg_int5.pth \
  --out outputs/teacher/teacher_thermal_test.pkl
echo DONE
