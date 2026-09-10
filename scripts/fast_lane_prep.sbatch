#!/bin/bash
#SBATCH -J cuhkx-fastprep
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
mkdir -p logs outputs/teacher outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) | fast-lane prep (teacher+OOF) ===="

# A) 教师软标签（distill 前置，53509 曾静默失败 → 重跑并打全程日志）
python scripts/prep_distill_teacher.py --modality main --root train --quantize --flip \
  --ckpts outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_ps_fold0_int5.pth \
  --out outputs/teacher/teacher_main_train.pkl; echo "teacher-main exit=$?"
python scripts/prep_distill_teacher.py --modality thermal --root train --quantize --flip \
  --ckpts outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth outputs/pack/thermal_avg_int5.pth \
  --out outputs/teacher/teacher_thermal_train.pkl; echo "teacher-thermal exit=$?"

# B) OOF/测试 logits（stacker 前置）
python scripts/extract_oof_logits.py --main --mode oof --flip \
  --fold_ckpts outputs/main_baseline/baseline_aug2_fold0.pth outputs/main_baseline/baseline_aug2_fold1.pth outputs/main_baseline/baseline_aug2_fold2.pth \
  --out outputs/oof/main_oof.pkl; echo "main-oof exit=$?"
python scripts/extract_oof_logits.py --main --mode test --flip \
  --fold_ckpts outputs/main_full/r2plus1d34_depthir_full_seed42.pth --out outputs/oof/main_test.pkl; echo "main-test exit=$?"
python scripts/extract_oof_logits.py --thermal --mode test --flip \
  --fold_ckpts outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth --out outputs/oof/thermal_test.pkl; echo "th-test exit=$?"

echo "==== fast-lane prep DONE ===="
ls -la outputs/teacher/ outputs/oof/
