#!/bin/bash
# ============================================================
# CUHK-X Iter15: 第二套 thermal fold(seed2024) → 折级验证"2 thermal 成员"(隐形主力多样)
# 对照组: R(1main+1th) 基准; A(2main+1th) 已知+0.83; B(1main+2th) 新判定
# 用法: sbatch scripts/ab_th2nd_fold.sbatch
# ============================================================
#SBATCH -J cuhkx-th2nd
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/th_folds_s2024 outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) | Iter15 第二套 thermal fold(seed2024) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for F in 0 1 2; do
  OUT=outputs/th_folds_s2024
  CK=$(ls $OUT/r2plus1d34_thermal_fold${F}.pth 2>/dev/null || true)
  if [ -z "$CK" ]; then
    echo "===== thermal fold$F seed2024 60ep ====="
    python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
      --fold "$F" --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 \
      --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json \
      --modality thermal --seed 2024 --save_dir "$OUT" 2>&1 | tee "logs/th_fold2024_${F}.log" | tail -2 || echo "⚠$F"
  fi
done

echo "===== 提第二套 thermal OOF ====="
python scripts/extract_oof_logits.py --thermal --mode oof --flip \
  --fold_ckpts outputs/th_folds_s2024/r2plus1d34_thermal_fold0.pth outputs/th_folds_s2024/r2plus1d34_thermal_fold1.pth outputs/th_folds_s2024/r2plus1d34_thermal_fold2.pth \
  --crop bbox_thermal_train.json --out outputs/oof/thermal_oof_s2024.pkl || echo "⚠ext"

echo "===== 折级验证: 2 thermal 成员 ====="
python scripts/validate_ensemble_fold.py \
  --R 'm2=outputs/oof/main_oof.pkl,th1=outputs/oof/thermal_oof.pkl' \
  --A 'm2=outputs/oof/main_oof.pkl,m1=outputs/oof/main_oof_aug1.pkl,th1=outputs/oof/thermal_oof.pkl' --A_name 'A 2main+1th(已知+0.83)' \
  --A 'm2=outputs/oof/main_oof.pkl,th1=outputs/oof/thermal_oof.pkl,th2=outputs/oof/thermal_oof_s2024.pkl' --A_name 'B 1main+2th' \
  --A 'm2=outputs/oof/main_oof.pkl,m1=outputs/oof/main_oof_aug1.pkl,th1=outputs/oof/thermal_oof.pkl,th2=outputs/oof/thermal_oof_s2024.pkl' --A_name 'C 2main+2th' 2>&1 | tail -12

echo "==== Iter15 DONE ===="
