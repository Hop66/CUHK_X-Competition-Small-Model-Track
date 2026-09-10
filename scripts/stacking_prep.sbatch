#!/bin/bash
# ============================================================
# CUHK-X Iter14: 多模态 OOF Stacking —— 数据准备 + 首版融合 CV
#   0) 补训 thermal fold1/2(Kinetics, 与 fold0 同级协议)
#   1) 提 main OOF(baseline_aug2 f0/1/2) + thermal OOF(f0/1/2)
#   2) 提测试 logits(main_full / thermal_full fp32)
#   3) train_fusion_stacker 跑 leave-one-fold CV: 学习式融合 vs 概率平均(0.73134 线)
# 产物: outputs/oof/{main,thermal}_{oof,test}.pkl ; CV 报告; 若提升则出 sub_stack.csv
# 用法: sbatch scripts/stacking_prep.sbatch
# ============================================================
#SBATCH -J cuhkx-stack
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/oof outputs/th_folds
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter14 Stacking Prep ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1) thermal fold1/2（与 fold0 base 同配方: Kinetics, 60ep, 128px, aug2, ls0.1）
for F in 1 2; do
  OUT=outputs/th_folds
  CK=$(ls $OUT/r2plus1d34_thermal_fold${F}.pth 2>/dev/null || true)
  if [ -z "$CK" ]; then
    echo "===== [0] thermal fold$F 60ep ====="
    python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
      --fold "$F" --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 \
      --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json \
      --modality thermal --save_dir "$OUT" 2>&1 | tee "logs/th_fold${F}.log" | tail -3 || echo "⚠️ fold$F"
  else
    echo "thermal fold$F 已存在: $CK"
  fi
done

echo "===== [1] 提 OOF ====="
TF0=$(ls outputs/ab_th_fold0/base/r2plus1d34_thermal_fold0.pth outputs/th_folds/r2plus1d34_thermal_fold*.pth 2>/dev/null | sort -V)
echo "thermal fold ckpts: $TF0"
python scripts/extract_oof_logits.py --main --mode oof \
  --fold_ckpts outputs/main_baseline/baseline_aug2_fold0.pth \
              outputs/main_baseline/baseline_aug2_fold1.pth \
              outputs/main_baseline/baseline_aug2_fold2.pth \
  --flip --out outputs/oof/main_oof.pkl || echo "⚠️ main OOF"
python scripts/extract_oof_logits.py --thermal --mode oof \
  --fold_ckpts $TF0 \
  --crop bbox_thermal_train.json --flip --out outputs/oof/thermal_oof.pkl || echo "⚠️ th OOF"

echo "===== [2] 提测试 logits(fp32 full) ====="
python scripts/extract_oof_logits.py --main --mode test --flip \
  --fold_ckpts outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
  --out outputs/oof/main_test.pkl || echo "⚠️ main test"
python scripts/extract_oof_logits.py --thermal --mode test --flip \
  --fold_ckpts outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth \
  --out outputs/oof/thermal_test.pkl || echo "⚠️ th test"

echo "===== [3] stacker CV (main+thermal) ====="
python scripts/train_fusion_stacker.py --model mlp \
  --oof main=outputs/oof/main_oof.pkl thermal=outputs/oof/thermal_oof.pkl \
  --test main=outputs/oof/main_test.pkl thermal=outputs/oof/thermal_test.pkl \
  --output outputs/sub_stack_mt.csv || echo "⚠️ stacker"

echo ""
echo "==== Iter14 DONE: 判读 CV 报告(提升>0 且大 才把 stacker 焊进提交) ===="
