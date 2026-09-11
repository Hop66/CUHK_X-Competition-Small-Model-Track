#!/bin/bash
# ============================================================
# CUHK-X 09-11: 双塔同步难对重权 fold0 验证
# 假设检验（用户点名）: 历史训练侧难对只改 main 未同步 th → 现双塔共用同一
#   HARD_PAIRS（train_step1.py 模块级 11 对精选可学难对, main2c 0.65~0.85）, 
#   main+th 各自 hardpair_w=2 重权训练, 融合对照基线 fold0=0.6366。
# 对照: main_oof(0.6695 单/基线) + thermal_oof; 唯一变量 hardpair_w。
# 判读: 融合 fold0 acc > 0.6366 且难对区错误下降 → 双塔同步有效 → 上 3 折/full
# 用法: sbatch scripts/hardpair_dual_f0.sbatch
# ============================================================
#SBATCH -J cuhkx-hpdual
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/dual_hp_f0 outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) 双塔同步难对重权 fold0 (hardpair_w=2, 同难对表) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }

COMMON_MAIN="--backbone r2plus1d34 --weights ig65m_r2plus1d34.pth --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_train.json --modality depthir"
COMMON_TH="--backbone r2plus1d34 --weights ig65m_r2plus1d34.pth --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json --modality thermal"

echo "===== [1] main fold0 + hardpair_w=2.0 (难对样本 CE ×2) ====="
python scripts/train_step1.py $COMMON_MAIN --hardpair_w 2.0 \
  --save_dir outputs/dual_hp_f0/main 2>&1 | tee logs/hpdual_main_f0.log | tail -3 || echo "⚠️ main"

echo "===== [2] thermal fold0 + hardpair_w=2.0 (同一难对表同步) ====="
python scripts/train_step1.py $COMMON_TH --hardpair_w 2.0 \
  --save_dir outputs/dual_hp_f0/th 2>&1 | tee logs/hpdual_th_f0.log | tail -3 || echo "⚠️ th"

echo "===== [3] 难对双塔 vs 基线双塔 fold0 融合验证 ====="
python scripts/hardpair_dual_fold.py 2>&1 | tail -8

echo "==== 双塔同步难对重权 fold0 DONE ===="
