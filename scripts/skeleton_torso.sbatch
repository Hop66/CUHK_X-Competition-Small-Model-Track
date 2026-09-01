#!/bin/bash
# ============================================================
# CUHK-X —— 骨架 torso 归一化对照（0.711 notebook 建议：肩宽不稳 → 骨盆-颈躯干长）
# 骨架历史对照（fold0，Cosine）：
#   shoulder 无clean        = 0.5285~0.5447（最佳）
#   shoulder clean          = 0.5145（清洗反而降）
#   shoulder clean+part(all1)= 0.4984（part-aware v1 已证负）
# 本轮对照（fold0，Cosine）：
#   [1] torso 无clean            ：归一化由 shoulder→torso，其余全等
#   [2] torso + task_prior_part  ：part-aware v2，部位权重用 40 类任务先验初始化
#                               （26/40 类手/上肢主导 → 臂 1.12×/腿 1.05×/头 0.95×）
# 判读：[1] fold0 > 0.5285 → torso 有效；[2] fold0 > [1] → 任务先验 part 有效
# 前置：weights/mb_lite_latest_epoch.bin + NTU 骨架
# 用法: sbatch scripts/skeleton_torso.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-torso
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/skeleton_torso outputs/skeleton_torso_partprior
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 MotionBERT 权重"; exit 1; }
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
if [ ! -d "$NTU_DIR" ] || [ "$(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' 2>/dev/null | wc -l)" -le 0 ]; then
  echo "❌ 缺 NTU 骨架（$NTU_DIR）"; exit 1
fi

echo "########## [1/2] torso 归一化 无clean（对照 shoulder 0.5285~0.5447）##########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --norm torso --save_dir outputs/skeleton_torso --workers 8 \
  || echo "⚠️ [1] torso 失败"
echo "✅ [1] 完成（判读：fold0 > 0.5285 → torso 有效）"

echo "########## [2/2] torso + 任务先验 part-aware v2（对照 [1]）##########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --norm torso --part_aware --task_prior_part \
  --save_dir outputs/skeleton_torso_partprior --workers 8 \
  || echo "⚠️ [2] torso+partprior 失败"
echo "✅ [2] 完成（判读：fold0 > [1] → 任务先验 part 有效）"

echo "==== SKELETON TORSO DONE ===="
