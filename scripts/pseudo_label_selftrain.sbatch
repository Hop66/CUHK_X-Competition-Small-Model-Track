#!/bin/bash
# ============================================================
# CUHK-X —— 伪标签自训练（官方允许，2026-08-24）
# 策略：多模型 Co-training（main+thermal top1 一致）+ 动态 top-K 置信度
#       + 伪标签低权重 + fold0 val 早停（防过拟合）
# 流程：
#   [1] pseudo   : 全量 main+thermal(seed42) 对 405 测试打伪标签 → outputs/pseudo_labels.json
#   [2] selftrain thermal fold0（对照 0.6339）
#   [3] selftrain main    fold0（对照 0.6609）
# 前置: main_full/thermal_full seed42 权重 + bbox_test/bbox_thermal_test.json
# 用法: sbatch scripts/pseudo_label_selftrain.sbatch
# ============================================================
#SBATCH -J cuhkx-pseudo
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
mkdir -p logs outputs/selftrain
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for f in outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
         outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth \
         bbox_test.json bbox_thermal_test.json bbox_thermal_train.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

# ============ [1/3] 打伪标签 ============
echo "######## [1/3] 全量模型打伪标签（Co-training + top-K）########"
python scripts/pseudo_label_selftrain.py --stage pseudo
echo "######## [1/3] 伪标签完成 ########"

# ============ [2/3] selftrain thermal fold0 ============
echo "######## [2/3] 自训练 thermal fold0（对照 0.6339）########"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality thermal --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --weights ig65m_r2plus1d34.pth \
  --epochs 60 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain --workers 8
echo "######## [2/3] thermal selftrain 完成 ########"

# ============ [3/3] selftrain main fold0 ============
echo "######## [3/3] 自训练 main fold0（对照 0.6609）########"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality main --fold 0 --folds 3 \
  --crop_cache bbox_train.json --weights ig65m_r2plus1d34.pth \
  --epochs 60 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain --workers 8
echo "######## [3/3] main selftrain 完成 ########"

echo "==== PSEUDO LABEL SELF-TRAIN DONE（判读：thermal fold0 > 0.6339 / main fold0 > 0.6609 则有效）===="
