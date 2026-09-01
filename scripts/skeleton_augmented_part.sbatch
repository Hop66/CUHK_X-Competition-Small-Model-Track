#!/bin/bash
# ============================================================
# CUHK-X —— 骨架 part-aware 部位权重对照（用户思路："针对动作改变不同部位权重"）
# 机制：ActionNet 加可学习部位权重（每关节 1 个，1+tanh(logits)，初始=1.0 完全原行为），
#   训练中学每关节增强(>1)/抑制(<1)——ST-GCN edge-importance / CTR-GCN part attention 同思想。
# 组合：--clean（先修骨架数据质量，上一轮已实现）+ --part_aware（部位权重）。
# 对照：skeleton_augmented（无 clean 无 part_aware）fold0 = 0.5447（最佳基线）
# 判读：fold0 > 0.5447 且 > clean-only 结果 → part-aware 有效 → 全量重训参与融合
# 用法: sbatch scripts/skeleton_augmented_part.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-part
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
mkdir -p logs outputs/skeleton_aug_part
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

echo "######## 骨架清洗 + part-aware 部位权重 fold0 ########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 \
  --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --clean --part_aware \
  --save_dir outputs/skeleton_aug_part --workers 8
echo "######## 清洗+part-aware 完成 ########"

echo "==== SKELETON PART DONE（判读：fold0 > 0.5447 且 > clean-only → part-aware 有效）===="
