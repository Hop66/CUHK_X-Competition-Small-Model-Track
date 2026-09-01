#!/bin/bash
# ============================================================
# CUHK-X —— 骨架混合增强：NTU 清洗分类(A001-A040 中 13 类语义一致)直接混入训练
# 单一 40 类头（无 multi-task 干扰），NTU ~12,324 样本作为增强数据
# 对照：multi-task 0.5264 / 基线 0.51
# 判读：fold0 ours_val 显著 > 0.5264 → NTU 混合增强有效 → 全量+骨架融合
# 前置：weights/mb_lite_latest_epoch.bin + data/external/ntu/ntu60/*.skeleton
# 用法: sbatch scripts/skeleton_augmented.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-aug
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
mkdir -p logs outputs/skeleton_aug
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

echo "######## 骨架混合增强 fold0（对照 multi-task 0.5264 / 基线 0.51）########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 \
  --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --save_dir outputs/skeleton_aug --workers 8
echo "######## 混合增强完成 ########"

echo "==== SKELETON AUG DONE（判读：fold0 ours_val 显著 > 0.5264 则 NTU 混合有效）===="
