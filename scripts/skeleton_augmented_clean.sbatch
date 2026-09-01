#!/bin/bash
# ============================================================
# CUHK-X —— 骨架清洗对照：MotionBERT + NTU 混合 + 针对性清洗（--clean）
# 背景（用户质疑"骨架弱为什么不针对性解决"）：诊断显示本地样例骨架质量高（肩宽 std ±1.4%），
#   噪声是部分坏 clip（遮挡/单目 z 抖动/零帧）。已实现 src/skeleton_clean.py：
#   零帧插值 + 速度异常帧修复 + 肩宽漂移修复 + 时间平滑（原始米制坐标，中心化前）。
# 对照：skeleton_augmented（不清洗）fold0 = 0.5447（最佳）
# 判读：fold0 > 0.5447 且 > 0.56 → 清洗有效 → 全量清洗版骨架参与融合
# 前置：weights/mb_lite_latest_epoch.bin + NTU 骨架 + 诊断脚本先跑（scripts/diag_skeleton_quality.py）
# 用法: sbatch scripts/skeleton_augmented_clean.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-clean
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
mkdir -p logs outputs/skeleton_aug_clean
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

echo "==== 先跑骨架全量质量诊断（定位坏 clip 分布）===="
python scripts/diag_skeleton_quality.py --train_root ~/Multimodal/data/Training/HAR 2>&1 | tee "logs/diag-skel-$SLURM_JOB_ID.log" || echo "⚠️ 诊断脚本失败（不阻断）"

echo "######## 骨架混合增强 + 清洗 fold0（对照不清洗 0.5447）########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 \
  --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --clean --save_dir outputs/skeleton_aug_clean --workers 8
echo "######## 清洗版混合增强完成 ########"

echo "==== SKELETON CLEAN DONE（判读：fold0 > 0.5447 且 > 0.56 → 清洗有效）===="
