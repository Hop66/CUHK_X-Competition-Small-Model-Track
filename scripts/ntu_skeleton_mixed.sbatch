#!/bin/bash
# ============================================================
# CUHK-X —— 骨架 multi-task 混合训练（NTU A001-A040 外部补充）
# 对照：骨架基线（MB_lite, 2891 样本）fold0 ~0.45-0.51
# 思路：共享 MotionBERT backbone + 我们 40 类头 + NTU 40 类头（绕开类别映射）
# 判读：fold0 ours_val > 0.51 → NTU 补充有效 → 全量骨架+NTU 重训 → 参与三模态融合
# 前置：weights/mb_lite_latest_epoch.bin + data/external/ntu/ntu60/*.skeleton
# 用法: sbatch scripts/ntu_skeleton_mixed.sbatch
# ============================================================
#SBATCH -J cuhkx-ntu-skel
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
mkdir -p logs outputs/skeleton_multitask
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 MotionBERT 权重"; exit 1; }
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
# 用 find 计数（ls *.skeleton 会展开 5.6 万参数 -> Argument list too long）
if [ ! -d "$NTU_DIR" ] || [ "$(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' 2>/dev/null | wc -l)" -le 0 ]; then
  echo "❌ 缺 NTU 骨架（$NTU_DIR）"
  echo "   实际路径排查: find ~/Multimodal/data/external/ntu -maxdepth 2 -name '*.skeleton' | head"
  echo "   若在别处: 设置 NTU_ROOT=<绝对路径> 再 sbatch"
  exit 1
fi
echo "NTU 骨架数: $(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' | wc -l)"

echo "######## 骨架 multi-task fold0（对照 0.51）########"
python scripts/train_skeleton_multitask.py \
  --fold 0 --folds 3 \
  --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 32 --ntu_weight 0.5 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --save_dir outputs/skeleton_multitask --workers 8
echo "######## multi-task 完成 ########"

echo "==== NTU SKELETON MIXED DONE（判读：fold0 ours_val > 0.51 则 NTU 补充有效）===="
