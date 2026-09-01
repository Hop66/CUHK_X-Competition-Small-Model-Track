#!/bin/bash
# ============================================================
# CUHK-X —— 骨架部位信息量诊断（决策"骨架双专家"前提）
# 输出：每类样本数 / 坏clip率 / fold0 per-class val_acc 按手精细/臂主导/下肢全身/静坐分组
# 前置：weights/mb_lite_latest_epoch.bin + 现成 fold0 骨架 checkpoint
#       （默认 outputs/skeleton_aug/augmented_fold0.pth = 0.5447 基线；可用 *cos / 待跑 torso 的）
# 判读：
#   HAND_FINE 组 val_acc << LOCO 组 → 17 关节无指尖+无物体，双专家学噪声 → 不值得
#   HAND_FINE 个别类 ≥0.4 且明显高于同类 → 区域运动可分 → 双专家(上肢专责手类)值得实现
# 用法: sbatch scripts/skeleton_parts_diag.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-parts-diag
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="

echo "######## 骨架部位信息量诊断（fold0, ckpt=outputs/skeleton_aug/augmented_fold0.pth）########"
python scripts/diagnose_skeleton_parts.py \
  --train_root ~/Multimodal/data/Training/HAR \
  --ckpt outputs/skeleton_aug/augmented_fold0.pth \
  --norm shoulder --fold 0 --folds 3 --workers 4
echo "######## 诊断完成 ########"

echo "==== SKELETON PARTS DIAG DONE（判读：HAND_FINE << LOCO → 双专家不值；个别手类≥0.4 → 值得）===="
