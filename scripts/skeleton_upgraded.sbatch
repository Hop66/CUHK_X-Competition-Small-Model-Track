#!/bin/bash
# ============================================================
# CUHK-X —— 骨架升级：AimCLR 增强 + 24 帧 + NTU 混合（对照混合增强 0.5447）
# 用法: sbatch scripts/skeleton_upgraded.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-up
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
mkdir -p logs outputs/skeleton_upgraded
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 MotionBERT 权重"; exit 1; }
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
[ -d "$NTU_DIR" ] || { echo "⚠️ 无 NTU（纯我们数据升级）"; NTU_DIR=""; }

echo "######## 骨架升级（24帧 + AimCLR + NTU混合）fold0 ########"
if [ -n "$NTU_DIR" ]; then
  python scripts/train_skeleton_upgraded.py --fold 0 --folds 3 \
    --ntu_root "$NTU_DIR" --num_frames 24 \
    --pretrained weights/mb_lite_latest_epoch.bin \
    --save_dir outputs/skeleton_upgraded --workers 8
else
  python scripts/train_skeleton_upgraded.py --fold 0 --folds 3 \
    --num_frames 24 --pretrained weights/mb_lite_latest_epoch.bin \
    --save_dir outputs/skeleton_upgraded --workers 8
fi
echo "######## 骨架升级完成 ########"

echo "==== SKELETON UPGRADED DONE（判读：fold0 > 0.5447 显著则升级有效）===="
