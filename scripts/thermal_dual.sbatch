#!/bin/bash
# ============================================================
# CUHK-X —— Thermal 双流（静态热像大框 + 骨架运动）fold0 矩阵
# 架构（用户定案）: S 静态流(FrameNet/resnet18/34 per-frame logits mean) + M 骨架运动流(1D conv)
#                   融合 fused = α·softmax(S)+(1-α)·softmax(M)，α 可学；loss=CE(S)+CE(M)+CE(fused)
# 前提: 先跑 extract_motion_cache.py 产出 outputs/motion_cache.pkl（本脚本 [0] 自动补）
# 对照: S-only 大框 ≈ 0.3863(此前 2d_framevote crop 版) / thermal 3D 0.6339
#   [1] resnet18 S-only（同机基线）
#   [2] resnet18 S + 骨架M（主实验）
#   [3] framenet(从零) S + 骨架M（轻量双流）
#   [4] 骨架M only（运动流独立信息量）
# 判读: [2] > [1] → 骨架运动有真增益；[4] >> 0 → 运动本身有判别力；
#       [2]/[3] 逼近/超 0.6339 → thermal 双流成立 → 并入 0.73 融合冲 0.8
# 用法: sbatch scripts/thermal_dual.sbatch
# ============================================================
#SBATCH -J cuhkx-th-dual
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
mkdir -p logs outputs/thermal_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP"; exit 1; }
MC=outputs/motion_cache.pkl
EP=40

echo "########## [0/4] 骨架运动缓存（缺则构建）##########"
if [ ! -f "$MC" ]; then
  python scripts/extract_motion_cache.py || echo "⚠️ 运动缓存构建失败（[2-4] 的 M 流将降级）"
fi

echo "########## [1/4] resnet18 S-only（同机基线，对照 0.3863）##########"
python scripts/train_thermal_dual.py --fold 0 --arch resnet18 --use_static \
  --num_frames 8 --size 112 --gray_norm --epochs $EP \
  --crop_cache "$CROP" --save_dir outputs/thermal_dual --workers 8 \
  || echo "⚠️ [1] S-only 失败"

echo "########## [2/4] resnet18 S + 骨架M（主实验）##########"
python scripts/train_thermal_dual.py --fold 0 --arch resnet18 --use_static --use_motion \
  --num_frames 8 --size 112 --gray_norm --epochs $EP \
  --crop_cache "$CROP" --motion_cache "$MC" --save_dir outputs/thermal_dual --workers 8 \
  || echo "⚠️ [2] S+M 失败"

echo "########## [3/4] framenet(从零) S + 骨架M（轻量双流）##########"
python scripts/train_thermal_dual.py --fold 0 --arch framenet --use_static --use_motion \
  --num_frames 8 --size 112 --gray_norm --epochs $EP \
  --crop_cache "$CROP" --motion_cache "$MC" --save_dir outputs/thermal_dual --workers 8 \
  || echo "⚠️ [3] FrameNet+M 失败"

echo "########## [4/4] 骨架M only（运动流独立信息量）##########"
python scripts/train_thermal_dual.py --fold 0 --arch framenet --use_motion \
  --num_frames 8 --size 112 --gray_norm --epochs $EP \
  --crop_cache "$CROP" --motion_cache "$MC" --save_dir outputs/thermal_dual --workers 8 \
  || echo "⚠️ [4] M-only 失败"

echo "==== THERMAL DUAL DONE ===="
echo "结果: outputs/thermal_dual/*.pth"
echo "判读: [2]>[1] 骨架真增益；[4]>>0 运动有判别力；[2]/[3] vs 0.6339 = 双流高度"
echo "⚠️ 均 fold0 单折，选优后再补 fold1/2（subject-wise 纪律）"
