#!/bin/bash
# ============================================================
# CUHK-X —— 积压实验统一编排（vote 之后所有"待跑未跑"一次排队，2026-09-01）
# 顺序（每步 || echo 容错，失败不阻塞后续）：
#   [1] thermal-2D 域匹配 A/B: resnet18 + ImageNet-norm（vs gray-norm 对照）
#   [2] thermal-2D 域匹配 A/B: resnet18 + gray-norm（现状同机对照）
#   [3-6] main 双流 fold1/2 复核：S-only vs S+骨架M（判定 +0.4pt 是否稳定）
#   [7] thermal 3D + 骨架（新，从未做过）：th3D-only 复核
#   [8] thermal 3D + 骨架：th3D + 骨架M（主）
# 判读矩阵：
#   [2]-[1] ≥ +3pt → 域矛盾是 2D 弱元凶（反之 2D 静态 0.4 即天花板）
#   3-6 mean(SM)-mean(S) ≥ +0.5pt 且同向 → main 骨架真增益 → 全量+入 0.73
#   [8] > [7] → 骨架辅助 thermal 3D 成立 → 骨架辅助双 3D 齐 → 融合白拿
# 前置：ig65m_r2plus1d34.pth + bbox_thermal_train.json + bbox_train.json + motion_cache
# 用法: sbatch scripts/unified_pending.sbatch
# ============================================================
#SBATCH -J cuhkx-unified-pending
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/thermal_2d_norm outputs/main_dual outputs/thermal3d_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }
TCROP=bbox_thermal_train.json
[ -f "$TCROP" ] || { echo "❌ 缺 $TCROP"; exit 1; }
MCROP=bbox_train.json
[ -f "$MCROP" ] || { echo "❌ 缺 $MCROP（main 固定框）"; exit 1; }
MC=outputs/motion_cache.pkl
[ -f "$MC" ] || { echo "⚠️ 缺 $MC"; python scripts/extract_motion_cache.py || echo "⚠️ 缓存构建失败"; }
EP60=60
EP40=40

echo "########## [1/8] thermal-2D: resnet18 + ImageNet-norm（域匹配）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --imagenet_norm \
  --epochs $EP40 --crop_cache "$TCROP" --save_dir outputs/thermal_2d_norm --workers 8 \
  || echo "⚠️ [1] ImageNet-norm 失败"

echo "########## [2/8] thermal-2D: resnet18 + gray-norm（同机对照）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs $EP40 --crop_cache "$TCROP" --save_dir outputs/thermal_2d_norm --workers 8 \
  || echo "⚠️ [2] gray-norm 失败"

for FOLD in 1 2; do
  echo "########## [$([ $FOLD -eq 1 ] && echo 3 || echo 5)/8] main 双流 fold$FOLD S-only ##########"
  python scripts/train_main_dual.py --fold "$FOLD" --no_motion \
    --num_frames 16 --size 128 --epochs $EP60 --lr 1e-4 --batch_size 32 \
    --weights ig65m_r2plus1d34.pth --crop_cache "$MCROP" \
    --save_dir outputs/main_dual --workers 8 || echo "⚠️ main fold$FOLD S 失败"

  echo "########## [$([ $FOLD -eq 1 ] && echo 4 || echo 6)/8] main 双流 fold$FOLD S+骨架M ##########"
  python scripts/train_main_dual.py --fold "$FOLD" \
    --num_frames 16 --size 128 --epochs $EP60 --lr 1e-4 --batch_size 32 \
    --weights ig65m_r2plus1d34.pth --crop_cache "$MCROP" --motion_cache "$MC" \
    --save_dir outputs/main_dual --workers 8 || echo "⚠️ main fold$FOLD SM 失败"
done

echo "########## [7/8] thermal 3D-only 复核 ##########"
python scripts/train_thermal3d_dual.py --fold 0 --no_motion \
  --num_frames 16 --size 128 --epochs $EP60 --lr 1e-4 --batch_size 32 \
  --weights ig65m_r2plus1d34.pth --crop_cache "$TCROP" \
  --save_dir outputs/thermal3d_dual --workers 8 \
  || echo "⚠️ [7] th3D-only 失败"

echo "########## [8/9] thermal 3D + 骨架M（主，从未做过）##########"
python scripts/train_thermal3d_dual.py --fold 0 \
  --num_frames 16 --size 128 --epochs $EP60 --lr 1e-4 --batch_size 32 \
  --weights ig65m_r2plus1d34.pth --crop_cache "$TCROP" --motion_cache "$MC" \
  --save_dir outputs/thermal3d_dual --workers 8 \
  || echo "⚠️ [8] th3D+骨架 失败"

echo "########## [9/9] TTA4 / prior / imbalance val 对拍（yolo v9 三件套）##########"
M_CKPT=outputs/main_track/r2plus1d34_depthir_fold0.pth
T_CKPT=outputs/thermal_notebook14/t0/r2plus1d34_thermal_fold0.pth
if [ -f "$M_CKPT" ] && [ -f "$T_CKPT" ]; then
  python scripts/eval_v9_suite.py --main_ckpt "$M_CKPT" --thermal_ckpt "$T_CKPT" \
    --batch_size 8 --workers 8 \
    || echo "⚠️ [9] TTA/prior 对拍失败"
else
  echo "⚠️ [9] 跳过（缺 ckpt：$M_CKPT / $T_CKPT；可手动传路径跑 eval_v9_suite.py）"
fi

echo "==== UNIFIED PENDING DONE ===="
echo "判读: [1]vs[2] 域匹配 | 3-6 均值差 | [8]vs[7] 骨架辅助 th3D | [9] TTA4/prior"
