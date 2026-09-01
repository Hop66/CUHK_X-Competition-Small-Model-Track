#!/bin/bash
# ============================================================
# CUHK-X —— Thermal 3D 段级投票 A/B（14th 帧级投票精髓的 3D 版）
# 背景：2D 帧级投票(resnet18/34) 0.40/0.44 追不上 thermal 3D 0.6339——
#       "投票"该用在 3D 微段上：clip 拆 K 个 8帧短段，每段独立 R2+1D-34 前向 → 段级 logits 平均
#       = 保留 3D 时序 + 多段 bagging 压方差 + 短段难编"整16帧长程身份捷径"（防泄漏，14th 精髓）
#   [1] T=16 vote_k=1  基线（复核 0.6339）
#   [2] T=16 vote_k=2  2×8 段平均
#   [3] T=24 vote_k=3  3×8 段平均（覆盖更全）
# 判读：vote_k>1 任一 > [1]（尤其 >0.6339）→ 3D 段级投票有效 → thermal 全量/移植 main 3D
# 用法: sbatch scripts/thermal_vote3d.sbatch
# ============================================================
#SBATCH -J cuhkx-th-vote3d
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 14:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/thermal_vote3d
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP"; exit 1; }
EP=60

echo "########## [1/3] T=16 vote_k=1 基线（复核 0.6339）##########"
python scripts/train_backbone_compare.py --backbone r2plus1d34 --modality thermal \
  --weights ig65m_r2plus1d34.pth --lr 1e-4 --num_frames 16 --vote_k 1 \
  --epochs $EP --fold 0 --size 128 --crop_cache "$CROP" \
  --save_dir outputs/thermal_vote3d --workers 8 \
  || echo "⚠️ [1] 基线 失败"

echo "########## [2/3] T=16 vote_k=2（2×8 段平均）##########"
python scripts/train_backbone_compare.py --backbone r2plus1d34 --modality thermal \
  --weights ig65m_r2plus1d34.pth --lr 1e-4 --num_frames 16 --vote_k 2 \
  --epochs $EP --fold 0 --size 128 --crop_cache "$CROP" \
  --save_dir outputs/thermal_vote3d --workers 8 \
  || echo "⚠️ [2] vote2 失败"

echo "########## [3/3] T=24 vote_k=3（3×8 段平均）##########"
python scripts/train_backbone_compare.py --backbone r2plus1d34 --modality thermal \
  --weights ig65m_r2plus1d34.pth --lr 1e-4 --num_frames 24 --vote_k 3 \
  --epochs $EP --fold 0 --size 128 --crop_cache "$CROP" \
  --save_dir outputs/thermal_vote3d --workers 8 \
  || echo "⚠️ [3] vote3 失败"

echo "==== THERMAL VOTE3D DONE ===="
echo "结果: outputs/thermal_vote3d/r2plus1d34_thermal_fold0.pth（最后写者胜，逐档看日志 val）"
echo "判读: [2]/[3] 任一 > [1](基线) 且 >0.6339 → 3D 段级投票有效 → thermal 全量/移植 main"
