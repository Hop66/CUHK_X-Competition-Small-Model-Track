#!/bin/bash
# ============================================================
# CUHK-X —— 14th-place 0.8 thermal baseline 对接：组件 A/B（一次提交顺序跑 4 步）
#
# 背景：我们 thermal 基线 fold0=0.6339 / 3折 mean 0.6053；14 名队伍 thermal 单模 0.8（未公开优化）。
#       notebook 是 pre-optimization 简化版 → 不照搬，只移植可廉价 A/B 的组件逐项验证。
#   [1/4] T0  基线复核（uniform16 + Kinetics + YOLO crop）                 → 期望≈0.6339
#   [2/4] T1  +分段时间采样（segment, 16 帧）                              → 时间采样增量
#   [3/4] T2  +灰度归一化 (0.5,0.25)                                      → 归一化增量
#   [4/4] T3  segment + gray_norm + 8 帧（贴近 notebook）                  → 帧数增量
# 注：原 [5][6] FrameNet(从零 2D) 已删除 —— 已实测判负（8/16帧=0.21/0.23），重跑浪费 GPU；
#     真正有价值的 2D-per-frame 强化是 resnet18/34 ImageNet 预训练版（thermal_2d_framevote）。
# 判读矩阵（都对照 T0 / 0.6339）：
#   T1/T2/T3 中任一 fold0 > 0.64（尤其 >0.65）→ thermal 全量换该配置 → 0.73 融合重现冲 0.8
# 注意：以上全为单折 fold0 → 选出最优变体后，按 subject-wise 纪律补 fold1/fold2 复核再定案。
# 前置：ig65m_r2plus1d34.pth（项目根，与全库其它 sbatch 一致）+ bbox_thermal_train.json + 训练数据
# 用法: sbatch scripts/cuhkx_thermal_08_bundle.sbatch
# 注意：以上全为单折 fold0 → 选出最优变体后，按 subject-wise 纪律补 fold1/fold2 复核再定案。
# 前置：ig65m_r2plus1d34.pth（项目根，与全库其它 sbatch 一致）+ bbox_thermal_train.json + 训练数据
# 用法: sbatch scripts/cuhkx_thermal_08_bundle.sbatch
# ============================================================
#SBATCH -J cuhkx-thermal08-bundle
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
mkdir -p logs outputs/thermal_notebook14/t0 outputs/thermal_notebook14/t1 \
         outputs/thermal_notebook14/t2 outputs/thermal_notebook14/t3 \
         outputs/thermal_framenet
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重（项目根 ig65m_r2plus1d34.pth；不存在的跑 scripts/download_weights.py）"; exit 1; }
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP（先跑 detect --modality thermal）"; exit 1; }

echo "########## [1/4] T0 基线（uniform16+Kinetics）——复核 0.6339 ##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --label_smoothing 0.1 --crop_cache "$CROP" \
  --save_dir outputs/thermal_notebook14/t0 --workers 8 \
  || echo "⚠️ [1] T0 失败"

echo "########## [2/4] T1 = +分段时间采样（segment, 16 帧）##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --label_smoothing 0.1 --crop_cache "$CROP" \
  --sample_mode segment \
  --save_dir outputs/thermal_notebook14/t1 --workers 8 \
  || echo "⚠️ [2] T1 失败"

echo "########## [3/4] T2 = segment + 灰度归一化 (0.5,0.25) ##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --label_smoothing 0.1 --crop_cache "$CROP" \
  --sample_mode segment --gray_norm \
  --save_dir outputs/thermal_notebook14/t2 --workers 8 \
  || echo "⚠️ [3] T2 失败"

echo "########## [4/4] T3 = segment + gray_norm + 8 帧（贴近 notebook）##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --label_smoothing 0.1 --crop_cache "$CROP" \
  --sample_mode segment --gray_norm --num_frames 8 \
  --save_dir outputs/thermal_notebook14/t3 --workers 8 \
  || echo "⚠️ [4] T3 失败"

echo "==== CUHKX THERMAL 0.8 BUNDLE DONE（T0-T3）===="
echo "结果目录: outputs/thermal_notebook14/t{0,1,2,3}"
echo "判读（都对照 T0/0.6339）:"
echo "  T1/T2/T3 任一 >0.64(优选>0.65) → thermal 全量换该配置 → 0.73 融合重现"
echo "  ⚠️ 均为 fold0 单折，最优变体务必补 fold1/fold2 复核（subject-wise 纪律）"
