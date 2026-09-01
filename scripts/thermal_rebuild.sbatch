#!/bin/bash
# ============================================================
# CUHK-X —— THERMAL 重做网格：给 thermal（单通道灰度视频）系统选骨干 + 输入配方
# 背景：thermal 一直复用 main 的 recipe（R2+1D-34 IG65M + 128 + Kinetics + uniform）
#        folds 仅 0.6339/0.6053。14 名用 thermal 单模冲 0.8 → 值得重做。
# 现成 CNN 视频预训练框架（torchvision Kinetics-400 官方）：swin3d_t / x3d_m / s3d /
#        mvit_v2_s / slow_r50 … + 我们 r2plus1d34（IG65M）。128 与 224 两档输入。
# 网格（thermal，fold0，60ep，subject 协议，Kinetics norm 配预训练域；失败骨干自动跳过）：
#   128 档: [1] r2plus1d34(IG65M)基线 / [2] swin3d_t
#   224 档: [3] r2plus1d34 / [4] mvit_v2_s / [5] x3d_m / [6] s3d
#   组合 :  [7] swin3d_t + segment + gray（notebook 输入思路 挂在最强 128 骨干上）
# 减背景 : 背景=身份/场景捷径（同人高度相关下最危险），YOLO 紧框可有效抑制
#          detect --margin 1.2 生成 bbox_thermal_m12.json → 人物更满、背景更少
#          [8] swin3d_t + 紧框128 / [9] r2plus1d34 + 紧框128（隔离紧 crop 净值，对照 [1]）# 判读：谁 fold0 > 0.6339 → thermal 全量换该骨干+配方 → 0.73 融合重现冲 0.8
# 前置：ig65m_r2plus1d34.pth（r2plus1d34 用）+ bbox_thermal_train.json + 数据
# 用法: sbatch scripts/thermal_rebuild.sbatch
# ============================================================
#SBATCH -J cuhkx-thbm-rebuild
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
mkdir -p logs outputs/thermal_rebuild
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP"; exit 1; }
EP=60

echo "########## [1/7] 128  r2plus1d34(IG65M) 基线（对照 0.6339）##########"
[ -f ig65m_r2plus1d34.pth ] && python scripts/train_backbone_compare.py \
  --backbone r2plus1d34 --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 128 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [1] r2plus1d34 失败"

echo "########## [2/7] 128  swin3d_t (Kinetics400) ##########"
python scripts/train_backbone_compare.py \
  --backbone swin3d_t --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 128 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [2] swin3d_t 失败"

echo "########## [3/7] 224  r2plus1d34 (匹配 IG65M 预训练域 224) ##########"
[ -f ig65m_r2plus1d34.pth ] && python scripts/train_backbone_compare.py \
  --backbone r2plus1d34 --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 224 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [3] r2plus1d34@224 失败"

echo "########## [4/7] 224  mvit_v2_s (Kinetics400, 强 transformer) ##########"
python scripts/train_backbone_compare.py \
  --backbone mvit_v2_s --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 224 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [4] mvit_v2_s 失败"

echo "########## [5/7] 224  x3d_m (轻量高效) ##########"
python scripts/train_backbone_compare.py \
  --backbone x3d_m --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 224 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [5] x3d_m 失败"

echo "########## [6/7] 224  s3d (轻量 CNN) ##########"
python scripts/train_backbone_compare.py \
  --backbone s3d --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 224 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [6] s3d 失败"

echo "########## [7/7] 128  swin3d_t + segment + gray（notebook 输入思路）##########"
python scripts/train_backbone_compare.py \
  --backbone swin3d_t --modality thermal --crop_cache "$CROP" \
  --epochs $EP --fold 0 --size 128 \
  --sample_mode segment --gray_norm \
  --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [7] swin3d+seg+gray 失败"

echo "########## [8/9] 生成紧框 bbox_thermal_m12.json（--margin 1.2，人物更满背景更少）##########"
python -m src.detect --mode train --out bbox_thermal_m12.json --modality thermal --margin 1.2 \
  || echo "⚠️ detect margin1.2 失败"
CROP_TIGHT=bbox_thermal_m12.json

if [ -f "$CROP_TIGHT" ]; then
echo "########## [8/9] 128  swin3d_t + 紧框(margin1.2) ##########"
python scripts/train_backbone_compare.py \
  --backbone swin3d_t --modality thermal --crop_cache "$CROP_TIGHT" \
  --epochs $EP --fold 0 --size 128 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [8] swin3d+紧框失败"

echo "########## [9/9] 128  r2plus1d34 + 紧框(margin1.2)（对照 [1]，隔离紧 crop 净值）##########"
[ -f ig65m_r2plus1d34.pth ] && python scripts/train_backbone_compare.py \
  --backbone r2plus1d34 --modality thermal --crop_cache "$CROP_TIGHT" \
  --epochs $EP --fold 0 --size 128 --save_dir outputs/thermal_rebuild --workers 8 \
  || echo "⚠️ [9] r2plus1d34+紧框失败"
else
  echo "⚠️ [8][9] 跳过（紧框生成失败）"
fi

echo "==== THERMAL REBUILD DONE ===="
echo "结果: outputs/thermal_rebuild/"
echo "判读: 任一骨干/配方 fold0 > 0.6339 → thermal 全量换该配置 → 0.73 融合重现冲 0.8；"
echo "      [9] vs [1] = 紧框(减背景)基线净值；[8] vs [2] = 紧框在 swin 上的净值；"
echo "      建议对 fold0 最优者补 fold1/fold2 复核（subject-wise 纪律）。"
