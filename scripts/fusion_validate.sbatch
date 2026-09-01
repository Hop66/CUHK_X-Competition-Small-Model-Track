#!/bin/bash
# ============================================================
# CUHK-X —— fold 级多模态融合验证（不烧提交，先确认增益方向）
# 验证：单 seed main + 单 seed thermal 融合 > main 单独？
#   [1] 各模态单独 acc（fold0）
#   [2] main+thermal 概率平均 / logit 平均 / 置信门控
#   [3] main+thermal+skeleton（若 3d 结果已出）
#   [4] oracle 上限（完美选择器）
# 用法: sbatch scripts/fusion_validate.sbatch
# 产物: 终端输出各组合 acc，日志在 logs/cuhkx-fusion-<jobid>.out
# ============================================================
#SBATCH -J cuhkx-fusion
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------- 模型路径（自动探测，缺哪个就不参与融合）----------
# main fold0：优先强 main（aug_search s2 0.6609），否则旧 main_4ch_2fold
MAIN0=outputs/main_aug/r2plus1d34_depthir_fold0.pth
[ -f "$MAIN0" ] || MAIN0=outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth
# thermal fold0：优先 thermal_v2（recipe 升级 0.6339）
TH0=outputs/thermal_v2/r2plus1d34_thermal_fold0.pth
[ -f "$TH0" ] || TH0=outputs/thermal_34/r2plus1d34_thermal_fold0.pth
# skeleton fold0：3d 优先
SK0=outputs/skeleton_3d/3d/motionbert_fold0.pth
[ -f "$SK0" ] || SK0=outputs/skeleton_mb/motionbert_fold0.pth

echo "==== fold0 融合验证 ===="
echo "main:    $MAIN0"
echo "thermal: $TH0"
echo "skeleton: $SK0"

MAIN_ARGS=(--main_ckpts "$MAIN0")
[ -f "$TH0" ] && MAIN_ARGS+=(--thermal_ckpts "$TH0")
[ -f "$SK0" ] && MAIN_ARGS+=(--skeleton_ckpts "$SK0")

python scripts/fusion_validate.py --fold 0 \
  "${MAIN_ARGS[@]}" \
  --main_crop bbox_train.json \
  --thermal_crop bbox_thermal_train.json \
  --workers 8

echo "==== FUSION VALIDATE DONE（对比 main 单独，看融合净增益）===="
