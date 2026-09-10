#!/bin/bash
# ============================================================
# CUHK-X —— 只重跑伪标签 3 轮 fold0（teacher 已修为纯 main R2+1D ckpt）
# 背景：unified_revalidate [3] 首轮因 teacher 类型（dual SM）不匹配全崩；
#       本脚本用纯 main ckpt 作 teacher（默认 main_S_fold0.pth = 0.6598），
#       验证「F1 crop 键修复后」伪标签多轮自训练是否真增益/饱和。
# 判读：每轮 selftrain fold0 val（round0 基线 → round1/2 递增 = 多轮有效真增益）
# 用法: sbatch scripts/pseudo_iter3.sbatch   （MAIN_TEACHER=/TH_TEACHER= 可覆盖）
# ============================================================
#SBATCH -J cuhkx-pseudo-it3
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
mkdir -p logs outputs/pseudo_iter
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MAIN_T="${MAIN_TEACHER:-outputs/main_dual/main_S_fold0.pth}"
TH_T="${TH_TEACHER:-outputs/thermal_notebook14/t0/r2plus1d34_thermal_fold0.pth}"
[ -f "$MAIN_T" ] || { echo "⚠️ 缺纯 main teacher $MAIN_T（可用 --export=MAIN_TEACHER=... 覆盖）"; }

for R in 0 1 2; do
  PJ="outputs/pseudo_iter/pseudo_r${R}.json"
  echo "----- round$R 打伪标签 (teacher=${MAIN_T}) -----"
  if [ ! -f "$PJ" ]; then
    python scripts/pseudo_label_selftrain.py --stage pseudo \
      --main_ckpt "$MAIN_T" --thermal_ckpt "$TH_T" --pseudo_json "$PJ" \
      || { echo "⚠️ round$R 伪标签失败（跳过本轮 selftrain）"; continue; }
    [ -f "$PJ" ] || { echo "⚠️ round$R 未产出 $PJ（跳过）"; continue; }
  else
    echo "-- $PJ 已存在，复用"
  fi
  echo "----- round$R main selftrain fold0 -----"
  python scripts/pseudo_label_selftrain.py --stage selftrain --modality main --fold 0 \
    --pseudo_json "$PJ" --save_dir "outputs/pseudo_iter/r${R}" \
    --num_frames 16 --size 128 --epochs 60 --lr 1e-4 --batch_size 32 --crop_cache bbox_train.json \
    || { echo "⚠️ round$R selftrain 失败"; }
  NEW="outputs/pseudo_iter/r${R}/main_selftrain_fold0.pth"
  if [ -f "$NEW" ]; then MAIN_T="$NEW"; else echo "⚠️ 缺轮末 ckpt：${NEW}（teacher 不更新）"; fi
done

echo "==== PSEUDO ITER3 DONE ===="
echo "判读: 3 轮 selftrain fold0 val —— 递增=修复后多轮自训练真增益；饱和/下滑=单轮足矣"
