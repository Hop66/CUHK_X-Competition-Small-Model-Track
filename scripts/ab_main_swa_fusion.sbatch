#!/bin/bash
# ============================================================
# CUHK-X —— 待办 B:main 三 seed SWA + thermal SWA 融合升级候选
# 产物(已在登录节点 CPU 完成):
#   outputs/pack/main_avg_int5.pth    (seed42+2024+777 SWA, int5, 40.2MB)
#   outputs/pack/thermal_avg_int5.pth (seed42+777 SWA, int5, 40.2MB)
# 本脚本:同 job 内 4 个融合 AB 单元出 CSV(LB 比对 0.73 链)
#   1) main_s42 + thermal_s42  → 复现 0.73 链基线(参照)
#   2) main_avg + thermal_s42  → SWA main 是否更优
#   3) main_s42 + thermal_avg  → SWA thermal 是否更优
#   4) main_avg + thermal_avg  → 双 SWA
# 用法: sbatch scripts/ab_main_swa_fusion.sbatch
# ============================================================
#SBATCH -J cuhkx-ab-swa
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for f in outputs/pack/main_s42_fold0_int5.pth outputs/pack/thermal_s42_fold0_int5.pth \
         outputs/pack/main_avg_int5.pth outputs/pack/thermal_avg_int5.pth; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

CROPS="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"
BASE="python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize $CROPS"

echo "==== [1/4] 参照 main_s42+thermal_s42 (≈0.73 链) ===="
$BASE --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth \
  --output outputs/sub_B_mains42_ths42.csv || echo "⚠️ 1 失败"

echo "==== [2/4] main_avg + thermal_s42 ===="
$BASE --main outputs/pack/main_avg_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth \
  --output outputs/sub_B_mainavg_ths42.csv || echo "⚠️ 2 失败"

echo "==== [3/4] main_s42 + thermal_avg ===="
$BASE --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_avg_int5.pth \
  --output outputs/sub_B_mains42_thavg.csv || echo "⚠️ 3 失败"

echo "==== [4/4] main_avg + thermal_avg ===="
$BASE --main outputs/pack/main_avg_int5.pth \
  --thermal outputs/pack/thermal_avg_int5.pth \
  --output outputs/sub_B_mainavg_thavg.csv || echo "⚠️ 4 失败"

echo ""
echo "==== SWA AB DONE ===="
echo "提交候选(待 LB): sub_B_mainavg_thavg.csv (双 SWA) / sub_B_mainavg_ths42.csv"
echo "参照: sub_B_mains42_ths42.csv (复现 0.73 链)"
echo "判读: 任一 > 0.73 → 采纳该组合替换 main_s42+thermal_s42"
