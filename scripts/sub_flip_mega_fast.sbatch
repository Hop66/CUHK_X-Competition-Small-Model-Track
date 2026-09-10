#!/bin/bash
# ============================================================
# CUHK-X Iter11: 回到"仅 flip"基线做 ensemble（LB 已证 TTA4/多裁剪全负）
# 参照链(flip only) = 0.73134。测试纯集成效应(无任何时序/裁剪TTA)：
#   N1 参照        main_s42+th_s42 flip-only   (控制, 应≈0.73134)
#   N2 mega3       main×3 + th×3 flip-only
#   N3 megamax     6main + th×3 flip-only
#   N4 mega3+BN2   (可选 BN 适应但不用TTA)
# 用法: sbatch scripts/sub_flip_mega.sbatch
# ============================================================
#SBATCH -J cuhkx-flip-mega
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter11 flip-only mega ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

M3="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth"
T3="outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth outputs/pack/thermal_avg_int5.pth"
MS=outputs/pack/main_s42_fold0_int5.pth
TS=outputs/pack/thermal_s42_fold0_int5.pth
MX="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth outputs/pack/main_fold0_int5.pth outputs/pack/main_fold1_int5.pth outputs/pack/main_ps_fold0_int5.pth"
CR="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"

echo "==== [N1] 参照(flip only) ===="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MS --thermal $TS $CR --output outputs/sub_I_ref_flip.csv || echo "⚠️ N1"

echo "==== [N2] mega3(flip only) ===="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $M3 --thermal $T3 $CR --output outputs/sub_I_mega3_flip.csv || echo "⚠️ N2"

echo "==== [N3] megamax(flip only) ===="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main $MX --thermal $T3 $CR --output outputs/sub_I_megamax_flip.csv || echo "⚠️ N3"

echo ""
echo "==== Iter11 DONE ===="
ls -la outputs/sub_I_*.csv 2>/dev/null
echo "判读: N2/N3 若无损于参照(0.73134)的 flip-only 区间 → 纯集成安全加成"
