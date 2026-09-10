#!/bin/bash
# ============================================================
# CUHK-X Iter2(推理侧叠加): 多裁剪×时窗×BN适应×SWA/Mega 全套 AB
# 全部 prob_avg 融合; 逐单元隔离贡献, 出 CSV 待 LB
#   F1 BN only        (main_avg+th_avg, 1crop×1time, bn2)
#   F2 多裁剪+时窗    (main_avg+th_avg, 4crop×2time, 无BN)
#   F3 全叠加         (main_avg+th_avg, 4crop×2time, bn1)
#   F4 参照链+多裁剪  (main_s42+th_s42, 4crop×2time)
#   F5 Mega最大集+裁剪 (6main×3 + th_avg, 3crop×1time)
# 用法: sbatch scripts/ab_multicrop_stack.sbatch
# ============================================================
#SBATCH -J cuhkx-mc-stack
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter2 MultiCrop Stack ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MA="outputs/pack/main_avg_int5.pth"
TA="outputs/pack/thermal_avg_int5.pth"
TS="outputs/pack/thermal_s42_fold0_int5.pth"
MS="outputs/pack/main_s42_fold0_int5.pth"
MEGAMAX="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth outputs/pack/main_fold0_int5.pth outputs/pack/main_fold1_int5.pth outputs/pack/main_ps_fold0_int5.pth"

echo "==== [F1] BN only ===="
python scripts/infer_multicrop.py --main $MA --thermal $TA --quantize \
  --multicrop 1 --time_tta_n 1 --bn_passes 2 --flip --output outputs/sub_F1_bn.csv || echo "⚠️ F1"

echo "==== [F2] 多裁剪+时窗 ===="
python scripts/infer_multicrop.py --main $MA --thermal $TA --quantize \
  --multicrop 4 --time_tta_n 2 --bn_passes 0 --flip --output outputs/sub_F2_tta.csv || echo "⚠️ F2"

echo "==== [F3] 全叠加 ===="
python scripts/infer_multicrop.py --main $MA --thermal $TA --quantize \
  --multicrop 4 --time_tta_n 2 --bn_passes 1 --flip --output outputs/sub_F3_all.csv || echo "⚠️ F3"

echo "==== [F4] 参照链 0.73 + 多裁剪时窗 ===="
python scripts/infer_multicrop.py --main $MS --thermal $TS --quantize \
  --multicrop 4 --time_tta_n 2 --bn_passes 0 --flip --output outputs/sub_F4_ref_tta.csv || echo "⚠️ F4"

echo "==== [F5] Mega最大集 + 3crop ===="
python scripts/infer_multicrop.py --main $MEGAMAX --thermal $TA --quantize \
  --multicrop 3 --time_tta_n 1 --bn_passes 0 --flip --output outputs/sub_F5_mega3crop.csv || echo "⚠️ F5"

echo ""
echo "==== Iter2 DONE: 待 LB ===="
ls -la outputs/sub_F*.csv 2>/dev/null
