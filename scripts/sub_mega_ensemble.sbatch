#!/bin/bash
# ============================================================
# CUHK-X 冲刺0.8 · Iter1:  Mega-ensemble(纯推理,复用所有已训 pack) + TTA4
# 原理: 跨被试下,多 seed/fold 独立模型 logit 平均是免费泛化增量(通常+0.5~1.5pt)
# 单元(prob_avg + flip, time_tta=4 offset):
#   D2: 3main(  s42/s777/avg ) + 3thermal( s42/s777/avg )   -> sub_D_mega3_tta4
#   D3: 3main + 3th + fold( fold0/fold1 main + ps ) 最大集   -> sub_D_megamax_tta4
#   D4: 6main + 3th( 再加 main_fold0/fold1 + main_ps )      -> sub_D_mega6_tta4
# 参照: sub_C_ref_tta4 (main_s42+th_s42) 已有
# 用法: sbatch scripts/sub_mega_ensemble.sbatch
# ============================================================
#SBATCH -J cuhkx-mega
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

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter1 Mega-ensemble ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CR="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"
BASE="python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize --time_tta 4 --time_tta_mode offset $CR"

M3="outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth outputs/pack/main_avg_int5.pth"
T3="outputs/pack/thermal_s42_fold0_int5.pth outputs/pack/thermal_s777_fold0_int5.pth outputs/pack/thermal_avg_int5.pth"

echo "==== [D2] 3main+3th, TTA4 ===="
$BASE --main $M3 --thermal $T3 --output outputs/sub_D_mega3_tta4.csv || echo "⚠️ D2"

echo "==== [D3] 3main+3th+main_fold0/fold1+main_ps, TTA4 (最大集) ===="
MAX="outputs/pack/main_fold0_int5.pth outputs/pack/main_fold1_int5.pth outputs/pack/main_ps_fold0_int5.pth"
$BASE --main $M3 $MAX --thermal $T3 --output outputs/sub_D_megamax_tta4.csv || echo "⚠️ D3"

echo ""
echo "==== Iter1 DONE ===="
ls -la outputs/sub_D_*.csv 2>/dev/null
echo "待LB: 与 sub_C_ref_tta4(参照)对比; any >0.75 → 锁定 Mega+TTA 为提交主链"
