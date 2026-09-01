#!/bin/bash
# ============================================================
# CUHK-X —— 批次1：seed 配比探索（main42/777 × thermal42/777 四组合）
# 背景：0.73 = main42 + thermal42（已提交，作对照不重跑）。main 单 seed42=0.70646，
#       换 seed（main777 / thermal777）可能找到 >0.73 组合。零训练成本。
# 流程：打包 4 个 seed（main/thermal × 42/777）→ 跑 3 个新组合 → 3 个 submission
#       （m42_t42 = 0.73 已交，不重跑）
# 用法: sbatch scripts/submit_fusion_seed_sweep.sbatch
# ============================================================
#SBATCH -J cuhkx-seedsweep
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/pack
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BITS=5
# ---- 打包 4 个 seed（main/thermal × 42/777，不同 name 防覆盖）----
python scripts/quantize_pack.py --name main_s42  --checkpoints outputs/main_full/r2plus1d34_depthir_full_seed42.pth  --bits $BITS --out_dir outputs/pack
python scripts/quantize_pack.py --name main_s777 --checkpoints outputs/main_full/r2plus1d34_depthir_full_seed777.pth --bits $BITS --out_dir outputs/pack
python scripts/quantize_pack.py --name thermal_s42  --checkpoints outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth  --bits $BITS --out_dir outputs/pack
python scripts/quantize_pack.py --name thermal_s777 --checkpoints outputs/thermal_full/r2plus1d34_thermal_full_seed777.pth --bits $BITS --out_dir outputs/pack
echo "==== 4 seed 打包完成 ===="

# ---- 测试 bbox（无则生成）----
[ -f bbox_test.json ] || python -m src.detect --mode test --out bbox_test.json
[ -f bbox_thermal_test.json ] || python -m src.detect --mode test --out bbox_thermal_test.json --modality thermal

# ---- 跑 4 个融合组合 ----
run_fusion() { # $1=main_seed $2=thermal_seed $3=out_csv
  echo "==== main_s$1 + thermal_s$2 → $3 ===="
  python scripts/ensemble_inference.py \
    --main outputs/pack/main_s$1_fold0_int${BITS}.pth \
    --thermal outputs/pack/thermal_s$2_fold0_int${BITS}.pth \
    --quantize --flip_tta --prob_avg \
    --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
    --output "$3"
}

run_fusion 42 777 submission_fusion_m42_t777.csv
run_fusion 777 42 submission_fusion_m777_t42.csv
run_fusion 777 777 submission_fusion_m777_t777.csv

echo "==== SEED SWEEP DONE（3 新组合；m42_t42=0.73 已交不重跑）===="
python -c "
import pandas as pd
for f in ['submission_fusion_m42_t777.csv','submission_fusion_m777_t42.csv','submission_fusion_m777_t777.csv']:
    df = pd.read_csv(f)
    print(f'{f}: rows={len(df)} 类数={df.prediction.nunique()} 类0={int((df.prediction==0).sum())}')
"
