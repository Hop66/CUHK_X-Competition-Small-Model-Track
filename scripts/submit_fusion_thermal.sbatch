#!/bin/bash
# ============================================================
# CUHK-X —— 提交级融合：单 seed main + 单 seed thermal → 烧 LB
# 验证问题："把 1 个 main seed 换成 thermal seed（独立视角）能否超过 2×main 0.71144？"
# 大小账：main int5(40MB) + thermal int5(40MB) = 80MB ✅（和 2×main 一样）
# 用法: sbatch scripts/submit_fusion_thermal.sbatch   （先确认 main/thermal 全量模型存在）
# 产物: outputs/pack/main_*.pth + outputs/pack/thermal_*.pth + submission_fusion_mT.csv
# ============================================================
#SBATCH -J cuhkx-fusion-sub
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

MAX_MB=100
BITS=5
# ---- 选 seed：main 用最好的 1 个（42 或 777），thermal 用最好的 1 个 ----
MAIN_SEED=42
TH_SEED=42
MAIN_CKPT=outputs/main_full/r2plus1d34_depthir_full_seed${MAIN_SEED}.pth
TH_CKPT=outputs/thermal_full/r2plus1d34_thermal_full_seed${TH_SEED}.pth

for f in "$MAIN_CKPT" "$TH_CKPT"; do
  [ -f "$f" ] || { echo "❌ 缺少: $f"; exit 1; }
done
echo "main:    $MAIN_CKPT"
echo "thermal: $TH_CKPT"

# ---- 0) 测试 bbox ----
if [ ! -f bbox_test.json ]; then
  echo "==== 生成测试 bbox ===="
  python -m src.detect --mode test --out bbox_test.json
fi
if [ ! -f bbox_thermal_test.json ]; then
  echo "==== 生成测试 thermal bbox ===="
  python -m src.detect --mode test --out bbox_thermal_test.json --modality thermal
fi

# ---- 1) main 单 seed int5 打包（独立，不用 SWA）----
echo "==== main seed${MAIN_SEED} int${BITS} 打包 ===="
python scripts/quantize_pack.py --name main \
  --checkpoints "$MAIN_CKPT" \
  --bits $BITS --out_dir outputs/pack --max_mb $MAX_MB

# ---- 2) thermal 单 seed int5 打包 ----
echo "==== thermal seed${TH_SEED} int${BITS} 打包 ===="
python scripts/quantize_pack.py --name thermal \
  --checkpoints "$TH_CKPT" \
  --bits $BITS --out_dir outputs/pack --max_mb $MAX_MB

# ---- 3) 大小检查 ----
echo "==== 大小检查 ===="
du -sh outputs/pack/main_fold*_int${BITS}.pth outputs/pack/thermal_fold*_int${BITS}.pth
PACK_SIZE=$(du -sm outputs/pack/main_fold*_int${BITS}.pth outputs/pack/thermal_fold*_int${BITS}.pth | awk '{s+=$1} END {print s+0}')
echo "main+thermal 总大小: ${PACK_SIZE} MB（上限 $MAX_MB MB）"
if [ "$PACK_SIZE" -gt "$MAX_MB" ]; then
  echo "❌ 超限！改用 int4 或减模型"; exit 1
fi

# ---- 4) main+thermal 融合推理（概率平均，flip TTA；⚠️ 不用 time_tta：连续窗口采样丢时间多样性降质）----
echo "==== main + thermal 融合推理（prob_avg + flip TTA）===="
python scripts/ensemble_inference.py \
  --main outputs/pack/main_fold0_int${BITS}.pth \
  --thermal outputs/pack/thermal_fold0_int${BITS}.pth \
  --quantize --flip_tta \
  --prob_avg \
  --main_crop bbox_test.json \
  --thermal_crop bbox_thermal_test.json \
  --output submission_fusion_mT.csv

echo "==== ALL DONE ===="
python -c "
import pandas as pd
df = pd.read_csv('submission_fusion_mT.csv')
print(f'rows={len(df)} 预测类数={df.prediction.nunique()}')
print(df['prediction'].value_counts().sort_index().to_string())
"
