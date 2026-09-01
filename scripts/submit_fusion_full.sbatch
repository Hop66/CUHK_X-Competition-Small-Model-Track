#!/bin/bash
# ============================================================
# CUHK-X —— 方案 A：满预算 main 双 seed + thermal（冲 0.73+）
# 背景：0.73 = main 单 seed42 int5 + thermal 单 seed42 int5（80MB），+1.86pt vs 0.71144
# 此脚本：main 双 seed(42+777) + thermal(42)，用满 100MB 余量
#
# 预算账（bit-packing，估算）：
#   main int5 单 seed = 40MB → main int5 双 seed = 80MB
#   thermal int5 = 40MB → int4 ≈ 32MB
#   配比[1] main int5 双 + thermal int4 = ~112MB ❌ 超
#   配比[2] main int4 双 + thermal int4 = ~96MB ✅（默认命中）
# 脚本自适应：先试 [1]，du 实测超限则自动降 main 到 int4 走 [2]。
#
# 注意：main 双 seed 不 --average（铁律：每 seed 独立打包，推理时 logits 平均，防预测塌缩）。
# 用法: sbatch scripts/submit_fusion_full.sbatch
# 产物: outputs/pack/main_fold{0,1}_int{4,5}.pth + thermal_fold0_int4.pth
#       + submission_fusion_full.csv
# ============================================================
#SBATCH -J cuhkx-fusion-full
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
MAIN_CKPTS=(outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
            outputs/main_full/r2plus1d34_depthir_full_seed777.pth)
TH_CKPT=outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth

for f in "${MAIN_CKPTS[@]}" "$TH_CKPT"; do
  [ -f "$f" ] || { echo "❌ 缺少: $f"; exit 1; }
done
echo "main 双 seed: ${MAIN_CKPTS[0]} + ${MAIN_CKPTS[1]}"
echo "thermal:      $TH_CKPT"

# ---- 0) 测试 bbox ----
if [ ! -f bbox_test.json ]; then
  echo "==== 生成测试 bbox ===="
  python -m src.detect --mode test --out bbox_test.json
fi
if [ ! -f bbox_thermal_test.json ]; then
  echo "==== 生成测试 thermal bbox ===="
  python -m src.detect --mode test --out bbox_thermal_test.json --modality thermal
fi

# ---- 1) thermal int4 打包（固定 32MB）----
echo "==== thermal 单 seed int4 打包 ===="
python scripts/quantize_pack.py --name thermal --checkpoints "$TH_CKPT" \
  --bits 4 --out_dir outputs/pack --max_mb "$MAX_MB"

# ---- 2) main 双 seed int5 打包（独立，不用 SWA）----
echo "==== main 双 seed int5 打包 ===="
python scripts/quantize_pack.py --name main \
  --checkpoints "${MAIN_CKPTS[0]}" "${MAIN_CKPTS[1]}" \
  --bits 5 --out_dir outputs/pack --max_mb "$MAX_MB"

# ---- 3) 大小检查，超限降 main 到 int4 ----
size_mb() { du -sm "$@" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
MAIN5=$(size_mb outputs/pack/main_fold0_int5.pth outputs/pack/main_fold1_int5.pth)
TH4=$(size_mb outputs/pack/thermal_fold0_int4.pth)
TOTAL=$((MAIN5 + TH4))
echo "配比[1] main int5 双(${MAIN5}MB) + thermal int4(${TH4}MB) = ${TOTAL}MB"

MAIN_BITS=5
if [ "$TOTAL" -gt "$MAX_MB" ]; then
  echo "⚠️ 配比[1] 超限 ${TOTAL}MB > ${MAX_MB}MB → 降 main 到 int4"
  python scripts/quantize_pack.py --name main \
    --checkpoints "${MAIN_CKPTS[0]}" "${MAIN_CKPTS[1]}" \
    --bits 4 --out_dir outputs/pack --max_mb "$MAX_MB"
  MAIN_BITS=4
  MAIN4=$(size_mb outputs/pack/main_fold0_int4.pth outputs/pack/main_fold1_int4.pth)
  TOTAL=$((MAIN4 + TH4))
  echo "配比[2] main int4 双(${MAIN4}MB) + thermal int4(${TH4}MB) = ${TOTAL}MB"
fi

if [ "$TOTAL" -gt "$MAX_MB" ]; then
  echo "❌ 最终仍超限 ${TOTAL}MB > ${MAX_MB}MB"; exit 1
fi
echo "✅ 合规：main int${MAIN_BITS} 双 seed + thermal int4 = ${TOTAL}MB（上限 ${MAX_MB}MB）"

# ---- 4) 融合推理（prob 平均 + flip TTA）----
echo "==== main int${MAIN_BITS} 双 seed + thermal int4 融合推理 ===="
python scripts/ensemble_inference.py \
  --main outputs/pack/main_fold0_int${MAIN_BITS}.pth \
        outputs/pack/main_fold1_int${MAIN_BITS}.pth \
  --thermal outputs/pack/thermal_fold0_int4.pth \
  --quantize --flip_tta \
  --prob_avg \
  --main_crop bbox_test.json \
  --thermal_crop bbox_thermal_test.json \
  --output submission_fusion_full.csv

echo "==== ALL DONE ===="
python -c "
import pandas as pd
df = pd.read_csv('submission_fusion_full.csv')
print(f'rows={len(df)} 预测类数={df.prediction.nunique()}')
print(df['prediction'].value_counts().sort_index().to_string())
"
echo "提交文件: submission_fusion_full.csv（对比 0.73 = submission_fusion_mT.csv）"
