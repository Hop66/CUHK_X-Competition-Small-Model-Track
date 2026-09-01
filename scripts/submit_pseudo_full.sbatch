#!/bin/bash
# ============================================================
# CUHK-X —— 批次2：伪标签全量自训练 + 融合提交（烧 LB）
# 背景：fold0 伪标签微降（thermal -0.64 / main -0.43，在 std 0.027 噪声内），
#       但 fold0 是训练分布，伪标签针对测试分布 → 正确评价 = 全量 + LB。
# 流程：pseudo(现成) → selftrain --full main/thermal(80ep 含 120 伪标签低权重)
#       → 打包 int5 → main+thermal 融合推理 → 提交
# 大小：main int5(40) + thermal int5(40) = 80MB ✅
# 用法: sbatch scripts/submit_pseudo_full.sbatch
# ============================================================
#SBATCH -J cuhkx-pseudo-full
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/pack outputs/selftrain
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }
for f in bbox_train.json bbox_test.json bbox_thermal_train.json bbox_thermal_test.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

# ============ [1/4] 打伪标签（若已存在则跳过，用现成）============
if [ ! -f outputs/pseudo_labels.json ]; then
  echo "######## [1/4] 打伪标签 ########"
  python scripts/pseudo_label_selftrain.py --stage pseudo
fi
echo "==== 伪标签: $(python -c "import json;print(len(json.load(open('outputs/pseudo_labels.json'))))") 个 ===="

# ============ [2/4] 全量自训练 thermal（80ep + 伪标签低权重）============
echo "######## [2/4] thermal 全量自训练 ########"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality thermal --full --crop_cache bbox_thermal_train.json \
  --weights ig65m_r2plus1d34.pth --epochs 80 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain --workers 8
echo "######## [2/4] thermal 完成 ########"

# ============ [3/4] 全量自训练 main（80ep + 伪标签低权重）============
echo "######## [3/4] main 全量自训练 ########"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality main --full --crop_cache bbox_train.json \
  --weights ig65m_r2plus1d34.pth --epochs 80 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain --workers 8
echo "######## [3/4] main 完成 ########"

# ============ [4/4] 打包 + 融合提交 ============
echo "######## [4/4] 打包 int5 + 融合推理 ########"
python scripts/quantize_pack.py --name main_ps  --checkpoints outputs/selftrain/main_selftrain_full.pth  --bits 5 --out_dir outputs/pack
python scripts/quantize_pack.py --name thermal_ps --checkpoints outputs/selftrain/thermal_selftrain_full.pth --bits 5 --out_dir outputs/pack

# ---- 提交总大小汇总检查（两模型相加才是真实提交大小）----
echo "==== 提交总大小检查 ===="
PACK_SIZE=$(du -sm outputs/pack/main_ps_fold0_int5.pth outputs/pack/thermal_ps_fold0_int5.pth | awk '{s+=$1} END {print s+0}')
echo "main_ps + thermal_ps 总大小: ${PACK_SIZE} MB（上限 100 MB）"
if [ "$PACK_SIZE" -gt 100 ]; then
  echo "❌ 超限 ${PACK_SIZE}MB！改用 int4 或减模型"; exit 1
fi
echo "✅ 合规，剩余 $((100 - PACK_SIZE)) MB"

python scripts/ensemble_inference.py \
  --main outputs/pack/main_ps_fold0_int5.pth \
  --thermal outputs/pack/thermal_ps_fold0_int5.pth \
  --quantize --flip_tta --prob_avg \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output submission_pseudo_full.csv

echo "==== PSEUDO FULL DONE（对比 0.73 保底 = submission_fusion_mT.csv）===="
python -c "
import pandas as pd
df = pd.read_csv('submission_pseudo_full.csv')
print(f'rows={len(df)} 类数={df.prediction.nunique()} 类0={int((df.prediction==0).sum())}')
"
