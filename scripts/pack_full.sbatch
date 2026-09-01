#!/bin/bash
# ============================================================
# CUHK-X —— 全量模型打包 + 推理（多 seed logits 平均）
# 默认：2 个 full seed 独立 int5（80MB）→ 留 20MB 余量（后续可加骨架 int4 ~10MB 等）
# 备选：若 2 seed 不够强，可改 3 个 seed int4（96MB，多样性更好但无余量+误差大）
# 注意：4ch 无帧差推理（不加 --frame_diff！）
# 用法: sbatch scripts/pack_full.sbatch   （在 run_full_main 完成后跑）
# ============================================================
#SBATCH -J cuhkx-full-pack
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/pack
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

MAX_MB=100
# 默认：2 个 seed int5=80MB（留余量）。改 3 seed 时用 BITS=4（96MB）
BITS=5
CKPTS=(outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
       outputs/main_full/r2plus1d34_depthir_full_seed2024.pth)
# 3 seed 备选（int4）：
# BITS=4
# CKPTS=(outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
#        outputs/main_full/r2plus1d34_depthir_full_seed2024.pth \
#        outputs/main_full/r2plus1d34_depthir_full_seed777.pth)

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 0) 检查 checkpoint ----
for f in "${CKPTS[@]}"; do
  [ -f "$f" ] || { echo "❌ 缺少: $f"; exit 1; }
done

# ---- 1) 测试 bbox ----
if [ ! -f bbox_test.json ]; then
  echo "==== [1/4] 生成测试 bbox ===="
  python -m src.detect --mode test --out bbox_test.json
else
  echo "==== [1/4] 测试 bbox 已存在 ===="
fi

# ---- 2) 多 seed 独立 int${BITS} 打包 ----
echo "==== [2/4] ${#CKPTS[@]} 个 full seed 独立 int${BITS} 打包 ===="
python scripts/quantize_pack.py \
  --name main --checkpoints "${CKPTS[@]}" \
  --bits $BITS --out_dir outputs/pack --max_mb $MAX_MB

# ---- 3) 大小检查 ----
PACK_SIZE=$(du -sm outputs/pack/main_fold*_int${BITS}.pth 2>/dev/null | awk '{s+=$1} END {print s+0}')
echo "打包实际总大小: ${PACK_SIZE} MB"
if [ "$PACK_SIZE" -gt "$MAX_MB" ]; then
  echo "❌ 超限 ${PACK_SIZE}MB > ${MAX_MB}MB（改用 2 个 seed int5）"; exit 1
fi

# ---- 4) 推理（4ch 无帧差，多 seed logits 平均）----
echo "==== [3/4] ${#CKPTS[@]} seed logits 平均推理（无 frame_diff） ===="
MAIN_ARGS=""
for i in $(seq 0 $((${#CKPTS[@]}-1))); do
  MAIN_ARGS="$MAIN_ARGS outputs/pack/main_fold${i}_int${BITS}.pth"
done
python scripts/ensemble_inference.py --main $MAIN_ARGS \
  --main_backbone r2plus1d34 --quantize \
  --flip_tta --main_crop bbox_test.json \
  --output submission.csv

echo "==== ALL DONE ===="
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
print(f'rows={len(df)} 预测类数={df.prediction.nunique()}')
print(df['prediction'].value_counts().sort_index().to_string())
"
