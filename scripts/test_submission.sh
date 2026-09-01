#!/bin/bash
# ============================================================
# CUHK-X —— 统一打包 + 推理测试脚本（CPU/GPU 自适应，任意 seed 组合）
# 用法（登录节点 CPU，推荐）：
#   CUDA_VISIBLE_DEVICES="" bash scripts/test_submission.sh "42"          # 单 seed
#   CUDA_VISIBLE_DEVICES="" bash scripts/test_submission.sh "42 777"      # 2 seed 集成
#   CUDA_VISIBLE_DEVICES="" bash scripts/test_submission.sh "42 2024 777" # 3 seed
# 用 GPU（计算节点 sbatch 里）：
#   bash scripts/test_submission.sh "42 777"
# 输出: submission_<seeds>.csv（如 submission_42-777.csv，不覆盖主文件）
# ============================================================

SEEDS="$1"
if [ -z "$SEEDS" ]; then
  echo "用法: bash scripts/test_submission.sh \"seed1 seed2 ...\""
  echo "示例: CUDA_VISIBLE_DEVICES=\"\" bash scripts/test_submission.sh \"42 777\""
  exit 1
fi

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

BITS=5
MAX_MB=100
OUT="submission_$(echo "$SEEDS" | tr ' ' '-').csv"

echo "==== seeds=[$SEEDS] bits=$BITS out=$OUT ===="
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}'（unset=自动检测，空=强制CPU）"

# ---- 0) 检查 checkpoint ----
CKPTS=()
for s in $SEEDS; do
  f="outputs/main_full/r2plus1d34_depthir_full_seed${s}.pth"
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
  CKPTS+=("$f")
done
echo "checkpoint: ${CKPTS[*]}"

# ---- 1) 测试 bbox（若不存在）----
if [ ! -f bbox_test.json ]; then
  echo "==== 生成测试 bbox（CPU 较慢 ~10 分钟）===="
  python -m src.detect --mode test --out bbox_test.json
else
  echo "==== bbox_test.json 已存在 ===="
fi

# ---- 2) 打包（每 seed 独立 int5）----
echo "==== 打包 ${#CKPTS[@]} 个 seed 独立 int${BITS} ===="
python scripts/quantize_pack.py --name main --checkpoints "${CKPTS[@]}" \
  --bits $BITS --out_dir outputs/pack --max_mb $MAX_MB

# ---- 3) 推理（多 seed logits 平均，4ch 无帧差）----
echo "==== 推理 ${#CKPTS[@]} seed logits 平均（无 frame_diff, flip TTA）===="
MAIN_ARGS=""
for i in $(seq 0 $((${#CKPTS[@]}-1))); do
  MAIN_ARGS="$MAIN_ARGS outputs/pack/main_fold${i}_int${BITS}.pth"
done
python scripts/ensemble_inference.py --main $MAIN_ARGS \
  --main_backbone r2plus1d34 --quantize \
  --flip_tta --main_crop bbox_test.json \
  --output "$OUT"

echo "==== ALL DONE ===="
python -c "import pandas as pd; df=pd.read_csv('$OUT'); print(f'rows={len(df)} 预测类数={df.prediction.nunique()}')"
