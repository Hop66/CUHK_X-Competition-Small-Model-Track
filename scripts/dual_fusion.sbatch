#!/bin/bash
# ============================================================
# CUHK-X —— main+thermal 双流特征融合 fold0（特征级中融合，对照 0.73 logits 平均）
# 判读：fold0 > 0.68 → 特征级融合有效 → 全量双流重训 → 可能超越 logits 平均
# 前置：ig65m_r2plus1d34.pth + bbox_train.json + bbox_thermal_train.json
# 用法: sbatch scripts/dual_fusion.sbatch
# ============================================================
#SBATCH -J cuhkx-dualfuse
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/dual_fusion
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
for f in bbox_train.json bbox_thermal_train.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

echo "######## main+thermal 双流特征融合 fold0（对照 0.73）########"
python scripts/train_dual_fusion.py \
  --fold 0 --folds 3 --weights ig65m_r2plus1d34.pth \
  --epochs 50 --batch_size 16 --lr 1e-4 \
  --main_crop bbox_train.json --thermal_crop bbox_thermal_train.json \
  --save_dir outputs/dual_fusion --workers 8
echo "######## 双流融合完成 ########"

echo "==== DUAL FUSION DONE（判读：fold0 > 0.68 → 特征级融合有效）===="
