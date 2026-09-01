#!/bin/bash
# ============================================================
# CUHK-X —— MISA 式多任务正则双流 fold0（main+thermal，shared/private 分离）
# 对照：dual_fusion（concat+MLP）fold0、0.73 logits 平均
# 判读：fold0 > 0.68 且 > dual_fusion → MISA 正则有效 → 全量重训
# 前置：ig65m_r2plus1d34.pth + bbox_train.json + bbox_thermal_train.json
# 用法: sbatch scripts/misa_dual.sbatch
# ============================================================
#SBATCH -J cuhkx-misa-dual
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
mkdir -p logs outputs/misa_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
for f in bbox_train.json bbox_thermal_train.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

echo "######## MISA 式多任务正则 双流 fold0（对照 dual_fusion/0.73）########"
python scripts/train_misa_dual.py \
  --fold 0 --folds 3 --weights ig65m_r2plus1d34.pth \
  --epochs 50 --batch_size 16 --lr 1e-4 \
  --main_crop bbox_train.json --thermal_crop bbox_thermal_train.json \
  --save_dir outputs/misa_dual --workers 8
echo "######## MISA 双流完成 ########"

echo "==== MISA DUAL DONE（判读：fold0 > 0.68 且 > dual_fusion → MISA 正则有效）===="
