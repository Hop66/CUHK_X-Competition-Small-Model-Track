#!/bin/bash
# ============================================================
# CUHK-X —— NTU 骨架单独训练诊断（验证 NTU 数据/映射有效性）
# 判读：NTU val > 0.7 → 数据有效，multi-task 方法问题 → 改两阶段预训练
#       NTU val ~0.5 → 数据/映射问题 → 先修数据管线
# 用法: sbatch scripts/ntu_diag.sbatch
# ============================================================
#SBATCH -J cuhkx-ntu-diag
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/ntu_diag
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 MotionBERT 权重"; exit 1; }
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
[ "$(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' 2>/dev/null | wc -l)" -gt 0 ] \
  || { echo "❌ 缺 NTU 骨架（$NTU_DIR）"; exit 1; }

echo "######## NTU 单独训练（NTU 内部按被试划分，lr=1e-3 StepLR）########"
python scripts/train_ntu_only.py \
  --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 --lr 1e-3 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --save_dir outputs/ntu_diag --workers 8
echo "######## NTU 诊断完成 ########"

echo "==== NTU DIAG DONE（判读：>0.7 数据有效→两阶段；~0.5 数据问题→修管线）===="
