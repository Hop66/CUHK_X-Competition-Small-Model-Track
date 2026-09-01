#!/bin/bash
# ============================================================
# CUHK-X —— 批次3：VideoMAE-S fold0 验证（thermal）
# 自监督预训练（K400）域鲁棒，可能是 thermal（弱模态）的更强骨干。
# 对照：R2Plus1D-34 thermal fold0 = 0.6339
# 前置：weights/videomae_s_k400_pretrain.pth（本地 VPN 下载后传服务器）
# 用法: sbatch scripts/videomae.sbatch
# ============================================================
#SBATCH -J cuhkx-videomae
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
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/videomae_s_k400_pretrain.pth ] || { echo "❌ 缺 VideoMAE-S 权重"; exit 1; }
[ -f bbox_thermal_train.json ] || { echo "❌ 缺 bbox_thermal_train.json"; exit 1; }

echo "######## VideoMAE-S thermal fold0 60ep（对照 0.6339）########"
python scripts/train_videomae.py \
  --ckpt weights/videomae_s_k400_pretrain.pth \
  --epochs 60 --fold 0 --folds 3 \
  --crop_cache bbox_thermal_train.json --lr 1e-4 \
  --save_dir outputs/videomae --workers 8
echo "######## VideoMAE-S 完成 ########"

echo "==== VIDEO MAE DONE（判读：fold0 > 0.65 则骨干有效 → thermal 全量换 VideoMAE-S）===="
