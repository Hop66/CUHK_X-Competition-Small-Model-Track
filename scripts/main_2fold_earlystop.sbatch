#!/bin/bash
# ============================================================
# CUHK-X —— ⭐ 决定性实验：复现公开 0.711 的「2 折 + best-val 早停」recipe
#
# 背景（用户质疑"框架不够好/0.73 是运气"）：
#   - 公开 0.711 notebook：2 折（每折只用一半被试）+ 每折 best-val checkpoint，
#     parent_val_acc = [0.7156, 0.7187]（subject-CV val！）
#   - 我们 0.73 保底 = 全量 80ep 无早停（ep70 达峰后可能过拟合）+ 无 val 选点，
#     main 单 seed42 LB 仅 0.70646 → 比公开 2 折（半数据）还低
#   - thermal 在严格 subject-CV val 上历来拖累 main（PLAN §10.2 实测 0.566<0.59）
#     → 0.73 里 thermal 的 +1.86pt 有 LB 采样运气嫌疑（LB 只 405 clip）
# 假设：main 本体有公开证明的 +1pt 空间（0.706→0.717），来源 = 2折+早停 recipe
# 判读（一锤定音，不烧 LB）：
#   2 折 val 平均 ≥ 0.71 → 对齐公开 → main 本体比全量强 → 2折模型集成重建 0.73→0.74+
#   2 折 val 仍 ~0.66 → recipe 差距不在 fold 结构 → 给最终定论
# 用法: sbatch scripts/main_2fold_earlystop.sbatch
# 产物: outputs/main_2fold_es/r2plus1d34_depthir_fold{0,1}.pth（各为 best-val checkpoint）
# ============================================================
#SBATCH -J cuhkx-main-2fold
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_2fold_es
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 ig65m_r2plus1d34.pth"; exit 1; }
[ -f bbox_train.json ] || { echo "⚠️ 缺 bbox_train.json，先生成"; python -m src.detect --mode train --out bbox_train.json; }

echo "==== 2 折 subject-CV + best-val 早停（对齐公开 0.711 recipe）===="
python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --num_frames 16 --epochs 80 --folds 2 \
  --label_smoothing 0.1 --aug_strength 2 \
  --crop_cache bbox_train.json --save_dir outputs/main_2fold_es 2>&1 | tee "logs/cuhkx-main-2fold-$SLURM_JOB_ID.log"

echo "==== 2 FOLD EARLY-STOP DONE ===="
echo "判读：输出中 'fold 0 best val' / 'fold 1 best val' 的平均"
echo "  ≥ 0.71 → main 本体比全量强 → 用这 2 折模型集成重建 0.73 保底"
echo "  ~0.66 → recipe 差距不在 fold 结构，给最终定论"
