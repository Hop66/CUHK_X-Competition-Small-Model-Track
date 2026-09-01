#!/bin/bash
# ============================================================
# CUHK-X —— main 损失×增强网格（可靠评估协议：全部 3 折 mean±std）
# 背景（用户方法论质疑）：单折噪声大（std 0.015-0.027），很多"无效"结论可能是误判。
#   aug_search fold0 显示 aug1(0.6652) > aug2(0.6609)；随机 vs 固定未验证。
# 网格（同结构/同 scheduler StepLR/同协议，三方可比）：
#   [1] CE baseline aug1   [2] CE baseline aug2   ← ArcFace 可靠对照
#   [3] ArcFace    aug1   [4] ArcFace    aug2   ← 损失层面跨身份
#   [5] CenterLoss aug2                         ← 损失层面（类内紧凑）
# 判读：同 aug 下，ArcFace/CenterLoss 的 3 折 mean 与 CE baseline 同 aug 对比，均值差>std 才有效
# 前置：ig65m_r2plus1d34.pth + bbox_train.json（独立于正在跑的 rerun）
# 用法: sbatch scripts/main_loss_sweep.sbatch
# ============================================================
#SBATCH -J cuhkx-loss-sweep
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_baseline outputs/main_arcface outputs/main_centerloss
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
[ -f bbox_train.json ] || { echo "❌ 缺 bbox_train.json"; exit 1; }
echo "==== 前置 OK（本脚本独立于 rerun，可排队）===="

# [1] CE baseline aug1（3 折）
echo "########## [1/5] CE baseline aug1 3折 ##########"
python scripts/train_main_baseline.py --fold -1 --folds 3 \
  --weights ig65m_r2plus1d34.pth --epochs 60 --batch_size 32 --lr 1e-4 \
  --aug_strength 1 --crop_cache bbox_train.json \
  --save_dir outputs/main_baseline --workers 8 \
  || echo "⚠️ [1] 失败"

# [2] CE baseline aug2（3 折）
echo "########## [2/5] CE baseline aug2 3折 ##########"
python scripts/train_main_baseline.py --fold -1 --folds 3 \
  --weights ig65m_r2plus1d34.pth --epochs 60 --batch_size 32 --lr 1e-4 \
  --aug_strength 2 --crop_cache bbox_train.json \
  --save_dir outputs/main_baseline --workers 8 \
  || echo "⚠️ [2] 失败"

# [3] ArcFace aug1（3 折）
echo "########## [3/5] ArcFace aug1 3折 ##########"
python scripts/train_main_arcface.py --fold -1 --folds 3 \
  --weights ig65m_r2plus1d34.pth --epochs 60 --batch_size 32 --lr 1e-4 \
  --s 30.0 --m 0.5 --aug_strength 1 --crop_cache bbox_train.json \
  --save_dir outputs/main_arcface --workers 8 \
  || echo "⚠️ [3] 失败"

# [4] ArcFace aug2（3 折）
echo "########## [4/5] ArcFace aug2 3折 ##########"
python scripts/train_main_arcface.py --fold -1 --folds 3 \
  --weights ig65m_r2plus1d34.pth --epochs 60 --batch_size 32 --lr 1e-4 \
  --s 30.0 --m 0.5 --aug_strength 2 --crop_cache bbox_train.json \
  --save_dir outputs/main_arcface --workers 8 \
  || echo "⚠️ [4] 失败"

# [5] CenterLoss aug2（3 折）
echo "########## [5/6] CenterLoss aug2 3折 ##########"
python scripts/train_main_centerloss.py --fold -1 --folds 3 \
  --weights ig65m_r2plus1d34.pth --epochs 60 --batch_size 32 --lr 1e-4 \
  --lambda_c 1e-3 --aug_strength 2 --crop_cache bbox_train.json \
  --save_dir outputs/main_centerloss --workers 8 \
  || echo "⚠️ [5] 失败"

# [6] 对比学习 aug1 多折（重测：历史 s1+对比 fold0=0.6771 +1.2% 有效，被 aug2 定版掩盖）
echo "########## [6/6] 跨模态对比学习 aug1 多折（重测）##########"
python scripts/train_contrastive.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --lr 1e-4 --epochs 60 --folds 3 --fold -1 --aug_strength 1 --lambda_contrast 0.1 \
  --crop_cache bbox_train.json --thermal_crop bbox_thermal_train.json \
  --save_dir outputs/contrastive --workers 4 \
  || echo "⚠️ [6] 对比学习失败"

echo "############################################################"
echo "LOSS SWEEP DONE（job ${SLURM_JOB_ID}）"
echo "对比表（同 aug 同协议）："
echo "  aug1: CE mean±std vs ArcFace mean±std vs 对比学习 mean±std"
echo "  aug2: CE mean±std vs ArcFace mean±std vs CenterLoss mean±std"
echo "判读：均值差 > std → 方法有效；也确认 aug1 vs aug2 谁更好"
echo "############################################################"
