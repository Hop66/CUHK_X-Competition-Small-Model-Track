#!/bin/bash
# ============================================================
# CUHK-X —— 14th-place thermal baseline 移植对照（fold0，逐组件 A/B）
# 来源: notebooks/cuhk-x-14th-place-0-8-thermal-baseline.ipynb（14 名、0.8+）
#   作者明说这是 pre-optimization 简化版 → 我们不照搬，只移植 2 个可廉价 A/B 的组件：
#     (a) 分段时间采样（全球覆盖分段，train 段内随机 / eval 段中）—— 呼应我们时间 TTA 翻车教训
#     (b) 灰度归一化 (x-0.5)/0.25 替代 Kinetics（thermal 单通道灰度更匹配）
# 对照：thermal 基线 fold0 = 0.6339（Kinetics norm + uniform16 + YOLO crop + aug2）
#   判断：T1 只看点时间采样增量；T2 归一化+采样；T3 再加 8 帧（贴近 notebook）
#   任一 fold0 > 0.64（尤其 >0.65）→ thermal 全量换该配置 → 0.73 融合重现冲 0.8
# 前置：ig65m_r2plus1d34.pth（项目根，与全库其它 sbatch 一致）+ bbox_thermal_train.json + 训练数据
# 用法: sbatch scripts/thermal_notebook14_grid.sbatch
# ============================================================
#SBATCH -J cuhkx-thermal-nb14
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
mkdir -p logs outputs/thermal_notebook14/t0 outputs/thermal_notebook14/t1 \
         outputs/thermal_notebook14/t2 outputs/thermal_notebook14/t3
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重（项目根 ig65m_r2plus1d34.pth；不存在的跑 scripts/download_weights.py）"; exit 1; }
CROP=bbox_thermal_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP（先跑 detect --modality thermal）"; exit 1; }

echo "########## [1/4] T0 基线（uniform16+Kinetics）——复核 0.6339 ##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --crop_cache "$CROP" \
  --save_dir outputs/thermal_notebook14/t0 --workers 8 \
  || echo "⚠️ [1] T0 失败"

echo "########## [2/4] T1 = +分段时间采样（uniform→segment, 16 帧）##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --crop_cache "$CROP" \
  --sample_mode segment \
  --save_dir outputs/thermal_notebook14/t1 --workers 8 \
  || echo "⚠️ [2] T1 失败"

echo "########## [3/4] T2 = segment + 灰度归一化 (0.5,0.25)##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --crop_cache "$CROP" \
  --sample_mode segment --gray_norm \
  --save_dir outputs/thermal_notebook14/t2 --workers 8 \
  || echo "⚠️ [3] T2 失败"

echo "########## [4/4] T3 = segment + gray_norm + 8 帧（贴近 notebook）##########"
python scripts/train_step1.py --modality thermal --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --fold 0 --folds 3 \
  --epochs 60 --batch_size 32 --lr 1e-4 --crop_cache "$CROP" \
  --sample_mode segment --gray_norm --num_frames 8 \
  --save_dir outputs/thermal_notebook14/t3 --workers 8 \
  || echo "⚠️ [4] T3 失败"

echo "==== THERMAL NB14 GRID DONE（判读：T1/T2/T3 fold0 vs 0.6339；>0.64 候选全量, >0.65 优选）===="
