#!/bin/bash
# ============================================================
# CUHK-X —— Main 双流 fold1/fold2 复核（判定 +0.5pt 是否稳定，subject-wise 纪律）
# 背景：fold0 S-only 0.6598 / S+M 0.6652(+0.4~0.5pt) → 方向正确但幅度在单折噪声内
# 运行：fold∈{1,2} × {S-only, S+M} 四档，判 mean 差是否稳定 >0
# 判读：mean(S+M)-mean(S) ≥ +0.5pt 且两折同向 → 骨架同源真增益 → 上全量+入 0.73
#       均值差 ≤0 或一正一负 → 增益为噪声 → main 骨架线归档，预算回 thermal vote
# 用法: sbatch scripts/main_dual_folds.sbatch
# ============================================================
#SBATCH -J cuhkx-main-dual-folds
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }
CROP=bbox_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP"; exit 1; }
MC=outputs/motion_cache.pkl
[ -f "$MC" ] || { echo "❌ 缺 $MC（先 extract_motion_cache）"; exit 1; }
EP=60
echo "==== node: $(hostname) | S-only vs S+M  fold1/2 ===="

for FOLD in 1 2; do
  echo "########## fold $FOLD · main S-only ##########"
  python scripts/train_main_dual.py --fold "$FOLD" --no_motion \
    --num_frames 16 --size 128 --epochs $EP --lr 1e-4 --batch_size 32 \
    --weights ig65m_r2plus1d34.pth --crop_cache "$CROP" \
    --save_dir outputs/main_dual --workers 8 || echo "⚠️ fold$FOLD S 失败"

  echo "########## fold $FOLD · main S+M ##########"
  python scripts/train_main_dual.py --fold "$FOLD" \
    --num_frames 16 --size 128 --epochs $EP --lr 1e-4 --batch_size 32 \
    --weights ig65m_r2plus1d34.pth --crop_cache "$CROP" --motion_cache "$MC" \
    --save_dir outputs/main_dual --workers 8 || echo "⚠️ fold$FOLD SM 失败"
done

echo "==== MAIN DUAL FOLDS DONE ===="
echo "看日志里 '== MainDual[SM] fold{1,2} best' 与 'MainDual[S] fold{1,2}' 均值差"
