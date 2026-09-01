#!/bin/bash
# ============================================================
# CUHK-X —— 综合执行：伪标签融合（放前）+ 重跑失败/待跑实验（2026-08-27）
#
# 顺序：
#   [1] 伪标签融合（最高 EV）：原始 0.73 int5 + selftrain_full fp32 → submission_fusion.csv
#   [2] v3 bounded-prior（需最新 ensemble_inference.py，含 --prior_experiment）
#   [3] YOLO pose 特征 + GBDT（yolo_pose_kpts 形状 bug 已修；装 sklearn）
#   [4] 骨架 Cosine 基线（恢复原版 0.5447 来源；StepLR 0.5113 已证伪）
#   [5] 骨架 Cosine + 清洗（对照 [4]）
#   [6] 骨架 Cosine + part-aware（对照 [5]）
#   [7] MISA 修复版（recon 标准化 + CMD eps，NaN 已修）
#
# 前置：最新 ensemble_inference.py / fuse_selftrain.py / train_misa_dual.py 已上传！
# 用法: sbatch scripts/fusion_and_rerun.sbatch
# ============================================================
#SBATCH -J cuhkx-fusion-rerun
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/feats outputs/skeleton_aug_cos \
         outputs/skeleton_aug_cos_clean outputs/skeleton_aug_cos_part outputs/misa_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MISSING=0
for f in ig65m_r2plus1d34.pth weights/mb_lite_latest_epoch.bin \
         outputs/pack/main_s42_fold0_int5.pth outputs/pack/thermal_s42_fold0_int5.pth \
         outputs/selftrain_full/main_selftrain_full.pth outputs/selftrain_full/thermal_selftrain_full.pth \
         bbox_train.json bbox_test.json bbox_thermal_train.json bbox_thermal_test.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; MISSING=1; }
done
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
NTU_OK=0
if [ -d "$NTU_DIR" ] && [ "$(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' 2>/dev/null | wc -l)" -gt 0 ]; then NTU_OK=1; fi
[ "$MISSING" -eq 0 ] || { echo "❌ 关键文件缺失，终止"; exit 1; }
echo "==== 前置检查通过（NTU_OK=$NTU_OK）===="

# ============================================================
echo "########## [1/7] 伪标签融合（最高 EV，原始 0.73 + selftrain_full）##########"
# ============================================================
python scripts/fuse_selftrain.py \
  --main_orig outputs/pack/main_s42_fold0_int5.pth \
  --thermal_orig outputs/pack/thermal_s42_fold0_int5.pth \
  --main_st outputs/selftrain_full/main_selftrain_full.pth \
  --thermal_st outputs/selftrain_full/thermal_selftrain_full.pth \
  --w_orig 1.0 --w_st 1.0 --flip_tta \
  --output outputs/feats/submission_fusion.csv \
  || echo "⚠️ [1] 融合失败"
echo "✅ [1] 完成 → submission_fusion.csv（判读：提交后 LB > 0.73 → 伪标签有效）"

# ============================================================
echo "########## [2/7] v3 bounded-prior（需最新 ensemble_inference.py）##########"
# ============================================================
python scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth \
  --quantize --flip_tta --prob_avg \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --prior_experiment --output outputs/feats/submission_main.csv \
  || echo "⚠️ [2] prior 失败（确认已上传最新 ensemble_inference.py）"
echo "✅ [2] 完成（判读：soft-count imbalance ≥2 → 烧 spare 提交 submission_main_prior.csv）"

# ============================================================
echo "########## [3/7] YOLO pose 特征 + GBDT ##########"
# ============================================================
echo "[3] 安装 scikit-learn（GBDT 需要）..."
pip install -q scikit-learn || echo "⚠️ [3] sklearn 安装失败（GBDT 将跳过）"
python scripts/yolo_pose_kpts.py --train_root ~/Multimodal/data/Training/HAR \
  --out outputs/feats/kpts_feats_train.csv --stride 2 \
  || echo "⚠️ [3] pose 特征失败"
python scripts/eval_feats_gbdt.py --csv outputs/feats/kpts_feats_train.csv \
  || echo "⚠️ [3] pose GBDT 失败"
echo "✅ [3] 完成（判读：GBDT acc 显著 >2.5% 随机 → 部位信号）"

# ============================================================
if [ "$NTU_OK" -eq 1 ]; then
echo "########## [4/7] 骨架 Cosine 基线（恢复原版 0.5447 来源）##########"
# ============================================================
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --save_dir outputs/skeleton_aug_cos --workers 8 \
  || echo "⚠️ [4] 骨架基线失败"
echo "✅ [4] 完成（判读：与 0.5447 对照确认；后续 clean/part 对照它）"

# ============================================================
echo "########## [5/7] 骨架 Cosine + 清洗 ##########"
# ============================================================
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --clean --save_dir outputs/skeleton_aug_cos_clean --workers 8 \
  || echo "⚠️ [5] 骨架清洗失败"
echo "✅ [5] 完成（判读：fold0 > [4] Cosine 基线 → 清洗有效）"

# ============================================================
echo "########## [6/7] 骨架 Cosine + part-aware ##########"
# ============================================================
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --clean --part_aware --save_dir outputs/skeleton_aug_cos_part --workers 8 \
  || echo "⚠️ [6] 骨架 part 失败"
echo "✅ [6] 完成（判读：fold0 > [5] clean → part 有效）"
else
  echo "⚠️ [4][5][6] 跳过（无 NTU 骨架）"
fi

# ============================================================
echo "########## [7/7] MISA 修复版（recon 标准化 + CMD eps）##########"
# ============================================================
python scripts/train_misa_dual.py \
  --fold 0 --folds 3 --weights ig65m_r2plus1d34.pth \
  --epochs 50 --batch_size 16 --lr 1e-4 \
  --main_crop bbox_train.json --thermal_crop bbox_thermal_train.json \
  --save_dir outputs/misa_dual --workers 8 \
  || echo "⚠️ [7] MISA 失败"
echo "✅ [7] 完成（判读：fold0 >0.66 留 / <0.66 关；NaN 已修应能正常收敛）"

echo "############################################################"
echo "ALL DONE（job ${SLURM_JOB_ID}）— 汇总判读："
echo "  [1] 融合: LB>0.73 → 伪标签有效（烧 LB，最高优先）"
echo "  [2] prior: imbalance≥2 → 烧 spare 提交"
echo "  [3] GBDT: acc>2.5% → 部位信号"
echo "  [4][5][6] 骨架 Cosine: clean/part 对照 Cosine 基线"
echo "  [7] MISA: fold0>0.66 → 留，否则关"
echo "############################################################"
