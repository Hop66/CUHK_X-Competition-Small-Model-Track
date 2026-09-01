#!/bin/bash
# ============================================================
# CUHK-X —— 主控：一次提交跑完所有待做实验（串行，2026-08-26）
#
# 顺序（按投入产出比）：
#   [1] 伪标签全量自训练（最高 EV：同分布 405 数据，官方允许）
#   [2] v3 bounded-prior（零训练，类偏置修正，只看 imbalance 判据）
#   [3] YOLO 部位特征 + GBDT（bbox 运动特征 + YOLO-pose 关键点）
#   [4] 骨架清洗（对照 0.5447）
#   [5] 骨架 part-aware 部位权重（对照 clean-only）
#   [6] MISA 双流多任务正则（低预期，跑一次定生死）
#
# 原则：单个实验失败不阻断后续（set -uo pipefail，不用 -e）
# 判读：每步结束打印判读标准；结果汇总看日志尾部
# 用法: sbatch scripts/run_all_experiments.sbatch
# ============================================================
#SBATCH -J cuhkx-runall
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 30:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/selftrain_full outputs/feats outputs/skeleton_aug_base \
         outputs/skeleton_aug_clean outputs/skeleton_aug_part outputs/misa_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 前置检查（缺关键项直接退出；单项缺失仅警告）----
MISSING=0
for f in ig65m_r2plus1d34.pth \
         outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
         outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth \
         weights/mb_lite_latest_epoch.bin \
         bbox_train.json bbox_test.json bbox_thermal_train.json bbox_thermal_test.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; MISSING=1; }
done
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
if [ ! -d "$NTU_DIR" ] || [ "$(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' 2>/dev/null | wc -l)" -le 0 ]; then
  echo "❌ 缺 NTU 骨架（$NTU_DIR，骨架实验将跳过）"; NTU_OK=0
else
  NTU_OK=1
fi
[ "$MISSING" -eq 0 ] || { echo "❌ 关键文件缺失，终止"; exit 1; }
echo "==== 前置检查通过（NTU_OK=$NTU_OK）===="

# ============================================================
echo "########## [1/6] 伪标签全量自训练（最高 EV）##########"
# ============================================================
python scripts/pseudo_label_selftrain.py --stage pseudo \
  || echo "⚠️ [1] pseudo 打标签失败"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality main --full \
  --crop_cache bbox_train.json --weights ig65m_r2plus1d34.pth \
  --epochs 80 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain_full --workers 8 \
  || echo "⚠️ [1] main selftrain 失败"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality thermal --full \
  --crop_cache bbox_thermal_train.json --weights ig65m_r2plus1d34.pth \
  --epochs 80 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain_full --workers 8 \
  || echo "⚠️ [1] thermal selftrain 失败"
echo "✅ [1] 完成 → outputs/selftrain_full/{main,thermal}_selftrain_full.pth（判读：融合后 LB>0.73）"

# ============================================================
echo "########## [2/6] v3 bounded-prior（零训练）##########"
# ============================================================
python scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth \
  --quantize --flip_tta --prob_avg \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --prior_experiment --output outputs/feats/submission_main.csv \
  || echo "⚠️ [2] prior 失败"
echo "✅ [2] 完成（判读：soft-count imbalance ≥2 才烧 spare 提交测 submission_main_prior.csv）"

# ============================================================
echo "########## [3/6] YOLO 部位特征 + GBDT ##########"
# ============================================================
echo "[3] 安装 scikit-learn（GBDT 需要，cuhk_x 默认没有）..."
pip install -q scikit-learn || echo "⚠️ [3] sklearn 安装失败（GBDT 将跳过）"
python scripts/yolo_motion_feats.py --train_root ~/Multimodal/data/Training/HAR \
  --out outputs/feats/bbox_feats_train.csv --stride 2 \
  || echo "⚠️ [3] bbox 特征失败"
python scripts/yolo_pose_kpts.py --train_root ~/Multimodal/data/Training/HAR \
  --out outputs/feats/kpts_feats_train.csv --stride 2 \
  || echo "⚠️ [3] pose 特征失败"
python scripts/eval_feats_gbdt.py --csv outputs/feats/bbox_feats_train.csv \
  || echo "⚠️ [3] bbox GBDT 失败"
python scripts/eval_feats_gbdt.py --csv outputs/feats/kpts_feats_train.csv \
  || echo "⚠️ [3] pose GBDT 失败"
echo "✅ [3] 完成（判读：GBDT acc 显著 >2.5% 随机 → 部位信息有信号）"

# ============================================================
if [ "$NTU_OK" -eq 1 ]; then
# ============================================================
echo "########## [4/7] 骨架诊断 + StepLR 基线（旧 0.5447 是 Cosine 版，需重跑对照）##########"
# ============================================================
python scripts/diag_skeleton_quality.py --train_root ~/Multimodal/data/Training/HAR \
  || echo "⚠️ [4] 骨架诊断失败"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --save_dir outputs/skeleton_aug_base --workers 8 \
  || echo "⚠️ [4] 骨架基线失败"
echo "✅ [4] 完成（StepLR 基线，后续 clean/part 对照它）"

# ============================================================
echo "########## [5/7] 骨架清洗（对照 StepLR 基线）##########"
# ============================================================
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --clean --save_dir outputs/skeleton_aug_clean --workers 8 \
  || echo "⚠️ [5] 骨架清洗失败"
echo "✅ [5] 完成（判读：fold0 > StepLR 基线 → 清洗有效）"

# ============================================================
echo "########## [6/7] 骨架 part-aware 部位权重（对照 clean-only）##########"
# ============================================================
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --clean --part_aware --save_dir outputs/skeleton_aug_part --workers 8 \
  || echo "⚠️ [6] 骨架 part-aware 失败"
echo "✅ [6] 完成（判读：fold0 > clean-only → 部位加权有效）"
else
  echo "⚠️ [4][5][6] 跳过（无 NTU 骨架）"
fi

# ============================================================
echo "########## [7/7] MISA 双流多任务正则（低预期）##########"
# ============================================================
python scripts/train_misa_dual.py \
  --fold 0 --folds 3 --weights ig65m_r2plus1d34.pth \
  --epochs 50 --batch_size 16 --lr 1e-4 \
  --main_crop bbox_train.json --thermal_crop bbox_thermal_train.json \
  --save_dir outputs/misa_dual --workers 8 \
  || echo "⚠️ [7] MISA 失败"
echo "✅ [7] 完成（判读：fold0 >0.66 留 / <0.66 关）"

# ============================================================
echo "############################################################"
echo "ALL EXPERIMENTS DONE（job ${SLURM_JOB_ID}）"
echo "汇总判读："
echo "  [1] 伪标签: 融合后 LB>0.73 → 有效（烧 LB）"
echo "  [2] prior : imbalance≥2 → 烧 spare 提交"
echo "  [3] GBDT  : acc>2.5% 随机 → 部位信号"
echo "  [4] 骨架 StepLR 基线: 新对照（旧 0.5447 是 Cosine 版）"
echo "  [5] 骨架清洗: fold0>StepLR 基线 → 有效"
echo "  [6] part-aware: >clean-only → 有效"
echo "  [7] MISA  : fold0>0.66 → 留，否则关"
echo "详见 logs/cuhkx-runall-${SLURM_JOB_ID}.out"
echo "############################################################"
