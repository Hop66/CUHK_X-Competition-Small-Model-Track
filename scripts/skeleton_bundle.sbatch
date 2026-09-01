#!/bin/bash
# ============================================================
# CUHK-X —— 骨架线收官编排（一次提交，顺序跑完 4 步）
#   [1/4] per-class: main vs skeleton/thermal 互补 + 骨架胜类判据（选择性 teacher 是否成立）
#   [2/4] 骨架双专家（BHaRNet-B 复刻）fold0（λcpl=0.5, λnor=0.1）
#   [3/4] 骨架双专家 A/B：λnor=0（隔离 Noisy-OR 贡献）
#   [4/4] 融合对拍：fold0 val 上 prob_avg vs Noisy-OR（决策级重信度融合）
# 判读汇总：
#   [1] 骨架胜类(diff≥+0.10, n≥10)≥3 个 → 选择性蒸馏有空间（再开 train_main_baseline --aux_teach）
#   [2] fold0 > 0.5447 → 双专家有效 → 全量+融合；否则骨架训练侧收官
#   [4] Noisy-OR acc - prob_avg ≥ +0.01 → 换融合重出 0.73 测试 submit；否则不动
# 前置：weights/mb_lite_latest_epoch.bin + NTU 骨架 + main/thermal fold0 ckpt + bbox json
# 用法: sbatch scripts/skeleton_bundle.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-bundle
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/feats outputs/skeleton_dual outputs/skeleton_dual_nonor
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 MotionBERT 权重"; exit 1; }
NTU_DIR="${NTU_ROOT:-data/external/ntu/ntu60}"
if [ ! -d "$NTU_DIR" ] || [ "$(find "$NTU_DIR" -maxdepth 1 -name '*.skeleton' 2>/dev/null | wc -l)" -le 0 ]; then
  echo "❌ 缺 NTU 骨架（$NTU_DIR）"; exit 1
fi

# 骨架 checkpoint（增强后的优先）
SKEL_CKPT=outputs/skeleton_aug/motionbert_fold0.pth
[ -f "$SKEL_CKPT" ] || SKEL_CKPT=outputs/skeleton_aug/augmented_fold0.pth
[ -f "$SKEL_CKPT" ] || SKEL_CKPT=outputs/skeleton_mb/motionbert_fold0.pth
echo "使用 skeleton checkpoint: $SKEL_CKPT"

# main / thermal fold0（融合对拍用；不存在则跳过 [4]）
MAIN_CKPT=${MAIN_CKPT:-outputs/main_4ch_2fold/r2plus1d34_depthir_fold0.pth}
TH_CKPT=${TH_CKPT:-outputs/thermal_34/r2plus1d34_thermal_fold0.pth}

echo "########## [1/4] per-class 互补分析（骨架胜类 = 选择性 teacher 判据）##########"
python scripts/analyze_complementarity.py --fold 0 --per_class \
  --main_ckpt "$MAIN_CKPT" --skeleton_ckpt "$SKEL_CKPT" --thermal_ckpt "$TH_CKPT" \
  || echo "⚠️ [1] per-class 互补失败"
echo "✅ [1] 完成（判读：骨架胜类 ≥3 且 n≥10 → 选择性蒸馏可开）"

echo "########## [2/4] 骨架双专家 fold0（对照 0.5447）##########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --dual_expert \
  --save_dir outputs/skeleton_dual --workers 8 \
  || echo "⚠️ [2] 双专家失败"
echo "✅ [2] 完成（判读：fold0 > 0.5447 → 双专家有效）"

echo "########## [3/4] 双专家 A/B：λnor=0（隔离 Noisy-OR）##########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --dual_expert --lam_nor 0 \
  --save_dir outputs/skeleton_dual_nonor --workers 8 \
  || echo "⚠️ [3] 双专家(nonor)失败"
echo "✅ [3] 完成（判读：步3 vs 步2 → Noisy-OR 增量）"

echo "########## [4/4] 融合对拍：prob_avg vs Noisy-OR（fold0 val）##########"
if [ -f "$MAIN_CKPT" ] && [ -f "$TH_CKPT" ]; then
  python scripts/eval_fusion_noisyor.py --fold 0 \
    --main_ckpt "$MAIN_CKPT" --thermal_ckpt "$TH_CKPT" \
    || echo "⚠️ [4] 融合对拍失败"
else
  echo "⚠️ [4] 跳过（缺 main/thermal fold0 ckpt: $MAIN_CKPT / $TH_CKPT）"
fi
echo "✅ [4] 完成（判读：Noisy-OR - prob ≥ +1pt → 换融合）"

echo "==== SKELETON BUNDLE DONE（汇总：双专家 fold0 / 骨架胜类 / Noisy-OR 增量）===="
