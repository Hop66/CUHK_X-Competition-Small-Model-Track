#!/bin/bash
# ============================================================
# CUHK-X —— 骨架双专家对照（BHaRNet-B 最小复刻）
# 依据 diagnose_skeleton_parts.py（job 46474）：手精细(25)组 val_acc 0.451 << 下肢 0.753，
#   但手组大量 ≥0.4 可分信号（Wash_face 0.714 / Tap_keyboard 0.703 / Eat_food 0.625 / Selfie 0.700…）
# → 分离"上肢专家"(胸脊+头颈+双臂 9 关节, 手/上肢 28 类) + "躯干下肢专家"(8 关节, 12 类)，
#   损失 = CE(u)+CE(l)+λcpl·CE(和)+λnor·CE(NoisyOR)（BHaRNet III-D），推理 = logit 和。
# 对照：skeleton_aug（无clean 无dual）fold0 = 0.5447（同协议 Cosine）
#   判读：fold0 > 0.5447 且 手精细组(若复跑诊断)抬升 → 双专家有效 → 全量+融合
#         fold0 ≤ 0.5447 → 骨架线收官（信息上限+融合拖累），转 Noisy-OR 融合对拍
# 组合：默认（λcpl=0.5, λnor=0.1）；可用 --lam_nor 0 关 Noisy-OR 做 A/B
# 用法: sbatch scripts/skeleton_dual_expert.sbatch
# ============================================================
#SBATCH -J cuhkx-skel-dual
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
mkdir -p logs outputs/skeleton_dual outputs/skeleton_dual_nonor
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

echo "########## [1/2] 双专家（λcpl=0.5, λnor=0.1）fold0 ##########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --dual_expert \
  --save_dir outputs/skeleton_dual --workers 8 \
  || echo "⚠️ [1] 双专家失败"
echo "✅ [1] 完成（判读：fold0 > 0.5447 → 双专家有效）"

echo "########## [2/2] 双专家 A/B：λnor=0（关 Noisy-OR，分离损失影响）##########"
python scripts/train_skeleton_augmented.py \
  --fold 0 --folds 3 --ntu_root "$NTU_DIR" \
  --epochs 60 --batch_size 64 \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --sched cosine --dual_expert --lam_nor 0 \
  --save_dir outputs/skeleton_dual_nonor --workers 8 \
  || echo "⚠️ [2] 双专家(nonor)失败"
echo "✅ [2] 完成（判读：相比 [1] → Noisy-OR 增量）"

echo "==== SKELETON DUAL EXPERT DONE ===="
