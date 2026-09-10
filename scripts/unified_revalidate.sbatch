#!/bin/bash
# ============================================================
# CUHK-X —— 修复后需复验的统一编排（2026-09-01）
# 背景：上一轮修了一批代码缺陷 → 以下几项旧结论已被污染，修复后需重跑复核：
#   [1] 14名从未真正复刻（train_framenet 现支持 --no_balanced/--aug_strength）
#   [2] 热像 2D 骨架脏数据（extract_thermal_skeleton F2 修复后重抽 + 统计）
#   [3] 伪标签 crop 键 bug（F1 修复后：3 轮 fold 迭代验证多轮自训练是否真增益/饱和）
#   [4] auxpose flip 未镜像骨架（修复后 fold0 复验；该线已判负，可跳）
# 判读：①FrameNet 严格复刻 >0.21 旧先验 → 冤杀翻案；②conf/帧统计干净 → 2D 骨架线可重开；
#       ③伪标签 round1/2 val 较 round0 递增 → 多轮有效；④auxpose > 历史负值 → 翻案
# 用法: sbatch scripts/unified_revalidate.sbatch
# ============================================================
#SBATCH -J cuhkx-revalidate
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/revalidate outputs/pseudo_iter outputs/thermal_2d_norm outputs/thermal2d_kpts
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TCROP=bbox_thermal_train.json
[ -f "$TCROP" ] || { echo "⚠️ 缺 $TCROP"; exit 1; }
EP40=40

echo "########## [1/4] 14名严格复刻（resnet18整帧 / FrameNet严格8ep / FrameNet长训40ep）##########"
python scripts/train_framenet_thermal.py --fold 0 --arch resnet18 \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs $EP40 --save_dir outputs/revalidate --workers 8 \
  || echo "⚠️ [1a] resnet18 整帧复刻 失败"
python scripts/train_framenet_thermal.py --fold 0 --arch framenet \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs 8 --no_smooth --lr 3e-4 --no_balanced --aug_strength 0 \
  --save_dir outputs/revalidate --workers 8 \
  || echo "⚠️ [1b] FrameNet 严格复刻(8ep无增) 失败"
python scripts/train_framenet_thermal.py --fold 0 --arch framenet \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs $EP40 --save_dir outputs/revalidate --workers 8 \
  || echo "⚠️ [1c] FrameNet 长训 失败"
echo "[1] 判读: [1b] 严格复刻 > 旧 0.21 先验 → FrameNet 冤杀翻案"

echo "########## [2/4] 热像 2D 骨架重抽（F2 修复：无检测占位+失效全零+硬 mask）##########"
python scripts/extract_thermal_skeleton.py --root ~/Multimodal/data/Training/HAR \
  --out outputs/thermal2d_kpts --step 1 \
  || echo "⚠️ [2] 骨架重抽 失败"
echo "[2] 判读: conf 统计/帧数统计干净（无丢帧错位）→ 2D 骨架线可重开（重训 dual 再议）"

echo "########## [3/4] 伪标签 3 轮 fold0 迭代（F1 crop 键修复后验证多轮）##########"
# teacher 必须是「纯 main R2+1D」ckpt（dual SM 含 motion/a 键与 build_model 不匹配会崩）
MAIN_T="${MAIN_TEACHER:-outputs/main_dual/main_S_fold0.pth}"
TH_T="${TH_TEACHER:-outputs/thermal_notebook14/t0/r2plus1d34_thermal_fold0.pth}"
for R in 0 1 2; do
  PJ="outputs/pseudo_iter/pseudo_r${R}.json"
  echo "----- [3] round$R 打伪标签 (teacher=${MAIN_T}) -----"
  python scripts/pseudo_label_selftrain.py --stage pseudo \
    --main_ckpt "$MAIN_T" --thermal_ckpt "$TH_T" --pseudo_json "$PJ" \
    || echo "⚠️ [3] round$R 伪标签失败"
  echo "----- [3] round$R main selftrain fold0 -----"
  python scripts/pseudo_label_selftrain.py --stage selftrain --modality main --fold 0 \
    --pseudo_json "$PJ" --save_dir "outputs/pseudo_iter/r${R}" \
    --num_frames 16 --size 128 --epochs 60 --lr 1e-4 --batch_size 32 \
    || echo "⚠️ [3] round$R selftrain 失败"
  NEW="outputs/pseudo_iter/r${R}/main_selftrain_fold0.pth"
  if [ -f "$NEW" ]; then MAIN_T="$NEW"; fi
done
echo "[3] 判读: 每轮 selftrain fold0 val（round0 基线→round1/2 递增=多轮有效）"

echo "########## [4/4] auxpose fold0 复验（flip 镜像修复；该线已判负，可选跑）##########"
if [ "${RUN_AUXPOSE:-0}" = "1" ]; then
  python scripts/train_auxpose.py --modality main --fold 0 --epochs 60 \
    --crop_cache bbox_train.json --save_dir outputs/revalidate --workers 8 \
    || echo "⚠️ [4] auxpose 复验失败"
else
  echo "⚠️ [4] 跳过（RUN_AUXPOSE=1 才跑；auxpose 已判负，仅修复后自愿复核）"
fi

echo "==== REVALIDATE DONE ===="
echo "判定汇总: ①FrameNet严格复刻 ②2D骨架干净重抽 ③伪标签3轮翻转 ④auxpose(可选)"
