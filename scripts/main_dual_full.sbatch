#!/bin/bash
# ============================================================
# CUHK-X —— main 骨架双流 全量重训（证据：unified 51671 三折全正 mean +2.9pt，fold2 0.7058）
# 模式：main R2+1D-34(IG65M 4ch 固定框) ⊕ 骨架运动 MotionNet，3 seed 顺序全量（快照集成）
# 产出：outputs/main_dual_full/main_SM_full_seed{42,2024,777}.pth（末快照，每 10ep 覆写）
# 后用：quantize_pack int5 每 seed + 与现有 thermal_s42 融合 / main 二择 → 0.73 升级
# 前置：ig65m + bbox_train + outputs/motion_cache.pkl（覆盖 100%）
# 用法: sbatch scripts/main_dual_full.sbatch
# ============================================================
#SBATCH -J cuhkx-main-dual-full
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_dual_full
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }
CROP=bbox_train.json
[ -f "$CROP" ] || { echo "❌ 缺 $CROP"; exit 1; }
MC=outputs/motion_cache.pkl
[ -f "$MC" ] || { echo "⚠️ 缺 $MC"; python scripts/extract_motion_cache.py || exit 1; }

# ============================================================
# Phase 0: 【先行】融合提交 —— 只要有 dual ckpt 就先出可提交 CSV，再跑其它
#   1) dual SM 主模态 int5 打包（quantize_pack 通用 state_dict 量化，SM 可 pack）
#   2) ensemble_inference --main_dual（static⊕MotionNet，测试骨架提取 motion）
#      + thermal_s42 int5 --prob_avg + flip TTA → submission_dual0.csv
#   覆盖: --export=MAIN_DUAL_CKPT=xxx（换 teacher） / TH_PACK_PATH=xxx
# ============================================================
MAIN_DUAL="${MAIN_DUAL_CKPT:-outputs/main_dual_full/main_SM_full_seed42.pth}"
TH_PACK="${TH_PACK_PATH:-outputs/pack/thermal_s42_fold0_int5.pth}"
DUAL5="${DUAL5_PATH:-outputs/pack/main_dual_s42_fold0_int5.pth}"
if [ -f "$MAIN_DUAL" ]; then
  if [ ! -f "$DUAL5" ]; then
    echo "[提交#0a] dual SM int5 打包（$MAIN_DUAL → $DUAL5）..."
    python scripts/quantize_pack.py --name main_dual_s42 --checkpoints "$MAIN_DUAL" \
      --bits 5 --out_dir outputs/pack || echo "⚠️ dual int5 pack 失败（后续用 fp32 兜底）"
  else
    echo "[提交#0a] $DUAL5 已存在，复用"
  fi
  DUAL_ARG="--main_dual $MAIN_DUAL"
  [ -f "$DUAL5" ] && DUAL_ARG="--main_dual $DUAL5 --quantize"
  if [ -f "$TH_PACK" ]; then
    echo "[提交#0b] 先行融合提交：dual SM + thermal_s42 int5（prob_avg + flip）→ submission_dual0.csv"
    python scripts/ensemble_inference.py --prob_avg --flip_tta $DUAL_ARG --test_speed_scale 0.5 --w_main_dual 1.0 \
      --thermal "$TH_PACK" --w_thermal 1.0 \
      --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
      --output outputs/submission_dual0.csv || echo "⚠️ 先行融合 CSV 失败"
  else
    echo "[提交#0c] 缺 $TH_PACK → 先出 dual-only 后备 CSV（thermal int5 待 0.73 链先 pack）"
    python scripts/ensemble_inference.py --prob_avg --flip_tta $DUAL_ARG --test_speed_scale 0.5 --w_main_dual 1.0 \
      --main_crop bbox_test.json \
      --output outputs/submission_dual0.csv || echo "⚠️ dual-only CSV 失败"
  fi
else
  echo "[提交#0] 缺 $MAIN_DUAL → 先行提交跳过（先训练，Phase 2 再出 CSV）"
fi

EP=80
SEEDS="${SEEDS:-42}"   # 默认先一个 seed（快速 LB 验证）；要用全部: --export=SEEDS='42 2024 777'
for SEED in $SEEDS; do
  OUT=outputs/main_dual_full/main_SM_full_seed${SEED}.pth
  if [ -f "$OUT" ]; then echo "✅ seed$SEED 已存在（$OUT）→ 跳过训练"; continue; fi
  echo "########## FULL seed=$SEED · main S+骨架M ##########"
  python scripts/train_main_dual.py --full --seed "$SEED" \
    --num_frames 16 --size 128 --epochs $EP --lr 1e-4 --batch_size 32 \
    --weights ig65m_r2plus1d34.pth --crop_cache "$CROP" --motion_cache "$MC" \
    --full_save_every 10 --save_dir outputs/main_dual_full --workers 8 \
    || echo "⚠️ seed$SEED full 失败"
done

# ============================================================
# Phase 2: 【其它】升级提交 —— 用最新 seed42 ckpt 再叠 TTA(roll±1)+prior（
#   主 CSV + *_prior.csv 侧文件；unified[9] 已证 TTA4 +1.3 / prior +0.4）
# ============================================================
if [ -f "$MAIN_DUAL" ] && [ -f "$TH_PACK" ]; then
  if [ ! -f "$DUAL5" ]; then
    python scripts/quantize_pack.py --name main_dual_s42 --checkpoints "$MAIN_DUAL" \
      --bits 5 --out_dir outputs/pack || echo "⚠️ Phase2 dual int5 pack 失败"
  fi
  if [ -f "$DUAL5" ]; then
    echo "[提交#2] 升级融合：dual SM + thermal int5 + TTA(roll±1×3 采样) + prior → submission_dual_tta.csv(_prior)"
    python scripts/ensemble_inference.py --prob_avg --flip_tta \
      --time_tta 3 --time_tta_mode roll_jitter --num_infer_samples 3 \
      --main_dual "$DUAL5" --quantize --w_main_dual 1.0 \
      --thermal "$TH_PACK" --w_thermal 1.0 \
      --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
      --output outputs/submission_dual_tta.csv --prior_experiment \
      || echo "⚠️ 升级 CSV 失败"
  else
    echo "[提交#2] $DUAL5 打包失败（缺 ckpt 或 pack 出错）→ 升级 CSV 跳过"
  fi
else
  echo "[提交#2] 缺 $MAIN_DUAL 或 $TH_PACK → 升级 CSV 跳过"
fi

echo "==== MAIN DUAL FULL DONE ===="
echo "已提交候选:"
echo "  outputs/submission_dual0.csv      （先行，prob_avg+flip）"
echo "  outputs/submission_dual_tta.csv   （TTA+prior side file，若有）"
echo "产出: outputs/main_dual_full/main_SM_full_seed{42,2024,777}.pth"
echo "⚪ 判读: submission_dual0.csv 替换 main_s42 进 0.73 链，LB > 0.73 则采纳（+2.9pt 骨架增益）"
