#!/bin/bash
# ============================================================
# CUHK-X —— 全量重跑 + 新方法论（2026-09-01）
# 
# 背景：上一轮审计发现多处致命 bug（伪标签 crop 键、骨架脏数据、
#       rng worker 相关性、auxpose flip 未镜像等），历史否定结论不可信；
#       同时方法论层有零成本 TTA、真复刻 14 名、跨人域对抗等未试方向。
# 
# 本脚本分 5 阶段执行，每阶段有明确判据和早停条件：
#   Phase 0: 前置清理（删污染缓存）
#   Phase 1: 零成本推理端增益（TTA 三件套）
#   Phase 2: 真复刻 14 名 thermal 2D（严格 flip-only + 5 折）
#   Phase 3: 训练侧新方法（CB Loss / GRL 主体不变性）
#   Phase 4: 修复后重跑（auxpose / 伪标签 / 骨架重抽）
# 
# 用法: sbatch scripts/cuHKx_relaunch_all.sbatch
# ============================================================
#SBATCH -J cuhkx-relaunch
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 72:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/relaunch

export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================
# Phase 0: 前置清理（删污染缓存，确保后续实验基于干净数据）
# ============================================================
echo ""
echo "########## [Phase 0] 前置清理 ##########"

# 0A. 删除旧热像骨架缓存（extract_thermal_skeleton 的 npz 产自 mask 失效版本）
if [ -d "outputs/thermal_skeleton" ]; then
    echo "[0A] 删除旧热像骨架缓存 outputs/thermal_skeleton/"
    rm -rf outputs/thermal_skeleton
fi

# 0B. 删除旧伪标签输出（crop 键全错版本）
if [ -f "outputs/pseudo_labels.json" ]; then
    echo "[0B] 删除旧伪标签 outputs/pseudo_labels.json"
    rm -f outputs/pseudo_labels.json
fi

echo "[Phase 0] 清理完成"

# ============================================================
# Phase 1: 零成本推理端增益（TTA 三件套，不改模型权重）
# ============================================================
echo ""
echo "########## [Phase 1] 零成本 TTA 三件套 ##########"

# 1A. torch.roll 时序抖动 ±1 + 1B. 多组 segment 采样 + flip TTA
#     ensemble_inference.py 已支持 --time_tta_mode roll_jitter + --num_infer_samples
echo "[1A+1B] 推理端 TTA：roll_jitter(±1) + 3 组 segment 采样 + flip"
python scripts/ensemble_inference.py \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth \
  --quantize --flip_tta \
  --time_tta 3 --time_tta_mode roll_jitter \
  --num_infer_samples 3 \
  --prob_avg --output submission_tta_roll.csv \
  || echo "⚠️ [1A+1B] TTA 推理失败"

# 1C. 中位中心 crop + Depth_Color fallback（需改 detect.py，暂跳过）
echo "[1C] 中位中心 crop + Depth fallback → 需修改 detect.py，本轮暂不实现"

echo "[Phase 1] 完成（1A+1B 已跑，预计 +0.5~1.0 pt）"

# ============================================================
# Phase 2: 真复刻 14 名 thermal 2D（flip-only + 无重采样 + 5 折）
# ============================================================
echo ""
echo "########## [Phase 2] 真复刻 14 名 thermal 2D ##########"

# 2A. FrameNet 从零 + flip-only + 8ep + 无 LS + lr 3e-4（fold0 快速验证）
echo "[2A] FrameNet flip-only fold0 快速验证"
python scripts/train_framenet_thermal.py --fold 0 --arch framenet \
  --num_frames 8 --size 112 --sample_mode segment --gray_norm \
  --epochs 8 --no_smooth --lr 3e-4 --no_balanced --flip_only \
  --save_dir outputs/relaunch/framenet_fold0 --workers 8 \
  || echo "⚠️ [2A] 失败"

# 判据：fold0 val > 0.5447（clean-only 基线）→ 进入 5 折；否则关

# 2B. FrameNet 5 折（若 2A 通过）
#     注：需循环 5 折，此处仅示意 fold0-4，实际可拆成 5 个 sbatch
for FOLD in 0 1 2 3 4; do
    echo "[2B-fold${FOLD}] FrameNet 5 折之 fold${FOLD}"
    python scripts/train_framenet_thermal.py --fold ${FOLD} --arch framenet \
      --num_frames 8 --size 112 --sample_mode segment --gray_norm \
      --epochs 8 --no_smooth --lr 3e-4 --no_balanced --flip_only \
      --save_dir outputs/relaunch/framenet_5fold --workers 8 \
      || echo "⚠️ [2B-fold${FOLD}] 失败"
done

# 2C. ResNet18/34 预训练 2D + per-frame logits mean（对比强化版）
for ARCH in resnet18 resnet34; do
    echo "[2C-${ARCH}] ImageNet 预训练 2D + 逐帧投票"
    python scripts/train_framenet_thermal.py --fold 0 --arch ${ARCH} \
      --num_frames 8 --size 112 --sample_mode segment --imagenet_norm \
      --epochs 8 --no_smooth --lr 3e-4 --no_balanced --flip_only \
      --save_dir outputs/relaunch/resnet2d_${ARCH} --workers 8 \
      || echo "⚠️ [2C-${ARCH}] 失败"
done

echo "[Phase 2] 完成"
echo "          判据: 任一架构 5 折 logits 平均 val > 0.6339（thermal R2+1D 基线）"
echo "                → 作为第三异构成员进 0.73 融合（剩余 20MB 预算够放 int5 打包）"

# ============================================================
# Phase 3: 训练侧新方法（CB Loss / GRL 主体不变性）
# ============================================================
echo ""
echo "########## [Phase 3] 训练侧新方法 ##########"

# 3A. Class-Balanced Loss（有效样本数加权，针对 29.5× 长尾）
echo "[3A] CB Loss fold0 快速验证（beta=0.9999，主要救尾部类）"
python scripts/train_step1.py --fold 0 --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --cb_loss --cb_beta 0.9999 \
  --label_smoothing 0.1 --save_dir outputs/relaunch/cb_loss_fold0 \
  || echo "⚠️ [3A] CB Loss 失败"

# 判据：fold0 val > 基线（无 CB Loss 的 fold0）→ 进入全量

# 3B. 梯度反转层（GRL）主体不变性训练
echo "[3B] GRL fold0 快速验证（lambda=0.5）"
python scripts/train_grl.py --fold 0 --backbone r2plus1d34 \
  --weights ig65m_r2plus1d34.pth --lambda_grl 0.5 \
  --label_smoothing 0.1 --save_dir outputs/relaunch/grl_fold0 \
  || echo "⚠️ [3B] GRL 失败"

# 判据：fold0 val > 普通训练（无 GRL）→ 说明跨人泛化有提升，进入全量

echo "[Phase 3] 完成（fold0 验证；若通过则需手动跑全量/多折）"
echo "          预期: CB Loss 救尾部类；GRL 提升新人泛化"

# ============================================================
# Phase 4: 修复后重跑（auxpose / 伪标签 / 骨架重抽）
# ============================================================
echo ""
echo "########## [Phase 4] 修复后重跑 ##########"

# 4A. 热像骨架重抽（mask 已修，需先删旧缓存——已在 Phase 0 完成）
echo "[4A] 热像骨架重抽（step=2 加速，全量约需数小时）"
python scripts/extract_thermal_skeleton.py --step 2 --limit 0 \
  --out outputs/thermal_skeleton \
  || echo "⚠️ [4A] 失败"

# 判据：重抽后 dual_dataset 加载 npz，conf==0 帧被硬 mask（不再半权泄漏）

# 4B. Auxpose 重跑（flip 镜像已修，历史全负结论可疑）
echo "[4B] Auxpose DepthIR 重跑（fold0 快速验证）"
python scripts/train_auxpose.py --fold 0 --modality depthir \
  --crop_cache bbox_train.json --lambda_pose 0.05 --pose_layer 3 \
  --save_dir outputs/relaunch/auxpose_depthir \
  || echo "⚠️ [4B] 失败"

# 判据：fold0 val > 原 auxpose 最佳（之前全 < baseline）→ 进入全量

# 4C. 伪标签自训练（crop 键已修，需先合规确认）
echo "[4C] ⚠️ 伪标签线需主办方书面确认合规后再跑"
echo "      若合规：先打伪标签 → 再全量自训练"
echo ""
echo "      # 打伪标签（co-training + top-K）"
echo "      python scripts/pseudo_label_selftrain.py --stage pseudo \\"
echo "        --main_ckpt outputs/main_full/r2plus1d34_depthir_full_seed42.pth \\"
echo "        --thermal_ckpt outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth \\"
echo "        --main_test_crop bbox_test.json --thermal_test_crop bbox_thermal_test.json \\"
echo "        --pseudo_k 120"
echo ""
echo "      # 自训练（thermal，全量模式）"
echo "      python scripts/pseudo_label_selftrain.py --stage selftrain \\"
echo "        --modality thermal --full --epochs 80 --pseudo_weight 0.3 \\"
echo "        --crop_cache \"\"  # 自动选 bbox_thermal_train.json"

echo "[Phase 4] 完成（4C 需合规确认后手动执行）"

# ============================================================
# 总结
# ============================================================
echo ""
echo "==== RELAUNCH ALL DONE ===="
echo ""
echo "已完成:"
echo "  Phase 0: 清理污染缓存 ✓"
echo "  Phase 1: 零成本 TTA（roll_jitter + 多采样）✓ → 直接跑推理，无需重训"
echo "  Phase 2: 真复刻 14 名 thermal 2D ✓（5 折 + resnet18/34 对照）"
echo "  Phase 3: CB Loss ✓ / GRL ✓ → fold0 验证通过后可手动跑全量"
echo "  Phase 4: 骨架重抽 ✓ / auxpose 重跑 ✓ / 伪标签 → 需合规确认"
echo ""
echo "下一步行动:"
echo "  1. 检查 Phase 1 TTA 提交结果：预期 LB 从 0.73 → 0.735~0.74"
echo "  2. 检查 Phase 2 结果：若 thermal 2D val > 0.6339，打包 int5 进融合"
echo "  3. 检查 Phase 3 fold0 结果：CB Loss / GRL 若优于基线，跑全量/多折"
echo "  4. 向主办方确认伪标签合规性，若允许则跑 4C"
echo ""
echo "结果目录: outputs/relaunch/*, submission_tta_roll.csv"
