#!/bin/bash
# ============================================================
# CUHK-X —— 模态工作流统一入口
# 用法（在服务器 ~/Multimodal 下）：
#   bash scripts/run_modality_pipeline.sh explore   # CPU：IMU/Radar 数据格式检测（登录节点直接跑）
#   bash scripts/run_modality_pipeline.sh skeleton  # GPU：骨架增强训练（sbatch 提交）
#   bash scripts/run_modality_pipeline.sh analyze   # GPU：互补分析（sbatch 提交）
# ============================================================

MODE="${1:-help}"

case "$MODE" in
  explore)
    echo "==== [explore] IMU/Radar 数据格式检测（CPU）===="
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cuhk_x
    python scripts/debug_modality.py --modality imu
    echo "---"
    python scripts/debug_modality.py --modality radar
    echo "==== explore 完成，把输出发给 agent 以写 IMU/Radar 加载器 ===="
    ;;
  skeleton)
    echo "==== [skeleton] 提交骨架增强训练（GPU, sbatch）===="
    sbatch scripts/train_skeleton_aug.sbatch
    ;;
  analyze)
    echo "==== [analyze] 提交互补分析（GPU, sbatch）===="
    sbatch scripts/analyze_complementarity.sbatch
    ;;
  *)
    echo "用法: bash scripts/run_modality_pipeline.sh {explore|skeleton|analyze}"
    echo "  explore  : CPU 数据格式检测（IMU/Radar）"
    echo "  skeleton : 骨架增强训练（3折, 增强已自动生效）"
    echo "  analyze  : main vs skeleton/thermal 互补性分析"
    ;;
esac
