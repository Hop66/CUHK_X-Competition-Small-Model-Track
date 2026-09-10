#!/bin/bash
# ============================================================
# CUHK-X Iter10: 蒸馏后最终装配（等待 predistill/cb 落地后再跑）
#   [0] 蒸馏 student 打包 int5（main_distill / thermal_distill）
#   [1] 融合出三档 TTA 提交（沿用最稳形状: Mega-风格多crop+时窗）
# 前置产物: outputs/main_distill/main_distill_full_seed42.pth
#           outputs/thermal_distill/thermal_distill_full_seed42.pth
# 用法: sbatch scripts/assemble_distill.sbatch
# ============================================================
#SBATCH -J cuhkx-assemble
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/pack
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter10 Assemble ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MD=outputs/main_distill/main_distill_full_seed42.pth
TD=outputs/thermal_distill/thermal_distill_full_seed42.pth
[ -f "$MD" ] && [ -f "$TD" ] || { echo "❌ 缺蒸馏产物"; exit 1; }

echo "===== [0] 打包蒸馏 int5 ====="
python scripts/quantize_pack.py --name main_ds --checkpoints "$MD" --bits 5 --out_dir outputs/pack
python scripts/quantize_pack.py --name thermal_ds --checkpoints "$TD" --bits 5 --out_dir outputs/pack

echo "===== [1] 融合输出（flip-only, 遵 LB 教训: TTA4/多裁剪全负）====="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_ds_int5.pth outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/thermal_ds_int5.pth outputs/pack/thermal_s42_fold0_int5.pth \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_H_ds_flip.csv || echo "⚠️ H1"

# 参考: 若 LB 显示蒸馏 student 单流有进步, 可在此加纯集成(不重TTA)变体
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_ds_int5.pth \
  --thermal outputs/pack/thermal_ds_int5.pth \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_H_dsonly_flip.csv || echo "⚠️ H2"

echo "==== Iter10 Assemble DONE ===="
ls -la outputs/sub_H_*.csv 2>/dev/null
