#!/bin/bash
# ============================================================
# CUHK-X —— 伪标签全量自训练（决定性实验，2026-08-26）
# 背景：fold0 验证已做（job 44054：thermal 0.6275 / main 0.6566，微降）
#       但 fold0 val 是训练分布，测不出伪标签价值（伪标签针对测试分布）
#       唯一没做的 = 全量自训练 + 烧 LB（决定性实验）
# 流程：
#   [1] pseudo     : 全量 main+thermal(seed42) 对 405 测试打伪标签（top-120 置信度 0.907）
#   [2] selftrain main --full    : 全量 2931 真标签 + 伪标签 重训（epochs 80）
#   [3] selftrain thermal --full
# 产出: outputs/selftrain_full/{main,thermal}_selftrain_full.pth
# 之后: 用 ensemble_inference.py 与原始 0.73 模型决策级融合 → 烧 LB
# 判读：融合后 LB > 0.73 → 伪标签有效（同分布 405 数据是唯一合法大杠杆）
# 用法: sbatch scripts/pseudo_label_selftrain_full.sbatch
# ============================================================
#SBATCH -J cuhkx-pseudo-full
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
mkdir -p logs outputs/selftrain_full
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for f in outputs/main_full/r2plus1d34_depthir_full_seed42.pth \
         outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth \
         bbox_test.json bbox_thermal_test.json bbox_train.json bbox_thermal_train.json; do
  [ -f "$f" ] || { echo "❌ 缺 $f"; exit 1; }
done

# ============ [1/3] 打伪标签 ============
echo "######## [1/3] 全量模型打伪标签（Co-training + top-K）########"
python scripts/pseudo_label_selftrain.py --stage pseudo
echo "######## [1/3] 伪标签完成 ########"

# ============ [2/3] selftrain main 全量 ============
echo "######## [2/3] 自训练 main 全量（2931 真标签 + 伪标签，epochs 80）########"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality main --full \
  --crop_cache bbox_train.json --weights ig65m_r2plus1d34.pth \
  --epochs 80 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain_full --workers 8
echo "######## [2/3] main selftrain 全量完成 ########"

# ============ [3/3] selftrain thermal 全量 ============
echo "######## [3/3] 自训练 thermal 全量（epochs 80）########"
python scripts/pseudo_label_selftrain.py --stage selftrain \
  --modality thermal --full \
  --crop_cache bbox_thermal_train.json --weights ig65m_r2plus1d34.pth \
  --epochs 80 --pseudo_weight 0.5 \
  --save_dir outputs/selftrain_full --workers 8
echo "######## [3/3] thermal selftrain 全量完成 ########"

echo "==== PSEUDO FULL DONE ===="
echo "产物: outputs/selftrain_full/{main,thermal}_selftrain_full.pth"
echo "下一步: 用 ensemble_inference.py 与原始 0.73 模型决策级融合 → 烧 LB"
echo "判读: 融合后 LB > 0.73 → 伪标签有效（同分布 405 数据 = 合法大杠杆）"
