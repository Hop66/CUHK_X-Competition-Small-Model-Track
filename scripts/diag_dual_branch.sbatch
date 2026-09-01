#!/bin/bash
# ============================================================
# CUHK-X —— 诊断：双分支 2D 分支不学的根因
# 现象（job 42191）：双分支 fused 下 2D 分支 acc 恒定 ~0.1（接近随机），3D 分支 0.4+ 正常。
# 需区分：①热像 2D 骨架质量差（数据） vs ②共享 backbone+fused loss 下 3D 主导压制 2D（架构）。
#
# 三连诊断（一次 job）：
#   [1] stats        —— 扫描热像 npz 输出骨架质量（conf/有效关节/坐标范围）
#   [2] 2D-only 训练 —— 只喂热像 2D 单分支（隔离数据因素）
#       acc~0.1-0.2 → 根因①（热像骨架质量差）；acc 0.3+ → 根因②（共享压制）
#   [3] dual loss 双分支 —— 两分支独立 CE（验证 ②：独立梯度能否激活 2D 分支）
#
# 前置: outputs/thermal_skeleton/*.npz（Step 2）+ weights/mb_lite_latest_epoch.bin
# 用法: sbatch scripts/diag_dual_branch.sbatch
# ============================================================
#SBATCH -J cuhkx-diagdual
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
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -f weights/mb_lite_latest_epoch.bin ] || { echo "❌ 缺 weights/mb_lite_latest_epoch.bin"; exit 1; }

# ============ [1/3] 热像骨架质量统计 ============
echo "######## [1/3] 热像 2D 骨架质量统计 ########"
python scripts/train_thermal_2d.py --mode stats
echo "######## [1/3] stats 完成 ########"

# ============ [2/3] 2D-only 单分支（隔离数据因素）============
echo "######## [2/3] 热像 2D 单分支（对照双分支的 2D acc）########"
python scripts/train_thermal_2d.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --epochs 30 --folds 3 --balanced --conf_norm sigmoid \
  --save_dir outputs/thermal_2d --workers 8
echo "######## [2/3] 2D-only 完成 ########"

# ============ [3/3] dual loss 双分支（独立梯度）============
echo "######## [3/3] 双分支 dual loss（两分支独立 CE，验证共享压制）########"
python scripts/train_dual_branch.py \
  --pretrained weights/mb_lite_latest_epoch.bin \
  --epochs 30 --folds 3 --balanced \
  --conf_norm sigmoid --rot_angle 30.0 \
  --loss_mode dual --fusion learn \
  --save_dir outputs/dual_branch_dual --workers 8
echo "######## [3/3] dual loss 完成 ########"

echo "==== 诊断完成：stats → 2D-only → dual loss ===="
