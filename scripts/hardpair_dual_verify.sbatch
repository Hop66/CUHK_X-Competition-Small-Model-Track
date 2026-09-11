#!/bin/bash
# 09-11: 重跑 fold0 双塔难对 vs 基线验证（59788 验证脚本 KeyError 修复后）
#SBATCH -J cuhkx-hpdualver
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
echo "==== 双塔难对 fold0 融合验证(修复交集域) ===="
python scripts/hardpair_dual_fold.py 2>&1 | tail -10
echo "==== VERIFY DONE ===="
