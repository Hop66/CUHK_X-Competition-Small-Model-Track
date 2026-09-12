#!/bin/bash
# 09-12 重跑: conditional residual fusion 正确配置 (fix freeze过头 bug)
# 前置教训: freeze=1.0 只留2.3%可训 → 0.2372(坏配置非方法失败)
# 这轮: freeze=0.8(保留微调空间) residual + freeze=0(全量微调) residual, fold0
# 对照: P1 baseline fold0 main+th 融合 ~0.65-0.80 (dev协议)
#SBATCH -J cuhkx-p0cond2
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
echo "== P0 condres 重跑 fold0 (freeze修复) =="
echo "== [A] residual freeze=0.8 (保留微调) =="
python scripts/train_cond_residual.py --fold 0 --epochs 60 --freeze_enc 0.8 --mode residual --lambda_main 0.3 2>&1 | tail -5
echo "== [B] residual freeze=0 (全量微调) =="
python scripts/train_cond_residual.py --fold 0 --epochs 45 --freeze_enc 0 --mode residual --lambda_main 0.3 2>&1 | tail -5
echo DONE
