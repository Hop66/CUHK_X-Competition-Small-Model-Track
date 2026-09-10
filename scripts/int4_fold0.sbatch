#!/bin/bash
#SBATCH -J cuhkx-int4eq
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/pack outputs/oof
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== 1. 打包 int4 全部成员 ===="
for spec in "main_s42:outputs/main_full/r2plus1d34_depthir_full_seed42.pth" \
            "main_s777:outputs/main_full/r2plus1d34_depthir_full_seed777.pth" \
            "main_avg:outputs/main_avg/r2plus1d34_depthir_avg.pth" \
            "thermal_s42:outputs/thermal_full/r2plus1d34_thermal_full_seed42.pth" \
            "thermal_s777:outputs/thermal_full/r2plus1d34_thermal_full_seed777.pth"; do
  nm="${spec%%:*}"; ck="${spec#*:}"
  [ -f "$ck" ] || { echo "⚠缺 $ck"; continue; }
  python scripts/quantize_pack.py --name "$nm" --checkpoints "$ck" --bits 4 \
    --out_dir outputs/pack 2>&1 | tail -1
done
echo "==== 2. fold0 int5 vs int4 等价验证 ===="
python scripts/eval_pack_fold0.py --workers 3
echo "== DONE =="
