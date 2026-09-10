#!/bin/bash
# ============================================================
# CUHK-X Iter3: BN 测试时自适应 (transductive BN adaptation)
# 路径A: fold 代理(离线信号)  main fold0/1/2 : no-adapt vs adapt 1/2pass
# 路径B: 真测试集出 CSV      main+thermal BN 适应后推理
# 判读: fold 代理 Δ>0 → 值得在测试集用; 出 CSV 待 LB
# 用法: sbatch scripts/tta_bn.sbatch
# ============================================================
#SBATCH -J cuhkx-bn-tta
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter3 BN-TTA ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "===== [A] fold 代理(offline 判定) main ====="
for F in 0 1 2; do
  echo "----- fold$F main -----"
  python scripts/tta_bn.py --modality main --mode fold --fold "$F" --flip --adapt_passes 2 || echo "⚠️ fold$F"
done

echo "===== [B] 真测试集 BN 适应 出 CSV ====="
python scripts/tta_bn.py --modality main --mode test --flip --adapt_passes 2 \
  --output outputs/sub_E_main_bn.csv || echo "⚠️ test main"
python scripts/tta_bn.py --modality thermal --mode test --flip --adapt_passes 2 \
  --output outputs/sub_E_thermal_bn.csv || echo "⚠️ test thermal"

echo ""
echo "==== Iter3 DONE: 待 LB ===="
ls -la outputs/sub_E_*.csv 2>/dev/null
