#!/bin/bash
#SBATCH -J cuhkx-thchain
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
CK=outputs/th_nf32_full/r2plus1d34_thermal_full_seed42.pth
[ -f "$CK" ] || { echo "缺 $CK"; exit 1; }
echo "== 打包 th_nf32full int5 =="
python scripts/quantize_pack.py --name th_nf32full --checkpoints "$CK" --bits 5 --out_dir outputs/pack 2>&1 | tail -1
echo "== test 推理: main_s42(16f) + th_nf32full(32f, --thermal_frames 32) flip =="
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_chain_nf32_flip.csv 2>&1 | tail -4
echo "== 与旧链(0.73134)对比 =="
python - <<'PY'
import csv
def rd(f):
    m={}
    with open(f) as fh:
        r=csv.reader(fh); next(r,None)
        for row in r:
            if len(row)>=2: m[row[0]]=row[1]
    return m
a=rd('outputs/sub_chain_nf32_flip.csv'); b=rd('outputs/sub_B_mains42_ths42.csv')
diff=sum(1 for k in a if a[k]!=b[k])
print(f"新链 vs 旧链 决策改动: {diff}/405")
from collections import Counter
print("类0数:", sum(1 for v in a.values() if v=='0'), " 去重类:", len(set(a.values())))
PY
echo DONE
