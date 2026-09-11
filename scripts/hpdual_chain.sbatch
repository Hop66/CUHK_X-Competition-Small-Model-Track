#!/bin/bash
#SBATCH -J cuhkx-hpchain
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
echo "== 难对双塔 full 组链 (main16 + th32 + flip) =="
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/hpdual_main_fold0_int5.pth \
  --thermal outputs/pack/hpdual_th_fold0_int5.pth \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_hpdual_full_flip.csv
echo "== 锚对比 =="
python - <<'PY'
import csv
def rd(f):
    m={}
    with open(f) as fh:
        r=csv.reader(fh); next(r,None)
        for row in r:
            if len(row)>=2: m[row[0]]=row[1]
    return m
a=rd('outputs/sub_hpdual_full_flip.csv'); b=rd('outputs/sub_chain_nf32_flip.csv')
diff=sum(1 for k in a if b.get(k) and a[k]!=b[k])
print(f"难对full vs 锚: flips={diff}/405 去重类={len(set(a.values()))}")
PY
echo "== CHAIN DONE =="
