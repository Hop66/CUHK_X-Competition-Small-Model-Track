#!/bin/bash
#SBATCH -J cuhkx-geomean
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

echo "== D4: 0.75 锚链(main_s42+th_nf32full) geomean+温度 AB | fold扫描 +0.42 (0.6777→0.6819) =="
MA=outputs/pack/main_s42_fold0_int5.pth
TH=outputs/pack/th_nf32full_fold0_int5.pth
[ -f "$MA" ] && [ -f "$TH" ] || { echo "缺 ckpt"; exit 1; }

echo "----- a. 复现锚(arith prob_avg, 应 100% == sub_chain_nf32_flip.csv) -----"
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main "$MA" --thermal "$TH" \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_geo_rep.csv 2>&1 | tail -3

echo "----- b. geomean T=1 -----"
python -u scripts/ensemble_inference.py --geomean --flip_tta --quantize \
  --main "$MA" --thermal "$TH" \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_geo_t1.csv 2>&1 | tail -3

echo "----- c. geomean T=2 (fold 上 T2=0.6799) -----"
python -u scripts/ensemble_inference.py --geomean --temp 2.0 --flip_tta --quantize \
  --main "$MA" --thermal "$TH" \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_geo_t2.csv 2>&1 | tail -3

echo "----- d. arith T=3 (fold T3=0.6810) -----"
python -u scripts/ensemble_inference.py --prob_avg --temp 3.0 --flip_tta --quantize \
  --main "$MA" --thermal "$TH" \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_geo_t3.csv 2>&1 | tail -3

echo "== 与锚对比 flip 量 =="
python - <<'PY'
import csv
def rd(f):
    r={}
    for row in csv.DictReader(open(f)):
        import re
        m=re.search(r"(SM_test_\d+)", row['path']); k=m.group(1) if m else row['path']
        r[k]=int(row['prediction'])
    return r
anchor=rd('outputs/sub_chain_nf32_flip.csv')
for name in ['sub_geo_rep','sub_geo_t1','sub_geo_t2','sub_geo_t3']:
    c=rd(f'outputs/{name}.csv')
    flips=sum(1 for k in anchor if k in c and c[k]!=anchor[k])
    zero=sum(1 for k in c if c[k]==0)
    print(f"{name}: vs锚 flips={flips}/405 类0={zero} 去重={len(set(c.values()))}")
print("(flips=0 即完全复现锚；>0 为待 LB 实测的变化方向)")
PY
echo "== DONE =="
