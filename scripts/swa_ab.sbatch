#!/bin/bash
#SBATCH -J cuhkx-swa
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

echo "== D2: SWA 单成员化 | main_avg(3seed 参数平均) 替换 main_s42 进锚链 | 雷=重TTA收缩30类, 这里只 flip =="
MA=outputs/pack/main_s42_fold0_int5.pth
MA_AVG=outputs/pack/main_avg_int5.pth
TH=outputs/pack/th_nf32full_fold0_int5.pth

echo "----- a. 复现锚 (main_s42+th_nf32full, flip) -----"
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main "$MA" --thermal "$TH" \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_swa_rep.csv 2>&1 | tail -3

echo "----- b. SWA: main_avg + th_nf32full, flip -----"
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main "$MA_AVG" --thermal "$TH" \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_swa_mainavg.csv 2>&1 | tail -3

echo "----- c. SWA 双: main_avg + thermal_avg(16f th), flip (对照) -----"
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main "$MA_AVG" --thermal outputs/pack/thermal_avg_int5.pth \
  --num_frames 16 --thermal_frames 16 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_swa_bothavg.csv 2>&1 | tail -3

echo "== 与锚对比 flip 量 =="
python - <<'PY'
import csv, re
def rd(f):
    r={}
    for row in csv.DictReader(open(f)):
        m=re.search(r"(SM_test_\d+)", row['path']); k=m.group(1) if m else row['path']
        r[k]=int(row['prediction'])
    return r
anchor=rd('outputs/sub_chain_nf32_flip.csv')
for name in ['sub_swa_rep','sub_swa_mainavg','sub_swa_bothavg']:
    c=rd(f'outputs/{name}.csv')
    flips=[k for k in anchor if k in c and c[k]!=anchor[k]]
    zero=sum(1 for k in c if c[k]==0)
    print(f"{name}: vs锚 flips={len(flips)}/405 类0={zero} 去重={len(set(c.values()))}")
print("(flips 大=SWA 改变多; SWA 雷=重TTA收缩30类, 这里纯flip 看是否还收缩)")
PY
echo "== DONE =="
