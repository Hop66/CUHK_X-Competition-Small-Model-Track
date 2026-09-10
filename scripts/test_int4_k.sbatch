#!/bin/bash
#SBATCH -J cuhkx-int4k
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
CR="--main_crop bbox_test.json --thermal_crop bbox_thermal_test.json"
echo "== int5 K_2m1th(参考, 已有) =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int5.pth outputs/pack/main_s777_fold0_int5.pth \
  --thermal outputs/pack/thermal_s42_fold0_int5.pth $CR \
  --output outputs/sub_K_2m1th_int5_ref.csv
echo "== int4 K_2m1th (A形态, 93MB 可提交) =="
python scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int4.pth outputs/pack/main_s777_fold0_int4.pth \
  --thermal outputs/pack/thermal_s42_fold0_int4.pth $CR \
  --output outputs/sub_K_2m1th_int4_flip.csv
echo "== int4 决策一致性(相对 int5 改动率) =="
python - <<'PY'
import csv
def rd(f):
    m={}
    with open(f) as fh: r=csv.reader(fh); next(r,None)
    for rows in r:
        if len(rows)>=2: m[rows[0]]=rows[1]
    return m
a=rd('outputs/sub_K_2m1th_int4_flip.csv'); b=rd('outputs/sub_K_2m1th_int5_ref.csv')
common=set(a)&set(b)
diff=sum(1 for k in common if a[k]!=b[k])
print(f"int4 vs int5 决策改动: {diff}/{len(common)} = {diff/len(common)*100:.1f}%")
print(f"int4 类0={sum(1 for v in a.values() if v=='0')}/405 去重类={len(set(a.values()))}")
PY
echo "== pack 总大小(3个int4) =="
du -m outputs/pack/main_s42_fold0_int4.pth outputs/pack/main_s777_fold0_int4.pth outputs/pack/thermal_s42_fold0_int4.pth | awk '{s+=$1} END{print s" MB"}'
echo "== DONE =="
