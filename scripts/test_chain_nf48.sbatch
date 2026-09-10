#!/bin/bash
#SBATCH -J cuhkx-thchain48
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
CK=outputs/th_nf48_full/r2plus1d34_thermal_full_seed42.pth
[ -f "$CK" ] || { echo "缺 $CK"; exit 1; }
echo "== 打包 th_nf48full int5 =="
python scripts/quantize_pack.py --name th_nf48full --checkpoints "$CK" --bits 5 \
  --out_dir outputs/pack 2>&1 | tail -1
echo "== test 推理: main_s42(16f) + th_nf48full(48f, --thermal_frames 48) flip =="
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf48full_fold0_int5.pth \
  --num_frames 16 --thermal_frames 48 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_chain_nf48_flip.csv 2>&1 | tail -4
echo "== 与 0.75 锚链(sub_chain_nf32_flip)对比 =="
python - <<'PY'
import csv
def rd(f):
    m={}
    with open(f) as fh:
        r=csv.reader(fh); next(r,None)
        for row in r:
            if len(row)>=2: m[row[0]]=row[1]
    return m
a=rd('outputs/sub_chain_nf48_flip.csv'); b=rd('outputs/sub_chain_nf32_flip.csv')
diff=sum(1 for k in a if a[k]!=b[k])
print(f"nf48链 vs nf32链(0.75锚) 决策改动: {diff}/405")
PY
echo DONE
