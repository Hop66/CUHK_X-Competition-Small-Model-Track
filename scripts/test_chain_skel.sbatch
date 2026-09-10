#!/bin/bash
#SBATCH -J cuhkx-skch
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH -t 01:29:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/pack
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
CK=outputs/skeleton_finetune_full/finetune_full.pth
[ -f "$CK" ] || { echo "缺 $CK"; exit 1; }
echo "== 打包 skel_full int5 =="
python scripts/quantize_pack.py --name skel_full --checkpoints "$CK" --bits 5 \
  --out_dir outputs/pack 2>&1 | tail -1
ls -la outputs/pack/skel_full_fold0_int5.pth
echo "== 三路链: main_s42(16f) + th_nf32full(32f) + skel_full, α=0.075 =="
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_s42_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --skeleton outputs/pack/skel_full_fold0_int5.pth \
  --w_main 0.4625 --w_thermal 0.4625 --w_skeleton 0.075 \
  --num_frames 16 --thermal_frames 32 --skeleton_frames 24 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_chain_skel.csv 2>&1 | tail -6
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
a=rd('outputs/sub_chain_skel.csv'); b=rd('outputs/sub_chain_nf32_flip.csv')
diff=sum(1 for k in a if a[k]!=b[k])
print(f"三路链 vs 0.75锚链 决策改动: {diff}/405")
PY
echo DONE
