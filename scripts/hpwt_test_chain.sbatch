#!/bin/bash
#SBATCH -J cuhkx-hpwtc2
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
CK=outputs/main_hpwt_full/r2plus1d_depthir_full_seed42.pth
[ -f "$CK" ] || { echo "缺 $CK"; exit 1; }
echo "== 打包 main_hpwtfull int5 =="
python scripts/quantize_pack.py --name main_hpwtfull \
  --checkpoints "$CK" --bits 5 --out_dir outputs/pack 2>&1 | tail -1
echo "== test 推理: main_hpwtfull(16f) + th_nf32full(32f) flip =="
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/main_hpwtfull_fold0_int5.pth \
  --thermal outputs/pack/th_nf32full_fold0_int5.pth \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_hpwt_chain.csv 2>&1 | tail -3
echo "== 与锚对比(候选) =="
python - <<'PY'
import csv, re
def rd(f):
    r={}
    for row in csv.DictReader(open(f)):
        m=re.search(r"(SM_test_\d+)", row['path']); r[m.group(1) if m else row['path']]=int(row['prediction'])
    return r
try:
    a=rd('outputs/sub_chain_nf32_flip.csv'); c=rd('outputs/sub_hpwt_chain.csv')
    flips=[k for k in a if k in c and c[k]!=a[k]]
    print(f"main_hpwt 链 vs 锚: flips={len(flips)}/405 类0={sum(1 for v in c.values() if v==0)} 去重={len(set(c.values()))}")
except Exception as e:
    print("对比失败:", e)
PY
echo "== DONE =="
