#!/bin/bash
# ============================================================
# CUHK-X 09-11: 双塔同步难对重权 full(seed42) —— 主链锚替换候选
# 前置: fold0 验证(59788) 难对双塔有效(acc>基线且难对区错↓) 才启此清单
# 配置与主链锚完全一致 + --hardpair_w:
#   main_s42(16f) + th_nf32full(32f) → prob_avg + flip (LB 0.75124 锚)
# 唯一变量: 两个塔都加难对重权(同一 HARD_PAIRS=11 对可学难对)
# 用法: sbatch scripts/hardpair_dual_full.sbatch   (等 59788 判读后再提交)
# ============================================================
#SBATCH -J cuhkx-hpdualfull
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/dual_hp_full
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) 双塔同步难对重权 full(seed42): main16 + th32 ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }

echo "===== [1] main full(seed42, 16f, 80ep) + hardpair_w=2.0 ====="
python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --num_frames 16 --size 128 --batch_size 32 --workers 8 --lr 1e-4 --epochs 80 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_train.json \
  --modality depthir --full --seed 42 --hardpair_w 2.0 --full_save_every 10 \
  --save_dir outputs/dual_hp_full/main 2>&1 | tee logs/hpdualfull_main.log | tail -4 || echo "⚠️ main"

echo "===== [2] th full(seed42, 32f, 80ep) + hardpair_w=2.0 (帧数红利保留, 同步难对) ====="
python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
  --num_frames 32 --size 128 --batch_size 16 --workers 8 --lr 1e-4 --epochs 80 \
  --label_smoothing 0.1 --aug_strength 2 --crop_cache bbox_thermal_train.json \
  --modality thermal --full --seed 42 --hardpair_w 2.0 --full_save_every 10 \
  --save_dir outputs/dual_hp_full/th 2>&1 | tee logs/hpdualfull_th.log | tail -4 || echo "⚠️ th"

echo "===== [3] int5 打包 (与锚同规格量化) ====="
python scripts/quantize_pack.py --name hpdual_main --checkpoints outputs/dual_hp_full/main/r2plus1d34_depthir_full_seed42.pth --out_dir outputs/pack --bits 5 || echo "⚠️ pack main"
python scripts/quantize_pack.py --name hpdual_th --checkpoints outputs/dual_hp_full/th/r2plus1d34_thermal_full_seed42.pth --out_dir outputs/pack --bits 5 || echo "⚠️ pack th"
echo "===== [4] 组链提交 (prob_avg + flip + 主链同款) ====="
python -u scripts/ensemble_inference.py --prob_avg --flip_tta --quantize \
  --main outputs/pack/hpdual_main_fold0_int5.pth \
  --thermal outputs/pack/hpdual_th_fold0_int5.pth \
  --num_frames 16 --thermal_frames 32 \
  --main_crop bbox_test.json --thermal_crop bbox_thermal_test.json \
  --output outputs/sub_hpdual_full_flip.csv 2>&1 | tail -4
echo "== 与锚对比 =="
python - <<'PY'
import csv
def rd(f):
    m={}
    with open(f) as fh:
        r=csv.reader(fh); next(r,None)
        for row in r:
            if len(row)>=2: m[row[0]]=row[1]
    return m
a=rd('outputs/sub_hpdual_full_flip.csv')
b=rd('outputs/sub_chain_nf32_flip.csv')
diff=sum(1 for k in a if b.get(k) and a[k]!=b[k])
print(f"难对双塔 vs 主链锚 决策改动: {diff}/405")
print("去重类数:", len(set(a.values())))
PY
echo "==== 双塔同步难对 full DONE (CSV: outputs/sub_hpdual_full_flip.csv) ===="
