#!/bin/bash
# 09-11 P1步骤1: 重建公平 baseline —— 环境平衡协议(locked 不参与) 重训 main+th 3折
# 目的: 新可信锚(用于替代旧的 split_by_subject(不平衡)+无locked 的 0.75124 链)
# 协议: 一实验一变量; 与旧锚唯一差异 = fold 划分(环境平衡)+locked 剔除
# 判定(out fs 后): dev 折 OOF 融合 + locked 4人判据 (fair_eval.py)
#SBATCH -J cuhkx-p1base
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/p1_base
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) P1 环境平衡 baseline (main16 + th16, 无难对) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }
FS="outputs/dev_folds.json"

for F in 0 1 2; do
  echo "===== [main] fold$F ====="
  python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
    --fold_split "$FS" --fold "$F" --num_frames 16 --size 128 --batch_size 32 --workers 8 \
    --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 \
    --crop_cache bbox_train.json --modality depthir \
    --save_dir outputs/p1_base/main 2>&1 | tee logs/p1base_main_${F}.log | tail -3

  echo "===== [thermal] fold$F (16f, 与旧锚 th32 不同 — 先用16 对齐 A/B 协议) ====="
  python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
    --fold_split "$FS" --fold "$F" --num_frames 16 --size 128 --batch_size 32 --workers 8 \
    --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 \
    --crop_cache bbox_thermal_train.json --modality thermal \
    --save_dir outputs/p1_base/th 2>&1 | tee logs/p1base_th_${F}.log | tail -3
done

echo "===== OOF 提取 (dev 折, 不含 locked) ====="
python scripts/extract_oof_logits.py --main --mode oof --flip \
  --fold_ckpts outputs/p1_base/main/r2plus1d34_depthir_fold0.pth \
              outputs/p1_base/main/r2plus1d34_depthir_fold1.pth \
              outputs/p1_base/main/r2plus1d34_depthir_fold2.pth \
  --crop bbox_train.json --out outputs/oof/p1_main_oof.pkl || echo "⚠️ main oof"
python scripts/extract_oof_logits.py --thermal --mode oof --flip \
  --fold_ckpts outputs/p1_base/th/r2plus1d34_thermal_fold0.pth \
              outputs/p1_base/th/r2plus1d34_thermal_fold1.pth \
              outputs/p1_base/th/r2plus1d34_thermal_fold2.pth \
  --crop bbox_thermal_train.json --out outputs/oof/p1_th_oof.pkl || echo "⚠️ th oof"

echo "===== dev 折 融合报告 ====="
python - <<'PY'
import pickle, numpy as np, json
def sf(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
MM=pickle.load(open('outputs/oof/p1_main_oof.pkl','rb'))
TT=pickle.load(open('outputs/oof/p1_th_oof.pkl','rb'))
tot=corr=0; per={}
for f in range(3):
    for k in MM[f]:
        if k not in TT[f]: continue
        g=int(k.split('/')[0]); s=sf(np.asarray(MM[f][k]))+sf(np.asarray(TT[f][k]))
        ok=(int(s.argmax())==g); tot+=1; corr+=ok
        sub=k.split('/')[1]; per.setdefault(sub,[0,0]); per[sub][1]+=1; per[sub][0]+=ok
print(f"[P1 baseline dev-3折] acc={corr/tot:.4f} ({corr}/{tot})")
pa={k:c/n for k,(c,n) in per.items() if n}
print(f"  主访者mean={np.mean(list(pa.values())):.4f} std={np.std(list(pa.values())):.4f} worst={min(pa.values()):.4f}")
# locked 不在 dev OOF (已被剔除) → 需单独 full 训练才能在 locked 判据; 这里先报 dev
print("  (locked 判据需 full 训练后 fair_eval; 本次只报 dev 折)")
PY

echo "==== P1 baseline DONE ===="
