#!/bin/bash
# 09-11: 难对 fold1/fold2 补训 (fold0 已完成双塔难对 +5.18)
# 目的: 3 折难对模型 → OOF 难对集成推理 test (绕开 full 难对表错配)
# 用法: sbatch scripts/hardpair_fold_123.sbatch  (fold=1 与 fold=2, fold0已有)
#SBATCH -J cuhkx-hpfold
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/dual_hp_f0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) 难对 fold1/fold2 双塔补训 (main16+th16, hardpair_w=2) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M"; exit 1; }

# fold0 已有 (dual_hp_f0/main/r2plus1d34_depthir_fold0.pth + th)
for F in 1 2; do
  echo "===== fold$F main ====="
  [ -f outputs/dual_hp_f0/main/r2plus1d34_depthir_fold${F}.pth ] || \
  python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
    --fold "$F" --num_frames 16 --size 128 --batch_size 32 --workers 8 \
    --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 \
    --crop_cache bbox_train.json --modality depthir --hardpair_w 2.0 \
    --save_dir outputs/dual_hp_f0/main 2>&1 | tee logs/hpfold_main_${F}.log | tail -3

  echo "===== fold$F th ====="
  [ -f outputs/dual_hp_f0/th/r2plus1d34_thermal_fold${F}.pth ] || \
  python scripts/train_step1.py --backbone r2plus1d34 --weights ig65m_r2plus1d34.pth \
    --fold "$F" --num_frames 16 --size 128 --batch_size 32 --workers 8 \
    --lr 1e-4 --epochs 60 --label_smoothing 0.1 --aug_strength 2 \
    --crop_cache bbox_thermal_train.json --modality thermal --hardpair_w 2.0 \
    --save_dir outputs/dual_hp_f0/th 2>&1 | tee logs/hpfold_th_${F}.log | tail -3
done

echo "===== 3 折 OOF logits 提取 (难对) ====="
python scripts/extract_oof_logits.py --main --mode oof --flip \
  --fold_ckpts outputs/dual_hp_f0/main/r2plus1d34_depthir_fold0.pth \
              outputs/dual_hp_f0/main/r2plus1d34_depthir_fold1.pth \
              outputs/dual_hp_f0/main/r2plus1d34_depthir_fold2.pth \
  --crop bbox_train.json --out outputs/oof/main_hpdual_oof.pkl || echo "⚠️ main"
python scripts/extract_oof_logits.py --thermal --mode oof --flip \
  --fold_ckpts outputs/dual_hp_f0/th/r2plus1d34_thermal_fold0.pth \
              outputs/dual_hp_f0/th/r2plus1d34_thermal_fold1.pth \
              outputs/dual_hp_f0/th/r2plus1d34_thermal_fold2.pth \
  --crop bbox_thermal_train.json --out outputs/oof/thermal_hpdual_oof.pkl || echo "⚠️ th"

echo "===== 3 折难对 OOF 融合 vs 基线 OOF (难对区/非难对) ====="
python - <<'PY'
import pickle, numpy as np
def sf(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
H11={(0,1),(6,37),(7,37),(8,9),(8,10),(8,15),(8,18),(11,14),(17,18),(18,20),(38,39)}
for tag,mp,tp in [("基线",'outputs/oof/main_oof.pkl','outputs/oof/thermal_oof.pkl'),
                   ("难对3折",'outputs/oof/main_hpdual_oof.pkl','outputs/oof/thermal_hpdual_oof.pkl')]:
    try:
        MM=pickle.load(open(mp,'rb')); TT=pickle.load(open(tp,'rb'))
        tot=corr=he=ne=0
        for f in range(3):
            for k in MM[f]:
                if k not in TT[f]: continue
                g=int(k.split('/')[0])
                fu=sf(np.asarray(MM[f][k]))+sf(np.asarray(TT[f][k]))
                ok=(int(fu.argmax())==g); tot+=1; corr+=ok
                if any(g in q for q in H11): he+=(not ok)
                else: ne+=(not ok)
        print(f"[{tag}] 3折融合 acc={corr/tot:.4f} 难对错={he} 非难对错={ne}")
    except Exception as e:
        print(f"[{tag}] ERROR {e}")
PY

echo "==== 难对 fold1/2 补训 DONE ===="
