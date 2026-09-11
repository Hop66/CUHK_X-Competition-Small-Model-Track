#!/bin/bash
# 09-11 P1 轻量方向对照 (baseline 60405 完成后运行): TSM / MixStyle 同协议 fold0
# 对比目标: p1_base main fold0 0.6502 (R2+1D34, env-balanced)
# 变量(各自独立, 一变量一实验):
#   TSM-only   main16 (tsm_mobilenet, 无MixStyle)
#   TSM+MS     main16 (use_mixstyle=True)
# 判定: fold0 val > 0.6502 → 进全量/full 候选; 用 fair_eval + locked 复核
#SBATCH -J cuhkx-p1tsm
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs outputs/p1_tsm
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "==== $(hostname) P1 TSM/MixStyle fold0 对照 (env-balanced) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
FS="outputs/dev_folds.json"

echo "===== [TSM-only] main16 tsm_mobilenet fold0 ====="
CK=outputs/p1_tsm/main/tsm_mobilenet_depthir_fold0.pth
[ -f "$CK" ] || python scripts/train_step1.py --backbone tsm_mobilenet \
    --fold_split "$FS" --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 \
    --lr 1e-3 --epochs 60 --label_smoothing 0.1 --aug_strength 2 \
    --crop_cache bbox_train.json --modality depthir \
    --save_dir outputs/p1_tsm/main 2>&1 | tee logs/p1tsm_main.log | tail -3

echo "===== [TSM+MS] main16 tsm_mobilenet + MixStyle fold0 ====="
CK2=outputs/p1_tsm/ms/tsm_mobilenet_depthir_fold0.pth
[ -f "$CK2" ] || python scripts/train_step1.py --backbone tsm_mobilenet --use_mixstyle \
    --fold_split "$FS" --fold 0 --num_frames 16 --size 128 --batch_size 32 --workers 8 \
    --lr 1e-3 --epochs 60 --label_smoothing 0.1 --aug_strength 2 \
    --crop_cache bbox_train.json --modality depthir \
    --save_dir outputs/p1_tsm/ms 2>&1 | tee logs/p1tsm_ms.log | tail -3

echo "==== 结果 vs baseline fold0 0.6502 ===="
grep -h "fold 0 best" logs/p1tsm_main.log logs/p1tsm_ms.log 2>/dev/null
echo "==== P1 TSM DONE ===="
