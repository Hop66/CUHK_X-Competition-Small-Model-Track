#!/bin/bash
# ============================================================
# CUHK-X Iter9: 蒸馏战役（0.8 主攻——把 multi-seed 集成收益固化进单流 student）
# 阶段: 0) 教师软标签(main×3 / thermal×3)  1) distill main --full 80ep
#       2) distill thermal --full 80ep
# 语义要点:
#   - teacher = prob-avg 多 seed（全量数据, 与提交同源）
#   - student = 同架构 R2+1D, CE + w·T²·KL(softmax/T vs teacher/T)
#   - --full: 用全部 2931 训练 clip（教师软标签 + 硬标签双监督）
# 用法: sbatch scripts/train_distill.sbatch
# ============================================================
#SBATCH -J cuhkx-distill
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$HOME/Multimodal"
mkdir -p logs outputs/teacher outputs/main_distill outputs/thermal_distill
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} | Iter9 Distill ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MS42=outputs/pack/main_s42_fold0_int5.pth
MS77=outputs/pack/main_s777_fold0_int5.pth
MPS=outputs/pack/main_ps_fold0_int5.pth
TS42=outputs/pack/thermal_s42_fold0_int5.pth
TS77=outputs/pack/thermal_s777_fold0_int5.pth
TAVG=outputs/pack/thermal_avg_int5.pth
TM=outputs/teacher/teacher_main_train.pkl
TT=outputs/teacher/teacher_thermal_train.pkl

echo "===== [0] 教师软标签(缺则生成; v2 修了 prep 缺 __main__ 的bug) ====="
[ -f "$TM" ] || python scripts/prep_distill_teacher.py --modality main --root train --quantize --flip \
  --ckpts $MS42 $MS77 $MPS --out "$TM" || echo "⚠️ teacher main"
[ -f "$TT" ] || python scripts/prep_distill_teacher.py --modality thermal --root train --quantize --flip \
  --ckpts $TS42 $TS77 $TAVG --out "$TT" || echo "⚠️ teacher thermal"
ls -la "$TM" "$TT"

echo "===== [1] distill main --full (IG65M init) ====="
python scripts/train_distill_stream.py --modality main --weights ig65m_r2plus1d34.pth \
  --teacher "$TM" --full --epochs 80 \
  --distill_w 0.5 --T 3.0 --lr 1e-4 --batch_size 32 --workers 8 --seed 42 \
  --save_dir outputs/main_distill || echo "⚠️ distill main"

echo "===== [2] distill thermal --full (IG65M init) ====="
python scripts/train_distill_stream.py --modality thermal --weights ig65m_r2plus1d34.pth \
  --teacher "$TT" --full --epochs 80 \
  --distill_w 0.5 --T 3.0 --lr 1e-4 --batch_size 32 --workers 8 --seed 42 \
  --save_dir outputs/thermal_distill || echo "⚠️ distill thermal"

echo ""
echo "==== Iter9 Distill DONE ===="
ls -la outputs/main_distill/*  outputs/thermal_distill/* 2>/dev/null
echo "下一步: quantize_pack int5 → 与现有配方融合+TTA 出 CSV → LB"
