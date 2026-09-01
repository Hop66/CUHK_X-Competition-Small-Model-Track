#!/bin/bash
# ============================================================
# CUHK-X —— Main 双流（静态 DepthIR 固定框 + 骨架运动同源增强）fold0
# 动机：main track_crop（逐帧跟人 + bbox 轨迹）两版均 -9pt 判负——轨迹 4 维太单薄、裁剪有害；
#       骨架与 IR/depth 同相机场景（同 key）、3D 米制 17 关节信息量大 → 用骨架运动当 main 同源增强
# 架构：S = R2+1D-34(IG65M) 4ch 固定框(crop bbox_train.json) | M = MotionNet(motion[T,29])
#       融合 fused = α·softmax(S)+(1-α)·softmax(M)，α 可学；loss=CE(S)+CE(M)+CE(fused)
# 前提：outputs/motion_cache.pkl（extract_motion_cache.py 已建；[0] 自动补）
# 对照：main 固定框 fold0 = 0.6609
#   [1] main S-only（同机复核 0.6609）
#   [2] main S + 骨架M（主实验，判读 > [1] 且 > 0.6609 → 同源骨架给 main 真增益）
# 用法: sbatch scripts/main_dual.sbatch
# ============================================================
#SBATCH -J cuhkx-main-dual
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="$HOME/Multimodal"
cd "$PROJECT_DIR"
mkdir -p logs outputs/main_dual
export HF_ENDPOINT=https://hf-mirror.com
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cuhk_x

echo "==== node: $(hostname) | job: ${SLURM_JOB_ID} ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f ig65m_r2plus1d34.pth ] || { echo "❌ 缺 IG65M 权重"; exit 1; }
CROP=bbox_train.json
[ -f "$CROP" ] || { echo "⚠️ 缺 $CROP（main 固定框）"; exit 1; }
MC=outputs/motion_cache.pkl
EP=60

echo "########## [0/2] 骨架运动缓存（缺则构建）##########"
if [ ! -f "$MC" ]; then
  python scripts/extract_motion_cache.py || echo "⚠️ 运动缓存构建失败（[2] 的 M 流将降级）"
fi

echo "########## [1/2] main S-only（同机复核 0.6609）##########"
python scripts/train_main_dual.py --fold 0 --no_motion \
  --num_frames 16 --size 128 --epochs $EP --lr 1e-4 --batch_size 32 \
  --weights ig65m_r2plus1d34.pth --crop_cache "$CROP" \
  --save_dir outputs/main_dual --workers 8 \
  || echo "⚠️ [1] main S-only 失败"

echo "########## [2/2] main S + 骨架M（主实验）##########"
python scripts/train_main_dual.py --fold 0 \
  --num_frames 16 --size 128 --epochs $EP --lr 1e-4 --batch_size 32 \
  --weights ig65m_r2plus1d34.pth --crop_cache "$CROP" --motion_cache "$MC" \
  --save_dir outputs/main_dual --workers 8 \
  || echo "⚠️ [2] main S+M 失败"

echo "==== MAIN DUAL DONE ===="
echo "结果: outputs/main_dual/main_{S,SM}_fold0.pth"
echo "判读: [2](SM) > [1](S) 且 > 0.6609 → 同源骨架运动给 main 真增益 → fold1/2 + 全量 + 入 0.73"
