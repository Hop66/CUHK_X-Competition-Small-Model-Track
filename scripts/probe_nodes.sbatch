#!/bin/bash
#SBATCH -J cuhkx-prbn
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
hostname
mount | grep " /home " | head -1
D="$HOME/Multimodal/data/Training/HAR/Depth_Color/0_Wash_face/user16/1-1-1"
echo "帧 $(ls $D | wc -l)"; f=$(ls $D | head -1)
echo "== 500 帧读测速 =="
time (for x in $(ls $D/*.png | head -500); do dd if=$x of=/dev/null bs=1k 2>/dev/null; done)
