#!/bin/bash
#SBATCH -J cuhkx-probe
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
hostname
echo "TMPDIR=$TMPDIR"
df -h /tmp | tail -1
df -h "$TMPDIR" 2>/dev/null | tail -1 || df -h . | tail -1
mount | grep -E " /tmp | /home | /local " 2>/dev/null | head
