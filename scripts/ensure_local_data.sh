#!/bin/bash
# 确保训练数据在节点本地盘(/tmp，SSD/LVM)，返回本地 root。
# 用法: source scripts/ensure_local_data.sh  → 变量 $LOCAL_TRAIN_ROOT
# 数据 42G，同节点只同步一次（/tmp/har_<user> 缓存）。
set -uo pipefail
LOCAL=/tmp/har_${USER##*/}
mkdir -p "$LOCAL"
if [ ! -d "$LOCAL/Training/HAR" ]; then
  echo "== 同步数据到本地盘($LOCAL) $(date +%T) =="
  mkdir -p "$LOCAL/Training"
  rsync -a "${1:-data}/Training/" "$LOCAL/Training/"
  echo "== 同步完成 $(date +%T) =="
else
  echo "== 本地缓存已存在: $LOCAL =="
fi
export LOCAL_TRAIN_ROOT="$LOCAL/Training/HAR"
