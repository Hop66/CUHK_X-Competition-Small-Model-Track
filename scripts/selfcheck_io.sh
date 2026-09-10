#!/bin/bash
# NFS IO 自检：源数据 200 帧读速慢则 exit 3（slurm 自然重分节点/时段）。
# source 进 sbatch 的 cd $HOME/Multimodal 之后。
set -uo pipefail
D="data/Training/HAR/Depth_Color/0_Wash_face/user16/1-1-1"
if [ ! -d "$D" ]; then echo "IO-check: 数据缺失?"; exit 0; fi
t0=$(date +%s%N)
ls "$D"/*.png | head -200 | while read -r x; do dd if="$x" of=/dev/null bs=1k 2>/dev/null; done
t1=$(date +%s%N)
dt=$(( (t1 - t0) / 1000000 ))  # ms
echo "IO 自检: 200 帧读 ${dt} ms (host: $(hostname))"
if [ "$dt" -gt 12000 ]; then
  echo "IO 自检失败(慢 NFS>12s) → exit 3, 重分"
  exit 3
fi
echo "IO 自检通过"
