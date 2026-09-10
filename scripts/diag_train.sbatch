#!/bin/bash
#SBATCH -J cuhkx-diag
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd "$HOME/Multimodal"; mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cuhk_x
export HF_ENDPOINT=https://hf-mirror.com
echo "diag start $(date +%T)"
python -u - <<'PY'
import time, sys, os
t0=time.time(); print("step1 import torch...", flush=True)
import torch
print(f"torch ok {time.time()-t0:.1f}s", flush=True)
sys.path.insert(0,'.')
t0=time.time(); print("step2 build_train_index...", flush=True)
from pathlib import Path
from src.dataset import build_train_index
clips=build_train_index(Path('data/Training/HAR'))
print(f"build index ok n={len(clips)} {time.time()-t0:.1f}s", flush=True)
t0=time.time(); print("step3 load crop+build ds...", flush=True)
import json
from src.dataset import DepthIRVideoDataset
from src.split import split_by_subject
folds=split_by_subject(clips,3); tr=[clips[i] for i in folds[0][0]]
crop=json.loads(open('bbox_train.json').read())
ds=DepthIRVideoDataset(tr,16,128,True,crop,aug_strength=4)
print(f"ds ok len={len(ds)} {time.time()-t0:.1f}s", flush=True)
t0=time.time(); print("step4 model load...", flush=True)
from src.model import build_model
m=build_model('r2plus1d34',num_classes=40,in_channels=4,n_segment=16)
print(f"model ok {time.time()-t0:.1f}s", flush=True)
t0=time.time(); print("step5 first batch...", flush=True)
from torch.utils.data import DataLoader
dl=DataLoader(ds,batch_size=32,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
b=next(iter(dl)); print(f"first batch ok {tuple(b[0].shape)} {time.time()-t0:.1f}s", flush=True)
print("DIAG-ALL-OK", flush=True)
PY
echo "diag done $(date +%T)"
