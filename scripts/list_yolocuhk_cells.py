# -*- coding: utf-8 -*-
"""列出 yolo-for-cuhk-x.ipynb 所有 cell 概览。"""
import json

NB = r"notebooks/yolo-for-cuhk-x.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
print("总 cell 数:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    first = [l for l in src.split("\n") if l.strip()]
    head = first[0][:100] if first else ""
    print(f"[{i}] {c['cell_type']:8s} len={len(src):5d} | {head}")
