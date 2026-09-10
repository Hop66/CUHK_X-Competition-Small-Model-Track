#!/usr/bin/env python3
"""导出孪生对的动态网格(Depth_Color 连续8帧 2x4), 供肉眼核对「动态下能否分辨」。
产物: outputs/confusion_examples/<a>_<b>_dyn.png
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("data/Training/HAR/Depth_Color")
OUT = Path("outputs/confusion_examples")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    (13, 12), (22, 21), (8, 9), (8, 10), (37, 6),
    (18, 17), (7, 6), (36, 32), (26, 24),
]
NAME = {}
for ad in sorted(ROOT.iterdir()):
    if ad.is_dir() and ad.name.split("_")[0].isdigit():
        NAME[int(ad.name.split("_")[0])] = ad.name


def pick_clip(cls):
    d = ROOT / NAME[cls]
    for u in sorted(x for x in d.iterdir() if x.is_dir()):
        for s in sorted(u.iterdir()):
            if not s.is_dir():
                continue
            files = sorted(x for x in s.iterdir() if x.suffix.lower() == ".png")
            if len(files) >= 8:
                return files, s
    return None, None


def frames_grid(cls, n=8):
    files, s = pick_clip(cls)
    if files is None:
        return None, None
    idxs = [int(f * (len(files) - 1)) for f in np.linspace(0, 1, n)]
    return [Image.open(str(files[i])).convert("RGB") for i in idxs], s


import numpy as np

for a, b in PAIRS:
    fa, sa = frames_grid(a)
    fb, sb = frames_grid(b)
    if fa is None or fb is None:
        print("skip", a, b); continue
    h = 140
    fa = [im.resize((int(im.width * h / im.height), h)) for im in fa]
    fb = [im.resize((int(im.width * h / im.height), h)) for im in fb]
    W = sum(im.width for im in fa[:4]) + 30
    H = 2 * (h + 30) + 44
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(canvas)
    dr.text((8, 4), f"cls{a} {NAME.get(a,'?')}  {sa}", fill=(200, 0, 0))
    dr.text((8, h + 32), f"cls{b} {NAME.get(b,'?')}  {sb}", fill=(0, 0, 200))
    for row, imgs, y0 in ((0, fa, 20), (1, fb, h + 44)):
        x = 6
        for im in imgs:
            canvas.paste(im, (x, y0)); x += im.width + 5
    out = OUT / f"{a}_{b}_dyn.png"
    canvas.save(out)
    print("saved", out)
print("done")
