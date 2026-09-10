#!/usr/bin/env python3
"""导出易混淆类对的真实样例帧(Depth_Color RGB), 供人工核对孪生动作的相似度。
产物: outputs/confusion_examples/<a>_<b>.png (每个孪生对一张对比图, 行=动作, 列=帧(0.3/0.7 时刻))。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("data/Training/HAR/Depth_Color")
OUT = Path("outputs/confusion_examples")
OUT.mkdir(parents=True, exist_ok=True)

# 孪生类对(action_id, action_id, 备注)
PAIRS = [
    (13, 12, "Mop_the_floor ↔ Sweep_the_floor"),
    (22, 21, "Turn_pages ↔ Read_documents"),
    (8, 9, "Take&use_tableware ↔ Pour_drinks"),
    (8, 10, "Take&use_tableware ↔ Stir_drinks"),
    (37, 6, "Take_medicine ↔ Drink_water"),
    (18, 17, "Write ↔ Tap_keyboard"),
    (7, 6, "Eat_food ↔ Drink_water"),
    (36, 32, "Walk ↔ Stand_up"),
    (26, 24, "Play_games ↔ Use_mobile_phone"),
]

NAME = {}
for ad in sorted(ROOT.iterdir()):
    if ad.is_dir() and ad.name.split("_")[0].isdigit():
        NAME[int(ad.name.split("_")[0])] = ad.name


def pick_frames(cls, fracs=(0.3, 0.7)):
    d = ROOT / NAME[cls]
    users = sorted(x for x in d.iterdir() if x.is_dir())
    for u in users:
        for s in sorted(u.iterdir()):
            if not s.is_dir():
                continue
            files = sorted(x for x in s.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg"))
            if len(files) < 3:
                continue
            idxs = [int(min(len(files) - 1, max(0, f * len(files)))) for f in fracs]
            return [Image.open(str(files[i])).convert("RGB") for i in idxs], s
    return None, None


def contact(a_cls, b_cls, label):
    fa, sa = pick_frames(a_cls)
    fb, sb = pick_frames(b_cls)
    if fa is None or fb is None:
        print("skip", a_cls, b_cls)
        return
    imgs = fa + fb
    h = 200
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    W = sum(im.width for im in imgs) + 20
    H = h + 30
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 4), label, fill=(0, 0, 0))
    x = 10
    dr.text((x, 26), f"cls{a_cls} {NAME.get(a_cls,'?' )}   {sa}", fill=(200, 0, 0))
    for im in imgs[:2]:
        canvas.paste(im, (x, 44)); x += im.width + 5
    dr.text((10, 44 + h + 4), f"cls{b_cls} {NAME.get(b_cls,'?')}   {sb}", fill=(0, 0, 200))
    x = 10
    for im in imgs[2:]:
        canvas.paste(im, (x, 44 + h + 20)); x += im.width + 5
    out = OUT / f"{a_cls}_{b_cls}.png"
    canvas.save(out)
    print("saved", out)


for a, b, lab in PAIRS:
    contact(a, b, lab)
print("done")
