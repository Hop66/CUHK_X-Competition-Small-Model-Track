#!/usr/bin/env python3
"""CUHK-X —— 批量 bbox 可视化验框（服务器跑，一行全流程）

对 track_crop 两个数据源做一次性「体检」：
  - 从 bbox_ir_perframe.json / bbox_thermal_perframe.json 各抽 N 个代表 clip
    （按 action_id 分层 → 覆盖不同动作；组内 linspace → 覆盖不同被试/样本）
  - 逐个调用 viz_track_boxes.py（cache 模式，按 clip_key 自动反查真实帧目录，无需手拼路径）
  - 汇总：通过 clip 数 / 覆盖 / 面积 / 中心偏移 / 位移平滑 / 失败清单

产出:
  - stdout 汇总表
  - outputs/viz_track/<box>/<clip>__<...>/frames_boxes.jpg + trajectory.jpg（每个抽查 clip 一组）

前置:
  - bbox_ir_perframe.json + bbox_thermal_perframe.json（detect_perframe.sbatch 产出）
  - viz_track_boxes.py 已更新（cache 模式自动定位目录版）

用法（服务器）:
  cd ~/Multimodal
  python scripts/viz_track_batch.py                 # 默认两个 json 各抽 6 个
  python scripts/viz_track_batch.py --n 10          # 抽更多
  python scripts/viz_track_batch.py --boxes bbox_ir_perframe.json   # 只看 main/IR
判读: 全部通过 ✅ 后再提交 main_track_crop / thermal_track_crop
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "scripts" / "viz_track_boxes.py"


def pick_keys(box_path: Path, n: int) -> "list[str]":
    """按 action_id 分层抽 n 个代表 clip key（保证动作类别多样）。"""
    d = json.loads(box_path.read_text(encoding="utf-8"))
    keys = sorted(d.keys())
    groups = defaultdict(list)
    for k in keys:
        groups[k.split("/")[0]].append(k)
    action_ids = sorted(groups, key=int)
    per = max(1, int(np.ceil(n / len(action_ids))))
    chosen = []
    for aid in action_ids:
        g = groups[aid]
        idx = np.linspace(0, len(g) - 1, min(per, len(g))).round().astype(int)
        for i in idx:
            chosen.append(g[i])
        if len(chosen) >= n:
            break
    return chosen[:n]


def grab(prefix: str, text: str) -> str:
    m = re.search(re.escape(prefix) + r"([0-9.]+)", text)
    return m.group(1) if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", nargs="+",
                    default=["bbox_ir_perframe.json", "bbox_thermal_perframe.json"])
    ap.add_argument("--n", type=int, default=6, help="每个 bbox json 抽多少个代表 clip")
    args = ap.parse_args()

    out_root = ROOT / "outputs" / "viz_track"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []  # (box_name, clip_key, cov, area, off, disp, ok)
    for bname in args.boxes:
        boxp = ROOT / bname
        if not boxp.exists():
            print(f"[SKIP] 缺 {boxp}（先跑 detect_perframe.sbatch）")
            continue
        keys = pick_keys(boxp, args.n)
        print(f"==== {bname}: 抽查 {len(keys)} 个 clip（按 action 分层）====", flush=True)
        for k in keys:
            od = out_root / Path(bname).stem / k.replace("/", "__")
            r = subprocess.run(
                [sys.executable, str(VIZ), "--mode", "cache",
                 "--box", str(boxp), "--clip_key", k, "--out", str(od)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                last = (r.stderr or r.stdout).strip().splitlines()
                print(f"[FAIL] {bname} {k} -> {last[-1] if last else '?'}")
                rows.append((bname, k, "-", "-", "-", "-", False))
                continue
            text = r.stdout
            cov = grab("检测覆盖=", text)
            area = grab("box 面积占比: mean=", text)
            off = grab("中心偏移(离画面中心): mean=", text)
            disp = grab("相邻帧中心位移: mean=", text)
            ok = "✅" in text
            rows.append((bname, k, cov, area, off, disp, ok))
            print(f"[{'OK  ' if ok else 'WARN'}] {bname} key={k} | "
                  f"cov={cov} area={area} off={off} disp={disp}", flush=True)

    print("\n==== 汇总 ====", flush=True)
    for bname in args.boxes:
        rr = [r for r in rows if r[0] == bname]
        if not rr:
            continue
        okn = sum(1 for r in rr if r[6])
        print(f"{bname}: {okn}/{len(rr)} 通过 (判读 ✅)", flush=True)
    bad = [r for r in rows if not r[6]]
    if bad:
        print("需检查的 clip:")
        for b, k, *_ in bad:
            print(f"  {b}  {k}", flush=True)
    else:
        print("全部通过 ✅ —— 可提交 main_track_crop / thermal_track_crop", flush=True)


if __name__ == "__main__":
    main()
