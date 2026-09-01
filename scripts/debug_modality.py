#!/usr/bin/env python3
"""CUHK-X —— IMU / Radar 数据格式探索（写加载器前的必要一步）

列出目录结构 + 读样本文件，弄清格式（文件类型/行列/列含义），
之后才能写正确的数据加载器 + 模型。

用法: python scripts/debug_modality.py [--modality imu|radar]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/Multimodal/data/Training/HAR")
    ap.add_argument("--modality", default="imu", choices=["imu", "radar"])
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    # 显式目录名映射（IMU 全大写，Radar 首字母大写）
    MOD_NAMES = {"imu": "IMU", "radar": "Radar"}
    mod = MOD_NAMES.get(args.modality, args.modality.capitalize())
    mod_root = root / mod
    print(f"==== {mod} 根目录: {mod_root} 存在={mod_root.is_dir()} ====")

    if not mod_root.is_dir():
        print("❌ 目录不存在，检查模态名（IMU/Radar 大小写）")
        # 列出顶层模态目录
        top = sorted(d.name for d in root.iterdir() if d.is_dir())
        print(f"训练集顶层目录: {top}")
        return

    # 1) 遍历一个 action 的完整路径
    actions = sorted(d for d in mod_root.iterdir() if d.is_dir())
    print(f"action 目录数: {len(actions)}，样例: {[a.name for a in actions[:3]]}")
    a0 = actions[0]
    subjs = sorted(d for d in a0.iterdir() if d.is_dir())
    print(f"  {a0.name}/ 下 subject: {[s.name for s in subjs[:5]]}")
    if subjs:
        samples = sorted(d for d in subjs[0].iterdir() if d.is_dir())
        print(f"    {subjs[0].name}/ 下 sample: {[s.name for s in samples[:5]]}")
        if samples:
            files = sorted(samples[0].iterdir())
            print(f"      {samples[0].name}/ 下文件: {[f.name for f in files[:10]]} "
                  f"(共 {len(files)} 个)")
            # 2) 读文件看格式：跳过空 csv（Radar 有的 sample 无检测目标，csv 为空）
            if files:
                shown = False
                for f in files[:20]:
                    if f.suffix == ".csv":
                        try:
                            df = pd.read_csv(f, nrows=5)
                        except Exception:
                            continue
                        if len(df) == 0:
                            continue
                        print(f"\n==== 样本文件: {f.name} ({f.stat().st_size} 字节) ====")
                        print(f"csv 形状(前5行): {df.shape}，列名: {list(df.columns)}")
                        print(df.head().to_string())
                        shown = True
                        break
                    elif f.suffix == ".json":
                        import json
                        data = json.loads(f.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            print(f"\n==== 样本文件: {f.name} (json dict) ====")
                            print(f"json keys: {list(data.keys())}")
                            for k, v in data.items():
                                print(f"  {k}: type={type(v).__name__} "
                                      f"shape={np.asarray(v).shape if hasattr(v, '__len__') else 'N/A'}")
                        else:
                            arr = np.asarray(data)
                            print(f"\n==== 样本文件: {f.name} (json array shape {arr.shape}) ====")
                        shown = True
                        break
                if not shown:
                    print(f"\n==== 该 sample 前 20 个文件全为空或无法解析 ====")
                    # 打印文件原始内容（前 200 字节）帮助判断
                    if files:
                        print(f"首个文件内容: {files[0].read_bytes()[:200]!r}")
    print("\n==== 探索完成 ====")


if __name__ == "__main__":
    main()
