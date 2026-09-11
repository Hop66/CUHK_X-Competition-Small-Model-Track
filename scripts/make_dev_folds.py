#!/usr/bin/env python3
"""P1 步骤1: 环境平衡 dev 内折 (基于 locked_split.json)。

规则(idea.md §4):
  - locked 4 人 (Env-A 2 + Env-B 2) 永久锁定, 绝不参与训练/选参/调阈值。
  - dev 14 人做环境平衡 Group CV → 3 折 (每折 Env-A 3人 + Env-B 3人, 共6人 val)。
  - 超参只在 dev 折上选; locked 只做一次最终判据。
产出: outputs/dev_folds.json = {"locked": [...], "dev_folds": [{"tr":[...],"va":[...]}, ...]}
"""
import json
import random
from pathlib import Path


def env(s):
    n = int(''.join(ch for ch in s if ch.isdigit()))
    return 'A' if n <= 15 else 'B'


def main():
    sp = json.loads(Path("outputs/locked_split.json").read_text())
    locked = set(sp["locked"])
    env_a = sorted(s for s in sp["dev"] if env(s) == "A")
    env_b = sorted(s for s in sp["dev"] if env(s) == "B")
    rng = random.Random(42)
    rng.shuffle(env_a); rng.shuffle(env_b)

    folds = []
    # 7 人 → 份 [3,2,2]; 8 人 → 份 [3,3,2] （每折 A/B 各≥2, 尽量均）
    def split_chunks(lst, n=3):
        base = len(lst) // n
        rem = len(lst) % n
        sizes = [base + (1 if i < rem else 0) for i in range(n)]
        out, p = [], 0
        for s in sizes:
            out.append(lst[p:p + s]); p += s
        return out
    for i, (va_a, va_b) in enumerate(zip(split_chunks(env_a), split_chunks(env_b))):
        va = va_a + va_b
        tr = [s for s in sp["dev"] if s not in va]
        folds.append({"tr": tr, "va": va})

    out = {"locked": sorted(locked), "dev": sp["dev"], "dev_folds": folds}
    Path("outputs/dev_folds.json").write_text(json.dumps(out, indent=2))
    print("locked:", sorted(locked))
    for i, f in enumerate(folds):
        print(f"fold{i}: tr={len(f['tr'])} (A:{sum(1 for s in f['tr'] if env(s)=='A')} "
              f"B:{sum(1 for s in f['tr'] if env(s)=='B')}) va={f['va']}")


if __name__ == "__main__":
    main()
