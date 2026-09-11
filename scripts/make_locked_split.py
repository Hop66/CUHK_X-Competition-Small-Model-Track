#!/usr/bin/env python3
"""P0 新协议: 生成 locked env-balanced audit split (2026-09-11, 见 idea.md §4)。

协议:
  - 训练集 18 subjects = Env-A(user1-9, 9人) + Env-B(user16-24, 9人)。
  - LOCKED AUDIT SET: 永久保留 4 subjects (Env-A 2 + Env-B 2), 在模型家族锁定前绝不查看。
    用途: 唯一允许的"最终判据"。所有方法/阈值/epoch/α 选择都只能在 inner CV 做。
  - DEVELOPMENT: 其余 14 subject (Env-A 7 + Env-B 7) 做 environment-balanced Group CV
    (每折 A∪B 各留固定比例, outer subject 从不参与任何超参选择)。

用法: python scripts/make_locked_split.py --env A --env_b 2  # Env-A 锁 2 人
      python scripts/make_locked_split.py --locked_a 2 --locked_b 2
产出: outputs/locked_split.json {locked:[...], dev:[...], env:{A:[...],B:[...]}}
一次生成, 固定不改; 之后所有提交级候选都先用 locked 验证。
"""
import argparse
import json
from pathlib import Path
import random


def env(subject):
    n = int("".join(ch for ch in subject if ch.isdigit()))
    return "A" if n <= 15 else "B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked_a", type=int, default=2)
    ap.add_argument("--locked_b", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/locked_split.json")
    args = ap.parse_args()

    random.seed(args.seed)
    subs = ["user1","user2","user3","user4","user5","user6","user7","user8","user9",
            "user16","user17","user18","user19","user20","user21","user22","user23","user24"]
    A = sorted([s for s in subs if env(s) == "A"])
    B = sorted([s for s in subs if env(s) == "B"])
    assert len(A) >= args.locked_a and len(B) >= args.locked_b, "locked 超过环境人数"

    random.shuffle(A); random.shuffle(B)
    locked = A[:args.locked_a] + B[:args.locked_b]
    random.shuffle(locked)

    out = {
        "locked": locked,          # 永久审计集(最终判据用)
        "dev": [s for s in subs if s not in locked],  # 开发集(inner CV)
        "env": {"A": A, "B": B},
        "protocol": "locked env-balanced audit split; 4 subjects (Env-A 2 + Env-B 2) "
                    "permanent; all hyperparams selected on dev CV only",
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"locked audit set: {locked}")
    print(f"  Env-A dev: {[s for s in A if s not in locked]}")
    print(f"  Env-B dev: {[s for s in B if s not in locked]}")
    print(f"dev total: {len(out['dev'])} → {args.out}")


if __name__ == "__main__":
    main()
