#!/usr/bin/env python3
"""auto_drive.py —— CUHK-X 自驱实验链（9h 无人值守版）

设计（路线3 语义）：
  - 常驻轮询（不依赖 watchdog）：每 POLL 秒查 squeue + 读 logs/<prefix>-<jobid>.out
  - 对每个已提交 job：探测 'fold0 best=<acc>' / '== DONE ==' / 'Traceback'
    -> 按 REGISTRY 判定阈值 => verdict(positive/negative/error)
  - 依据 verdict 自动排队下一个已验证实验（dry 预置链）
  - 把完整状态写 drive_state.json（原子写），Agent 压缩后读它即可无缝恢复
  - 边界：绝不自动提交 LB；只产出候选 CSV 供用户批准

用法：
  nohup python scripts/auto_drive.py --poll 60 --max-minutes 540 > logs/auto_drive.log 2>&1 &
    （9h = 540min；到点自动收尾写 FINAL_REPORT）
  python scripts/auto_drive.py --once          # 单次收一轮（手动）
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
LOGS = WS / "logs"
STATE = WS / "drive_state.json"
REPORT = WS / "logs" / "auto_drive_FINAL.txt"

DONE_RE = re.compile(r"== DONE ==")
# train_step1 fold 路径: '== fold 0 best val = 0.6631 -> ...' ；gate/aux 路径: 'fold0 best=0.6566'
FOLD_RE = re.compile(r"fold\s*0\s*best(?:\s*val)?\s*=\s*([0-9.]+)")
FOLD_FULL_RE = re.compile(r"==== FULL DONE:")
TRACE_RE = re.compile(r"Traceback|Error|error:")
JOBID_RE = re.compile(r"Submitted batch job (\d+)")

# ---------- 实验注册表 ----------
# name: {sbatch, active(已提交), threshold, next_pass, next_fail, note}
# next_* 填注册表 name；None=链终止
REGISTRY = {
    "hardpair_wt_fold0": {
        "sbatch": "scripts/hardpair_wt.sbatch",      # 59057 已提交
        "prefix": "cuhkx-hpwt",
        "threshold": 0.6695,                          # main16f 基线
        "next_pass": "hardpair_wt_full",
        "next_fail": "skel_gate_candidate",
        "note": "难对重权 fold0：≥0.6695 → 训 full；< → 骨架 gate 候选",
    },
    "hardpair_wt_full": {
        "sbatch": "scripts/hardpair_wt_full.sbatch",  # 待 auto 提交
        "prefix": "cuhkx-hpwtfull",
        "next_pass": "hpwt_test_chain",
        "next_fail": None,
        "note": "难对重权 full(seed42) → 打包 int5 → test 链",
    },
    "hpwt_test_chain": {
        "sbatch": "scripts/hpwt_test_chain.sbatch",   # 待 auto 提交（打包+推理）
        "prefix": "cuhkx-hpwtchain",
        "next_pass": None,
        "next_fail": None,
        "note": "main_hpwt+th_nf32full flip → sub_hpwt_chain.csv 候选",
    },
    "skel_gate_candidate": {
        "sbatch": None,  # 无 GPU；候选已产出 sub_hp_gate_skel.csv
        "prefix": "none",
        "next_pass": None,
        "next_fail": None,
        "note": "骨架-only gate 已出候选 CSV（fold +0.94 稳，翻3个），等用户 LB",
        "done": True,
    },
}


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "auto_drive.log", "a") as f:
        f.write(line + "\n")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"jobs": {}, "chain": {}, "candidates": [], "report": []}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    tmp.replace(STATE)


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WS)
    return p.stdout.strip(), p.returncode


def squeue_ids():
    out, _ = run("squeue -u $USER -h -o %i 2>/dev/null")
    return {int(x) for x in out.split() if x.strip().isdigit()}


def tail_of(prefix, jobid, n=20):
    f = LOGS / f"{prefix}-{jobid}.out"
    if not f.exists():
        return ""
    return f.read_text(errors="ignore")[-4000:]


def detect_jobid_from_out(prefix):
    """尚未 statcked 的 job：从新建的 out 或 sbatch 提交记录找。简化：主动记录。"""
    return None


def poll_job(st, jobid, prefix):
    tail = tail_of(prefix, jobid)
    has_done = bool(DONE_RE.search(tail))
    m = FOLD_RE.search(tail)
    acc = float(m.group(1)) if m else None
    full_done = bool(FOLD_FULL_RE.search(tail))
    trace = bool(TRACE_RE.search(tail))
    return {"done": has_done, "acc": acc, "full_done": full_done, "trace": trace, "tail_ok": bool(tail)}


def submit(sbatch):
    if sbatch is None:
        return None, "no-sbatch"
    sb = WS / sbatch
    out, rc = run(f"sbatch {sb} 2>&1")
    m = JOBID_RE.search(out)
    if m:
        return int(m.group(1)), out.strip()
    return None, out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-minutes", type=int, default=540)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    # 已知已提交 jobid -> name（首启动先手动登记 59057）
    known = {"hardpair_wt_fold0": [59057]}
    st = load_state()
    for name, ids in known.items():
        if name not in st["chain"]:
            st["chain"][name] = {"jobids": [int(i) for i in ids],
                                 "verdict": "running", "acc": None}
    # 预置候选（无 GPU 项，直接标记 done 供 FINAL 报告）
    for name, cfg in REGISTRY.items():
        if cfg.get("done") and name not in st["chain"]:
            st["chain"][name] = {"jobids": [], "verdict": "done", "acc": None}
            st["candidates"].append((name, REGISTRY[name]["note"]))
    save_state(st)

    t0 = time.time()
    deadline = t0 + args.max_minutes * 60
    log(f"auto_drive 启动: poll={args.poll}s max={args.max_minutes}min (deadline "
        f"{time.strftime('%m-%d %H:%M')})")
    log("已知 job: " + json.dumps(known))

    while time.time() < deadline:
        active = squeue_ids()
        progress = False
        for name, cfg in REGISTRY.items():
            ent = st["chain"].get(name)
            if not ent or ent.get("verdict") != "running":
                continue
            for jobid in ent["jobids"]:
                if jobid in active:
                    continue  # 仍在队列
                info = poll_job(st, jobid, cfg.get("prefix", name.split("_")[0]))
                # 若 done -> 判定
                if info["done"] and info["acc"] is not None:
                    thr = cfg["threshold"]
                    v = "positive" if info["acc"] >= thr else "negative"
                    ent["verdict"] = v
                    ent["acc"] = info["acc"]
                    log(f"✅ {name} (job {jobid}) fold0={info['acc']:.4f} vs "
                        f"{thr} → {v}")
                    st["report"].append(f"{name}: fold0={info['acc']:.4f} → {v} "
                                        f"(threshold {thr})")
                    # 排队下一实验
                    nxt = cfg["next_pass"] if v == "positive" else cfg["next_fail"]
                    if nxt and nxt not in st["chain"]:
                        nj = submit(REGISTRY[nxt]["sbatch"])
                        st["chain"][nxt] = {"jobids": [nj[0]] if nj[0] else [],
                                            "verdict": "queued", "acc": None}
                        log(f"  ↳ 排队 {nxt} -> job {nj[0] if nj[0] else nj[1]}")
                    elif nxt:
                        log(f"  ↳ {nxt} 已在链中(不重复排队)")
                    progress = True
                elif info["trace"]:
                    ent["verdict"] = "error"
                    log(f"❌ {name} (job {jobid}) Traceback → 标记 error, 关链")
                    st["report"].append(f"{name}: ERROR (traceback)")
                    progress = True
                elif info["done"] and info["acc"] is None:
                    # DONE 但没抓到 fold0（比如 full 训练无 fold）→ 按完成处理、走 pass
                    ent["verdict"] = "done"
                    log(f"✔ {name} (job {jobid}) DONE (无 fold0 数字) → 按完成走 pass")
                    nxt = cfg["next_pass"]
                    if nxt and nxt not in st["chain"]:
                        nj = submit(REGISTRY[nxt]["sbatch"])
                        st["chain"][nxt] = {"jobids": [nj[0]] if nj[0] else [],
                                            "verdict": "queued", "acc": None}
                        log(f"  ↳ 排队 {nxt} -> job {nj[0] if nj[0] else nj[1]}")
                    progress = True
        os.system("sleep 1")  # 让 .out 文件系统落盘
        if args.once:
            break
        if progress:
            save_state(st)
        # 静默等待直到有 job 结束或到点
        if active:
            time.sleep(args.poll)
        else:
            # 队列空：仍可能有 queued 链项（未提交成功）。检查全链终态
            unfinished = [n for n, c in st["chain"].items()
                          if c.get("verdict") in ("running", "queued")]
            if not unfinished:
                log("全链结束（无 running/queued）—— 收尾")
                break
            log(f"队列空但链项未终态: {unfinished}（等 submit 生效或 job 启动）")
            time.sleep(args.poll)

    # ---- 收尾：若还在跑则等待（最多 max-minutes 已保证） ----
    save_state(st)
    # FINAL REPORT
    cand_lines = [f"- {n}: {note}" for n, note in st["candidates"]]
    lines = ["=== auto_drive FINAL ===", f"end {time.strftime('%m-%d %H:%M')}",
             "--- verdicts ---"] + st["report"] + \
            ["--- candidates (待用户 LB 批准) ---"] + cand_lines
    REPORT.write_text("\n".join(lines))
    log(f"FINAL 报告 -> {REPORT}")
    log("auto_drive 结束")


if __name__ == "__main__":
    main()
