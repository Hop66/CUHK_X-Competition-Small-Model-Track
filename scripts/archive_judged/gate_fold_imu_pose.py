#!/usr/bin/env python3
"""方向A: IMU 姿态积分特征进难对判别器 (无泄漏 fold 验证)。

在已验证 LB 正票"骨架难对门控"(sub_hp_gate_skel) 基础上扩展：
  难对选择器 = main+th predict 落难对 + margin<tau + conf∈[cl,ch]（与 gate 相同）
  判别器升级 = GBDT( feats_v3[13] ⊕ imu_pose_agg[60] )
     - feats_v3: 骨架手工特征(已证有效, 只对难对二分类)
     - imu_pose_agg: 从 imu_pose_integ() 的 [T,30] 提取姿态积分聚合描述子
        (每设备 accumulate roll/pitch/yaw 位移的 mean/std/max + 变化率 → 域相对量)
  触发后: GBDT 在难对内二分类换头 (约束难对内, 不跑偏)

用法:
  python scripts/gate_fold_imu_pose.py            # 无泄漏 fold 扫配方
  python scripts/gate_fold_imu_pose.py --pairs_keep "13-12,22-21"  # 只留好对
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.dataset import build_train_index
from src.split import split_by_subject
from src.skeleton_dataset import build_skeleton_index
from twin_v3 import feats_v3, load_kp

HARD = [(13, 12), (22, 21), (8, 10), (18, 17), (26, 24)]
HPID = {x for p in HARD for x in p}

SKEL_CACHE = Path("outputs/skel_v3_only_cache.pkl")
IMU_CACHE = Path("outputs/imu_pose_feats_bench.pkl")   # {keys[], feats[N,128,30], labels[]}
IMU_AGG_CACHE = Path("outputs/imu_pose_agg_bench.pkl")


def sf(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def imu_pose_agg(seq: np.ndarray) -> np.ndarray:
    """从 [T,30] (5dev × [gyro3, integ_pose3]) 提聚合描述子 → 60d。

    每设备 12d = [gyro_abs_mean, gyro_abs_std, gyro_abs_max,
                  pose_disp(末-初绝对值 mean over 3), pose_abs_mean, pose_abs_std,
                  pose_abs_max, pose_abs_max_per_axis(3), pose_energy(Σ|d/dt|)]
    12d×5dev = 60d。姿态积分量是域相对量（对抗 gyro 幅值缩放）。
    """
    out = []
    T, D = seq.shape
    for di in range(5):
        seg = seq[:, di * 6:(di + 1) * 6]          # [T,6] gyro3+pose3
        gyro = seg[:, :3]
        pose = seg[:, 3:]
        ga = np.abs(gyro)
        pa = np.abs(pose)
        dpose = np.abs(np.diff(pose, axis=0))       # 姿态变化率
        feat = np.array([
            ga.mean(), ga.std(), ga.max(),
            np.abs(pose[-1] - pose[0]).mean(),
            pa.mean(), pa.std(), pa.max(),
            np.abs(pose[-1] - pose[0])[0], np.abs(pose[-1] - pose[0])[1], np.abs(pose[-1] - pose[0])[2],
            dpose.mean(), dpose.max(),
        ], np.float32)
        out.append(feat)
    return np.concatenate(out)


def prepare_imu_agg():
    """用 imu_pose_agg cache, key = action/subject/sample, 缺IMU→zeros。"""
    if IMU_AGG_CACHE.exists():
        return pickle.load(open(IMU_AGG_CACHE, "rb"))
    d = pickle.load(open(IMU_CACHE, "rb"))
    keys, feats = d["keys"], d["feats"]
    agg = {}
    for i, k in enumerate(keys):
        agg[k] = imu_pose_agg(feats[i].astype(np.float64))
    pickle.dump(agg, open(IMU_AGG_CACHE, "wb"))
    print(f"imu_pose_agg cache n={len(agg)} dim={len(next(iter(agg.values())))}", flush=True)
    return agg


def main():
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument("--pairs", type=str, default="")
    ap.add_argument("--pairs_keep", type=str, default="",
                    help="只对指定对启用 gate, 逗号分隔 a-b,c-d; 空=全部 HARD")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--conf_lo", type=float, default=0.5)
    ap.add_argument("--conf_hi", type=float, default=0.85)
    ap.add_argument("--gb_conf", type=float, default=0.6,
                    help="GBDT 判别器置信(两类的max prob)>此值才翻转")
    args = ap.parse_args()

    global HARD, HPID
    if args.pairs.strip():
        for token in args.pairs.split(","):
            try:
                a, b = (int(x) for x in token.strip().split("-"))
                if (a, b) not in HARD and (b, a) not in HARD:
                    HARD = HARD + [(a, b)]
            except Exception:
                pass
    # 只留好对
    if args.pairs_keep.strip():
        keep = set()
        for token in args.pairs_keep.split(","):
            a, b = (int(x) for x in token.strip().split("-"))
            keep.add((a, b))
        HARD = [p for p in HARD if p in keep or (p[1], p[0]) in keep]
    HPID = {x for p in HARD for x in p}
    print(f"HARD (final): {HARD}", flush=True)

    root = Path("data/Training/HAR")
    sk = [c for c in build_skeleton_index(root) if c.action_id in HPID]
    folds = split_by_subject(sk, 3)
    OOF = Path("outputs/oof")
    MK = {f: pickle.load(open(OOF / "main_oof.pkl", "rb"))[f] for f in range(3)}
    TK = {f: pickle.load(open(OOF / "thermal_oof.pkl", "rb"))[f] for f in range(3)}

    # 骨架 v3 特征 (仅难对 clips)
    if SKEL_CACHE.exists():
        SKEL = pickle.load(open(SKEL_CACHE, "rb"))
    else:
        SKEL = {}
        for c in sk:
            k = load_kp(c)
            if k is None:
                continue
            SKEL[f"{c.action_id}/{c.subject}/{c.sample}"] = feats_v3(k)
        pickle.dump(SKEL, open(SKEL_CACHE, "wb"))
    print(f"skel_v3 n={len(SKEL)}", flush=True)

    # IMU 姿态聚合
    IMU = prepare_imu_agg()

    # 3 折无泄漏评估
    results = []
    for f in range(3):
        tr_idx, va_idx = folds[f]
        clfs = {}
        for a, b in HARD:
            X, y = [], []
            for c in [sk[i] for i in tr_idx]:
                k = f"{c.action_id}/{c.subject}/{c.sample}"
                if k in SKEL and k in IMU:
                    X.append(np.concatenate([SKEL[k], IMU[k]]))
                    y.append(1 if int(k.split('/')[0]) == a else 0)
            if len(set(y)) < 2 or len(y) < 12:
                continue
            clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
            clf.fit(np.array(X, np.float32), y)
            clfs[(a, b)] = clf
        # ---------------- val 折 ----------------
        pre_b, marg, conf, yt, inhp, gb_side, gb_conf = [], [], [], [], [], [], []
        for c in [sk[i] for i in va_idx]:
            key = f"{c.action_id}/{c.subject}/{c.sample}"
            if key not in MK[f] or key not in TK[f] or key not in SKEL or key not in IMU:
                continue
            soft = sf(MK[f][key]) + sf(TK[f][key]); soft = soft / soft.sum()
            pred = int(np.argmax(soft)); srt = np.sort(soft)[::-1]
            marg.append(float(srt[0] - srt[1])); conf.append(float(srt[0]))
            pre_b.append(pred); yt.append(int(c.action_id)); inhp.append(pred in HPID)
            gb, gb_c = pred, 0.0
            if pred in HPID:
                for a, b in HARD:
                    if pred not in (a, b):
                        continue
                    clf = clfs.get((a, b))
                    if clf is None:
                        break
                    Xfeat = np.concatenate([SKEL[key], IMU[key]])[None].astype(np.float32)
                    pab = clf.predict_proba(Xfeat)[0]
                    gb_c = float(pab.max())
                    gb = (a if pab[list(clf.classes_).index(1)] >= pab[list(clf.classes_).index(0)] else b)
                    break
            gb_side.append(gb); gb_conf.append(gb_c)
        pre_b = np.array(pre_b); marg = np.array(marg); conf = np.array(conf)
        yt = np.array(yt); inhp = np.array(inhp); gb_side = np.array(gb_side); gb_conf = np.array(gb_conf)
        base = (pre_b == yt).mean()
        hp_mask = inhp
        base_hp = (pre_b[hp_mask] == yt[hp_mask]).mean() if hp_mask.any() else 0.0
        # 触发
        gate = inhp & (marg < args.tau) & (conf >= args.conf_lo) & (conf <= args.conf_hi)
        pred_g = np.where(gate, gb_side, pre_b)
        d_all = (pred_g == yt).mean() - base
        d_hp = (pred_g[hp_mask] == yt[hp_mask]).mean() - base_hp if hp_mask.any() else 0.0
        n_tr = int(gate.sum())
        # GBDT 置信门控 (只翻 confident)
        gate_c = gate & (gb_conf >= args.gb_conf)
        pred_gc = np.where(gate_c, gb_side, pre_b)
        d_all_c = (pred_gc == yt).mean() - base
        results.append((d_all, d_all_c, d_hp, n_tr, base, base_hp))
        print(f"  fold{f}: base={base:.4f}(hp {base_hp:.4f}) Δall={d_all:+.4f} Δall+gbfilter={d_all_c:+.4f} 触发={n_tr}", flush=True)
    r = np.array(results)
    print(f"\n== 3折汇总: Δall={np.mean(r[:,0]):+.4f}±{np.std(r[:,0]):.4f}  Δall+gbConf={np.mean(r[:,1]):+.4f}±{np.std(r[:,1]):.4f}")


if __name__ == "__main__":
    main()
