#!/usr/bin/env python3
"""验证 NTU RGB+D .skeleton 文件格式（detect：确认下载数据真实正确）。

NTU .skeleton 文本格式（官方 matlab 读取）：
  line1            : body 数量
  每个 body:
    1 行 body 信息 : bodyID, clippedEdges×16, centerX, centerY, range, numJoints
    每关节 1 行     : x, y, z, depthX, depthY, colorX, colorY, orient_w,x,y,z, tracking
                     （12 字段；x,y,z 为 3D 米制坐标，Kinect v2 约 [-5,5]）

用法:
    python scripts/ntu_skeleton_verify.py --root data/external/ntu/ntu60 --n 5
    python scripts/ntu_skeleton_verify.py --root data/external/ntu/ntu60 --all-stats
"""

import argparse
import random
import re
import sys
from pathlib import Path

# Windows GBK 控制台兜底（服务器 Linux UTF-8 不受影响）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FNAME_RE = re.compile(r"^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})(?:\.skeleton)?$")


def parse_skeleton(path: Path):
    """解析 NTU .skeleton 文件（token 流式，fscanf 语义，跨行分布）。

    真实格式（本地实测 S001C001P001R001A001.skeleton，32137 token）：
      token0         : 总帧数（如 103）
      每帧:
        nbody
        每 body: 11 token = bodyID, 6×clippedEdges, centerX, centerY, range, numJoints
        每关节:  12 token = x,y,z, depthX,depthY, colorX,colorY, orient(4), tracking
      校验: 1 + 103*(1 + 11 + 25*12) = 32137 ✓

    返回: (frames, err)；frames = [ [(num_joints, joints[N,12]), ...], ... ] per frame。
    """
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    frames = []
    i = 0

    def take(n):
        nonlocal i
        if i + n > len(tokens):
            raise IndexError(f"token 不足 @{i}，需 {n}，剩 {len(tokens) - i}")
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    try:
        nframes = int(take(1)[0])
        for _ in range(nframes):
            nbody = int(take(1)[0])
            fbody = []
            for _ in range(nbody):
                info = take(11)            # body 信息（11 token）
                num_joints = int(info[-1])
                joints = [take(12) for _ in range(num_joints)]
                fbody.append((num_joints, joints))
            frames.append(fbody)
        if i != len(tokens):
            return None, f"token 未消耗完：@{i}/{len(tokens)}（body 信息长度假设错？）"
    except (IndexError, ValueError) as e:
        return None, f"token@{i}: {e}"
    return frames, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--n", type=int, default=3, help="抽样验证文件数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-stats", action="store_true",
                    help="统计全部样本的类别/被试分布（混合训练选类用，快）")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.skeleton"))
    print(f"root={root}  .skeleton 文件数: {len(files)}", flush=True)
    if not files:
        print("❌ 没有找到 .skeleton 文件（解压失败？路径不对？）", flush=True)
        sys.exit(1)

    # 文件名格式检查
    bad_name = [f.name for f in files[:200] if not FNAME_RE.match(f.name)]
    if bad_name:
        print(f"⚠️ 非标准命名（前 5）: {bad_name[:5]}", flush=True)
    else:
        print("✅ 文件名格式符合 SsssCcccPpppRrrrAaaa", flush=True)

    # ---- class-stats：全部样本的类别/被试分布（混合训练选类用）----
    if args.class_stats:
        from collections import Counter
        subjects = Counter()
        classes = Counter()
        for f in files:
            m = FNAME_RE.match(f.name)
            if not m:
                continue
            _, _, p, _, a = m.groups()
            subjects[p] += 1
            classes[a] += 1
        n_sub = len(subjects)
        n_cls = len(classes)
        print(f"\n==== class-stats（{len(files)} 样本）====", flush=True)
        print(f"被试数: {n_sub}（跨被试多样性，NTU60 应为 40）", flush=True)
        print(f"类数: {n_cls}（NTU60 应为 60）", flush=True)
        # A001-A040 子集覆盖（与我们 40 类重叠区）
        sub40 = {a: c for a, c in classes.items() if int(a) <= 40}
        print(f"A001-A040 覆盖: {len(sub40)} 类 / {sum(sub40.values())} 样本", flush=True)
        cnts = sorted(sub40.values(), reverse=True)
        if len(cnts) >= 2:
            print(f"A001-A040 长尾比(最多/最少): {cnts[0]}/{cnts[-1]} = {cnts[0]/max(cnts[-1],1):.1f}x", flush=True)
        print(f"A001-A040 每类样本数（前 15）: {sorted(sub40.items())[:15]}", flush=True)
        # 高频类别
        print(f"高频类（全 60 类，前 10）: {classes.most_common(10)}", flush=True)

    # 抽样验证内容
    rng = random.Random(args.seed)
    sample = rng.sample(files, min(args.n, len(files)))
    all_num_joints = set()
    coord_ranges = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
    frame_counts = []
    errors = 0
    for f in sample:
        frames, err = parse_skeleton(f)
        if err:
            print(f"  ❌ {f.name}: {err}", flush=True)
            errors += 1
            continue
        nf = len(frames)
        frame_counts.append(nf)
        jc = {nj for fr in frames for nj, _ in fr}
        all_num_joints.update(jc)
        # 坐标范围（x,y,z 前 3 列）
        for fr in frames:
            for _, joints in fr:
                for j in joints:
                    if len(j) >= 3:
                        coord_ranges[0] = min(coord_ranges[0], j[0]); coord_ranges[1] = max(coord_ranges[1], j[0])
                        coord_ranges[2] = min(coord_ranges[2], j[1]); coord_ranges[3] = max(coord_ranges[3], j[1])
                        coord_ranges[4] = min(coord_ranges[4], j[2]); coord_ranges[5] = max(coord_ranges[5], j[2])
        print(f"  ✓ {f.name}: 帧数={nf} 关节数={sorted(jc)}", flush=True)

    print(f"\n==== 汇总 ====", flush=True)
    print(f"抽样 {len(sample)} 文件，解析错误 {errors}，帧数={frame_counts}", flush=True)
    print(f"关节数集合: {sorted(all_num_joints)}（NTU 官方应为 {{25}}）", flush=True)
    print(f"3D 坐标范围 x:[{coord_ranges[0]:.2f},{coord_ranges[1]:.2f}] "
          f"y:[{coord_ranges[2]:.2f},{coord_ranges[3]:.2f}] "
          f"z:[{coord_ranges[4]:.2f},{coord_ranges[5]:.2f}]（Kinect 米制通常 [-5,5]）", flush=True)

    # 判定
    if errors == 0 and all_num_joints <= {25} and len(all_num_joints) > 0:
        print("✅ 格式验证通过：合法 NTU .skeleton（25 关节，多帧序列）", flush=True)
    else:
        print("⚠️ 有异常，请检查（关节数非 25 或解析失败）", flush=True)


if __name__ == "__main__":
    main()
