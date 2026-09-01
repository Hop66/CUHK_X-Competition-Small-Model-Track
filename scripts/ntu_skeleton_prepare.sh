#!/bin/bash
# ============================================================
# NTU RGB+D 骨架数据 解压 + 校验 + detect（数据已上传，无需下载）
#
# 输入（zip 或已解压目录，默认从 data/external/ntu/ 找）：
#   --ntu60 <zip|dir>   NTU60 (S001-S017)
#   --ntu120 <zip|dir>  NTU120 (S018-S032)
#   --skip-unzip        已解压，跳过解压直接 detect
#
# 流程：定位 → 字节校验(防下载损坏) → 解压 → detect 混合训练所需信息
# 产物：
#   data/external/ntu/ntu60/*.skeleton
#   data/external/ntu/ntu120/*.skeleton
#
# 用法:
#   bash scripts/ntu_skeleton_prepare.sh \
#     --ntu60  data/external/ntu/nturgbd_skeletons_s001_to_s017.zip \
#     --ntu120 data/external/ntu/nturgbd_skeletons_s018_to_s032.zip
#   bash scripts/ntu_skeleton_prepare.sh --skip-unzip   # 已解压直接 detect
# ============================================================
set -euo pipefail

BASE="${NTU_ROOT:-data/external/ntu}"
mkdir -p "$BASE"

NTU60="${NTU60:-}"
NTU120="${NTU120:-}"
SKIP_UNZIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ntu60) NTU60="$2"; shift 2 ;;
    --ntu120) NTU120="$2"; shift 2 ;;
    --skip-unzip) SKIP_UNZIP=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 官方 zip 字节数（校验防损坏）
SZ_NTU60=6181024200
SZ_NTU120=4780938300

echo "==== NTU 骨架准备 (base=$BASE, skip_unzip=$SKIP_UNZIP) ===="

# ---------- 定位 ----------
if [ -z "$NTU60" ] && [ -f "$BASE/nturgbd_skeletons_s001_to_s017.zip" ]; then
  NTU60="$BASE/nturgbd_skeletons_s001_to_s017.zip"
fi
if [ -z "$NTU120" ] && [ -f "$BASE/nturgbd_skeletons_s018_to_s032.zip" ]; then
  NTU120="$BASE/nturgbd_skeletons_s018_to_s032.zip"
fi
# 已解压目录优先于 zip（跳过重复解压）
if [ -d "$BASE/ntu60" ] && find "$BASE/ntu60" -name '*.skeleton' | grep -q .; then
  NTU60="$BASE/ntu60"; SKIP_UNZIP=1
fi
if [ -d "$BASE/ntu120" ] && find "$BASE/ntu120" -name '*.skeleton' | grep -q .; then
  NTU120="$BASE/ntu120"; SKIP_UNZIP=1
fi
[ -z "$NTU60" ] && [ -z "$NTU120" ] && { echo "❌ 未找到 zip/目录，请用 --ntu60/--ntu120 指定，或放到 $BASE/"; exit 1; }

# ---------- 解压 ----------
unzip_one() {
  local src="$1" dest="$2" expect="$3"
  if [ -d "$src" ]; then
    echo "  $dest: 已是目录（跳过解压）"; return 0
  fi
  if [ ! -f "$src" ]; then echo "  ❌ 缺 $src"; return 1; fi
  local sz; sz=$(stat -c%s "$src")
  echo "  $src : $sz bytes (期望 $expect)"
  if [ "$sz" != "$expect" ]; then
    echo "  ❌ 大小不一致（下载损坏？请重传）"; return 1
  fi
  echo "  ✅ 大小一致，解压到 $dest ..."
  mkdir -p "$dest"
  unzip -q -o "$src" -d "$dest"
  echo "  ✅ 解压完成: $(find "$dest" -name '*.skeleton' | wc -l) 个 .skeleton"
}

if [ "$SKIP_UNZIP" = "1" ]; then
  echo "######## [--skip-unzip] 跳过解压 ########"
else
  echo "######## 解压 + 大小校验 ########"
  if [ -n "$NTU60" ]; then unzip_one "$NTU60" "$BASE/ntu60" "$SZ_NTU60" || exit 1; fi
  if [ -n "$NTU120" ]; then unzip_one "$NTU120" "$BASE/ntu120" "$SZ_NTU120" || exit 1; fi
fi

# ---------- 结构归一化：不同镜像解压层级不同 ----------
# NTU60 镜像解压出 ntu60/nturgb+d_skeletons/*.skeleton（中间层级），NTU120 无中间层级。
# 统一把子目录里的 .skeleton 移到 ntu60/、ntu120/ 顶层，清理空目录。
echo "######## 结构归一化（子目录 .skeleton → 顶层）########"
for sub in ntu60 ntu120; do
  if [ -d "$BASE/$sub" ]; then
    find "$BASE/$sub" -mindepth 2 -name '*.skeleton' -exec mv -f {} "$BASE/$sub/" \;
    find "$BASE/$sub" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    echo "  $sub: $(find "$BASE/$sub" -maxdepth 1 -name '*.skeleton' | wc -l) 个 .skeleton（顶层）"
  fi
done

# ---------- detect：验证 + 混合训练所需信息 ----------
echo "######## detect：格式验证 + 混合训练信息 ########"
if [ -d "$BASE/ntu60" ]; then
  python scripts/ntu_skeleton_verify.py --root "$BASE/ntu60" --class-stats
fi
if [ -d "$BASE/ntu120" ]; then
  python scripts/ntu_skeleton_verify.py --root "$BASE/ntu120" --class-stats
fi

echo "==== NTU 骨架准备完成（数据就绪，下一步：25→17 关节映射 + 类别映射）===="
