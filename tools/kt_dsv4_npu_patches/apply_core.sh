#!/usr/bin/env bash
# Apply the CORE main_repo series (0001..0013, WIP dropped) onto a clean
# d7b5b49 checkout, with the fixes the dry-run proved necessary:
#   * use `git apply --3way --index` (the split fragments are NOT git-am-able)
#   * read SERIES.core (correct filenames; no WIP)
#   * strip `model/...` delete hunks from 0012 (SLIM never bundled those weights)
#   * strip the third_party/sglang gitlink hunk from 0013, then pin the
#     submodule to the final fork tip 68a0bce65 explicitly
#   * repair the concatenated .gitignore line
#
# Usage: apply_core.sh [REPO_ROOT]
# Pre-req: REPO_ROOT is a CLEAN worktree at d7b5b49 (no overlapping untracked files).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HERE/main_repo"
SERIES="$DIR/SERIES.core"
REPO="${1:-$(cd "$HERE/../.." && pwd)}"
SGLANG_TIP="68a0bce65fec9e66be6babbe342ae0c8f2224407"

[[ -f "$SERIES" ]] || { echo "missing $SERIES" >&2; exit 1; }
cd "$REPO"

# section filter: drop every `diff --git a/<path> b/...` block whose path matches $1
filter() { python3 - "$2" "$1" <<'PY'
import sys, re
drop_re = re.compile(sys.argv[1])
buf, out, drop = [], [], False
def flush():
    global buf
    if buf and not drop: out.extend(buf)
    buf = []
for line in open(sys.argv[2], errors='replace'):
    if line.startswith('diff --git '):
        flush()
        m = re.match(r'diff --git a/(.*?) b/', line)
        drop = bool(drop_re.search(m.group(1) if m else ''))
        buf = [line]
    else:
        buf.append(line)
flush()
sys.stdout.write(''.join(out))
PY
}

tmp="$(mktemp)"
while IFS= read -r p; do
  [[ -z "$p" || "$p" =~ ^# ]] && continue
  src="$DIR/$p"
  case "$p" in
    0012-*-p0[23].patch) filter "$src" '^model/' > "$tmp"; src="$tmp" ;;
    0013-*)              filter "$src" '^third_party/sglang$' > "$tmp"; src="$tmp" ;;
  esac
  echo "[apply] $p"
  git apply --3way --index --whitespace=nowarn "$src"
done < "$SERIES"
rm -f "$tmp"

# 0013's gitlink bump was stripped above; set the final sglang tip explicitly.
git update-index --add --cacheinfo "160000,${SGLANG_TIP},third_party/sglang"

# repair `.gitignore` (patch 0001 left no trailing newline, so a later append
# concatenated two rules onto one line).
if grep -q 'DeepSeek-R1-GGUF/model/w8a8/' .gitignore; then
  sed -i 's#model/DeepSeek-R1-GGUF/model/w8a8/#model/DeepSeek-R1-GGUF/\nmodel/w8a8/#' .gitignore
fi

echo "[done] core series applied. NEXT (manual):"
echo "  1. repoint .gitmodules sglang -> your PUBLIC fork (wenxuewuhd/sglang@dsv4_release),"
echo "     llama.cpp -> public; currently set to internal codehub URLs by patch 0007."
echo "  2. build the sglang fork: clone iforgetmyname/sglang@dsv4_release (298193eb3),"
echo "     git am tools/kt_dsv4_npu_patches/sglang/000{1..5}*.patch, push to your fork."
echo "  3. git submodule sync && git submodule update --init --recursive"
echo "  4. cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 python3 setup.py build_ext --inplace"
