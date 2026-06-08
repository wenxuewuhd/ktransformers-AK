#!/usr/bin/env bash
# Replay the CORE series (0001..0013, WIP dropped) as 13 separate commits that
# restore the original logical granularity. Each commit reuses the original
# author / date / message taken from that group's first fragment header.
#
# Same content fixes as apply_core.sh:
#   * git apply --3way --index (fragments are not git-am-able)
#   * strip model/* dead deletes from 0012 (SLIM never bundled those weights)
#   * strip the sglang gitlink hunk from 0013, then pin sglang to 68a0bce65
#   * fold the .gitignore line-merge repair into the 0012 commit
#
# Usage: apply_core_commits.sh [REPO_ROOT]
# Pre-req: REPO_ROOT is a CLEAN worktree at d7b5b49.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HERE/main_repo"
SERIES="$DIR/SERIES.core"
REPO="${1:-$(cd "$HERE/../.." && pwd)}"
SGLANG_TIP="68a0bce65fec9e66be6babbe342ae0c8f2224407"
[[ -f "$SERIES" ]] || { echo "missing $SERIES" >&2; exit 1; }
cd "$REPO"

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

# read SERIES.core into an array, grouped by the leading 4-digit id
mapfile -t LINES < <(grep -vE '^\s*#|^\s*$' "$SERIES")
tmp="$(mktemp)"; msg="$(mktemp)"; info="$(mktemp)"

groups="$(printf '%s\n' "${LINES[@]}" | sed -E 's/^([0-9]{4})-.*/\1/' | uniq)"
for g in $groups; do
  # fragments of this group, in order
  frags=()
  for p in "${LINES[@]}"; do [[ "$p" == ${g}-* ]] && frags+=("$p"); done
  echo "=== group $g (${#frags[@]} fragment(s)) ==="
  for p in "${frags[@]}"; do
    src="$DIR/$p"
    case "$p" in
      0012-*-p0[23].patch) filter "$src" '^model/'              > "$tmp"; src="$tmp" ;;
      0013-*)              filter "$src" '^third_party/sglang$' > "$tmp"; src="$tmp" ;;
    esac
    echo "  [apply] $p"
    git apply --3way --index --whitespace=nowarn "$src"
  done

  # per-group post steps
  if [[ "$g" == 0012 ]]; then
    sed -i 's#model/DeepSeek-R1-GGUF/model/w8a8/#model/DeepSeek-R1-GGUF/\nmodel/w8a8/#' .gitignore || true
    git add .gitignore
  fi
  if [[ "$g" == 0013 ]]; then
    git update-index --add --cacheinfo "160000,${SGLANG_TIP},third_party/sglang"
  fi

  # commit metadata via `git mailinfo` (decodes RFC2047/MIME subjects and strips
  # the [PATCH] prefix). Scan the group's fragments: take author/date from the
  # first with a real From-header, and the subject+body from the first whose
  # subject is real (the splitter drops 0002's header and writes the rest as
  # "continuation (part N/M)"; we strip the part suffix and fall back to the
  # filename stem when no real subject survives).
  AN=""; AE=""; AD=""; SUBJ=""; : > "$msg"
  for fr in "${frags[@]}"; do
    mbody="$(mktemp)"
    mi="$(git mailinfo "$mbody" /dev/null < "$DIR/$fr" 2>/dev/null || true)"
    a="$(printf '%s\n' "$mi" | sed -n 's/^Author: //p')"
    e="$(printf '%s\n' "$mi" | sed -n 's/^Email: //p')"
    d="$(printf '%s\n' "$mi" | sed -n 's/^Date: //p')"
    s="$(printf '%s\n' "$mi" | sed -n 's/^Subject: //p')"
    [[ -z "$AN" && -n "$a" ]] && { AN="$a"; AE="$e"; AD="$d"; }
    if [[ -z "$SUBJ" && -n "$s" && "$s" != continuation* ]]; then
      SUBJ="$(printf '%s' "$s" | sed -E 's/ \(part [0-9]+\/[0-9]+\)$//')"
      cp "$mbody" "$msg"
    fi
    rm -f "$mbody"
  done
  : "${AN:=Yuan Yuan}" "${AE:=soc.yuan@gmail.com}"
  if [[ -z "$SUBJ" ]]; then
    stem="${frags[0]%.patch}"; stem="${stem#${g}-}"; stem="${stem%-p[0-9][0-9]}"
    SUBJ="${g}: ${stem//-/ }"
  fi
  { echo "$SUBJ"; [[ -s "$msg" ]] && { echo; cat "$msg"; }; } > "$tmp.commitmsg"
  if [[ -n "$AD" ]]; then
    GIT_AUTHOR_NAME="$AN" GIT_AUTHOR_EMAIL="$AE" GIT_AUTHOR_DATE="$AD" \
      git commit -q --no-verify -F "$tmp.commitmsg"
  else
    GIT_AUTHOR_NAME="$AN" GIT_AUTHOR_EMAIL="$AE" \
      git commit -q --no-verify -F "$tmp.commitmsg"
  fi
  echo "  [commit] $(git log -1 --format='%h %an  %ad  %s' --date=short)"
done
rm -f "$tmp" "$tmp.patch" "$tmp.commitmsg" "$msg" "$info"
echo "[done] replayed $(echo "$groups" | wc -w) commits"
