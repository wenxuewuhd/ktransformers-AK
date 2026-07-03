#!/usr/bin/env bash
# F2 整网冒烟：Handoff 附录 Z.8 四个 prompt（需服务已起，PORT 与 launch 一致）。
# 用法：HOST=127.0.0.1 PORT=8000 bash tools/p27_curl_f2_prompts.sh

set -euo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# Python: honor PYTHON_BIN/PYBIN, else the manual image path, else auto-detect
# (new hosts differ, e.g. /opt/buildtools/Python-3.11.4/bin/python3.11).
PYBIN="${PYTHON_BIN:-${PYBIN:-}}"
if [[ -z "$PYBIN" || ! -x "$PYBIN" ]]; then
  for cand in /usr/local/python3.11.14/bin/python3.11 \
              /opt/buildtools/Python-3.11.4/bin/python3.11 \
              "$(command -v python3.11 2>/dev/null)" \
              "$(command -v python3 2>/dev/null)"; do
    [[ -n "$cand" && -x "$cand" ]] && { PYBIN="$cand"; break; }
  done
fi

"$PYBIN" - <<'PY'
import json
import urllib.request

host = __import__("os").environ.get("HOST", "127.0.0.1")
port = __import__("os").environ.get("PORT", "8000")
base = f"http://{host}:{port}/generate"

# prompt 6: realistic ~7k-token long-context test (needle-in-a-haystack) + 1024-token output.
# An engineering meeting log with ONE planted fact; the model must retrieve it. Sized ~7k tokens so
# it fits in a SINGLE chunk at chunked-prefill-size 8192 (no cross-chunk crash) and activation stays
# small (no OOM) -> exercises the streaming prefill end to end. Correct answer ("QX-4729-ZK") =
# long-context retrieval works; garbage/wrong = broken. prompt_tokens is printed in the response.
# (For a 32k stress test bump range() to ~960, but that needs chunked-prefill-size>=32768 AND freed
#  HBM -- e.g. --context-length 34816 -- else it OOMs->hybrid or hits the cross-chunk bug.)
_topics = [
    "the database index migration", "the API rate limiter", "cache eviction tuning",
    "the on-call rotation", "log retention", "a CI flake in the auth module",
    "memory use in the ingest worker", "the gRPC timeout config", "dashboard latency",
    "the feature-flag rollout", "the backup-restore drill", "TLS certificate renewal",
]
_log = [
    f"Entry {i} (2026-{i % 12 + 1:02d}-{i % 27 + 1:02d}): the team discussed "
    f"{_topics[i % len(_topics)]}; action items were assigned and the status was set to in-progress."
    for i in range(210)
]
# the one fact to retrieve, buried in the middle of the log
_log[105] = ("Entry 105: IMPORTANT - the production deploy key for this quarter is QX-4729-ZK. "
             "Keep it confidential and do not share it outside the on-call group.")
_p6_text = (
    "You are reviewing the engineering team's meeting log below.\n\n"
    + "\n".join(_log)
    + "\n\nQuestion: What is the production deploy key mentioned somewhere in the log above? "
      "Reply with the exact key, then quote the line you found it on."
)

# prompt 5: SHORT-context needle probe -- same planted fact as prompt 6 but in a tiny log (~12
# entries, a few hundred tokens). It sits well inside NSA's dense/sliding window so the needle is
# always attended. This is the discriminator: if the model retrieves "QX-4729-ZK" here but NOT in
# prompt 6's ~7k log, the long-context failure is sparse-attention selection (NSA dropping the needle
# block), not our MoE/stream path. If it fails HERE too, the base model simply isn't following the
# retrieval instruction -> no accuracy bug to chase in the prefill path.
_short_log = [
    f"Entry {i} (2026-{i % 12 + 1:02d}-{i % 27 + 1:02d}): the team discussed "
    f"{_topics[i % len(_topics)]}; action items were assigned and the status was set to in-progress."
    for i in range(12)
]
_short_log[5] = ("Entry 5: IMPORTANT - the production deploy key for this quarter is QX-4729-ZK. "
                 "Keep it confidential and do not share it outside the on-call group.")
_p5_short_text = (
    "You are reviewing the engineering team's meeting log below.\n\n"
    + "\n".join(_short_log)
    + "\n\nQuestion: What is the production deploy key mentioned somewhere in the log above? "
      "Reply with the exact key, then quote the line you found it on."
)

# prompt 7: SAME ~7k log as prompt 6, but the needle is moved to the TAIL (Entry 205 of 210) so it
# falls inside NSA's recent/sliding-window dense region. Length is identical to prompt 6; only the
# needle POSITION differs. Discriminator for B: if the tail needle retrieves "QX-4729-ZK" while the
# mid-context one (prompt 6) does not, the long-context failure is NSA sparse-block SELECTION dropping
# the middle block -- not the base model, not RoPE/length. If the tail ALSO fails, the cause is
# length-wide (RoPE / position) rather than selection.
_tail_log = [
    f"Entry {i} (2026-{i % 12 + 1:02d}-{i % 27 + 1:02d}): the team discussed "
    f"{_topics[i % len(_topics)]}; action items were assigned and the status was set to in-progress."
    for i in range(210)
]
_tail_log[205] = ("Entry 205: IMPORTANT - the production deploy key for this quarter is QX-4729-ZK. "
                  "Keep it confidential and do not share it outside the on-call group.")
_p7_tail_text = (
    "You are reviewing the engineering team's meeting log below.\n\n"
    + "\n".join(_tail_log)
    + "\n\nQuestion: What is the production deploy key mentioned somewhere in the log above? "
      "Reply with the exact key, then quote the line you found it on."
)

prompts = [
    (1, 64, "Below is a Python function to compute Fibonacci numbers:"),
    (
        2,
        128,
        "Explain the difference between supervised and unsupervised learning in three short paragraphs.\n\n",
    ),
    (3, 80, "请用一句话解释什么是 transformer 模型："),
    (4, 256, "什么是 transformer 模型："),
    (5, 1024, _p5_short_text),  # SHORT-context needle probe (dense, NSA-safe) -- discriminator
    (6, 1024, _p6_text),  # ~7k prefill, needle at MIDDLE (Entry 105) -- NSA selection discriminator
    (7, 1024, _p7_tail_text),  # ~7k prefill, needle at TAIL (Entry 205, NSA dense window)
]

for pid, max_tok, text in prompts:
    print(f"========== prompt {pid} (max_new_tokens={max_tok}) ==========")
    body = {"text": text, "sampling_params": {"max_new_tokens": max_tok, "temperature": 0}}
    req = urllib.request.Request(
        base,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        print(r.read()[:2500].decode(errors="replace"))
    print()

print("[f2] done — 人工看 text：无 NaN/全感叹号/乱码即通过（base 模型不考核指令遵循）")
PY
