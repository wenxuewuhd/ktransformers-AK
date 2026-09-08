#!/usr/bin/env bash
# Send ONE request to the bs=1 server and print the decode rate CLIENT-SIDE.
#
#   ./ask.sh                      1024-token prompt, 256 new tokens
#   NTOK=512 ./ask.sh             longer generation
#   PROMPT_TOKENS=64 ./ask.sh     below the streaming threshold -> hybrid path
#   FULL=1 ./ask.sh               print the prompt in full instead of head+tail
#
# The rate printed here is measured from the SECOND token onward. That is deliberate:
# the gap before the first token is TTFT, and on the streaming-prefill path TTFT is
# ~18 s. Dividing total wall by total tokens folds that in and reports a single-digit
# "throughput" for a server that is actually decoding at >20 tok/s. The figure below
# is the same quantity sglang prints on its own "Decode batch ... gen throughput" lines.
set -uo pipefail
# Sourced, not re-declared: this used to carry its own port default (30039 against the
# documented 30013), its own interpreter path and its own corpus path, so "the default
# server" meant three different things inside one directory.  glm53_env.sh also unsets the
# proxy -- a proxy set for github hijacks 127.0.0.1 and every request below would be
# posted to it instead of to the server.
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
# shellcheck source=./glm53_env.sh
. "${_here}/glm53_env.sh"

case "${1:-}" in
  -h|--help)
    cat <<'USAGE'
usage: ask.sh

Send ONE request to an already-running server and print the CLIENT-side decode rate,
timed from the second token so TTFT is not folded in.

  GLM53_PORT=<n>       which server (default from glm53_env.sh)
  NTOK=<n>             tokens to generate      (default 256)
  PROMPT_TOKENS=<n>    approximate prompt size (default 1024)
  FULL=1               print the whole prompt instead of head+tail
USAGE
    exit 0 ;;
  "") ;;
  *) echo "[ask] unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

NTOK="${NTOK:-256}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"

if [ -z "${GLM53_EVAL_DIR}" ] || [ ! -d "${GLM53_EVAL_DIR}" ]; then
  echo "[ask] FATAL: no evaluation corpus. Set GLM53_EVAL_DIR to a directory holding" >&2
  echo "[ask]   wikitext/test.parquet (or any parquet of real prose)." >&2
  exit 1
fi

"${GLM53_PYTHON}" - "${GLM53_HOST}" "${GLM53_PORT}" "$NTOK" "$PROMPT_TOKENS" "${GLM53_EVAL_DIR}" <<'PY'
import json, os, pathlib, sys, time, urllib.request

host, port, ntok, want_tokens, eval_dir = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])

# ⚠ Real prose, not "Item 0001 is in bay 1." repeated. A repetitive prompt routes to far
# fewer experts than real text does, which inflates the resident hit rate and hands you a
# decode rate the model cannot reach on anything real.
src = pathlib.Path(eval_dir) / "wikitext" / "test.parquet"
if not src.is_file():
    sys.exit(f"no corpus at {src}; point this at any parquet of real prose")
import pandas as pd
txt = "\n".join(t for t in pd.read_parquet(src).iloc[:, 0].tolist() if t.strip())
prompt = txt[: want_tokens * 4]

full = os.environ.get("FULL", "0") == "1"
print(f"  prompt {len(prompt)} chars (~{len(prompt)//4} tokens), asking for {ntok} new tokens")
print("\n  ================= PROMPT =================")
if full or len(prompt) <= 1200:
    print("  " + prompt.replace("\n", "\n  "))
else:
    print("  " + prompt[:700].replace("\n", "\n  "))
    print(f"\n  [... {len(prompt)-1000} chars elided, FULL=1 to print all ...]\n")
    print("  " + prompt[-300:].replace("\n", "\n  "))
print("  ==========================================\n")

body = json.dumps({
    "text": prompt,
    "sampling_params": {"max_new_tokens": ntok, "temperature": 0, "ignore_eos": True},
    "stream": True,
}).encode()
req = urllib.request.Request(f"http://{host}:{port}/generate", data=body,
                             headers={"Content-Type": "application/json"})

print(f"  POST /generate (stream)  ->  timing from the 2nd token, TTFT excluded")
stamps, text, meta = [], "", {}
t0 = time.perf_counter()
# urllib honours no_proxy poorly for some setups; the env was cleared by the caller.
with urllib.request.urlopen(req) as r:
    for raw in r:
        raw = raw.strip()
        if not raw.startswith(b"data:"):
            continue
        payload = raw[5:].strip()
        if payload == b"[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        stamps.append(time.perf_counter())
        text = d.get("text", text)
        meta = d.get("meta_info", meta) or meta
        n = len(stamps)
        if n == 1:
            print(f"  first token at {stamps[0]-t0:8.2f} s   <- TTFT (prefill), excluded below")
        elif n % 40 == 0:
            r_now = (n - 1) / (stamps[-1] - stamps[0])
            print(f"  {n:4d} tokens   decode {r_now:6.2f} tok/s   ({1000/r_now:5.1f} ms/token)")

gen = meta.get("completion_tokens") or len(stamps)
print("\n  ================= OUTPUT =================")
print("  " + (text or "(empty)").replace("\n", "\n  "))
print("  ==========================================")

if len(stamps) >= 2:
    span = stamps[-1] - stamps[0]
    rate = (len(stamps) - 1) / span
    ttft = stamps[0] - t0
    total = stamps[-1] - t0
    print(f"\n  prompt_tokens   {meta.get('prompt_tokens')}")
    print(f"  completion      {gen} tokens")
    print(f"  TTFT            {ttft:8.2f} s      (prefill; streaming path re-reads the")
    print(f"                                   checkpoint per chunk, so this is large)")
    print(f"  ─────────────────────────────────────────")
    print(f"  DECODE          {rate:8.2f} tok/s  ({1000/rate:.1f} ms/token)   <- the number")
    print(f"  ─────────────────────────────────────────")
    print(f"  naive wall/tok  {gen/total:8.2f} tok/s  <- WRONG: TTFT folded in. Ignore it.")
    print(f"\n  cross-check: the server's own 'Decode batch ... gen throughput' lines")
    print(f"  in the serve_fg.sh terminal should read the same as DECODE above.")
else:
    print("\n  too few chunks to time; raise NTOK")
PY
