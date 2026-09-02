#!/usr/bin/env bash
# Decode-throughput measurement with collision detection.
#
#   NAME=base ./bench.sh                       one point, current env
#   NAME=t32  GLM53_CPUINFER=32 ./bench.sh     one point, overridden
#
# This host is shared. A measurement taken while someone else's job is running is
# not a slow measurement, it is a meaningless one -- and it does not look wrong.
# Every gate below exists because that happened: a full thread sweep was taken on
# top of a neighbouring TP8 eval job and produced a clean-looking saturation curve
# that meant nothing. Do not remove these checks to "just get a number".
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./glm53_env.sh
source "${_here}/glm53_env.sh"

case "${1:-}" in
  -h|--help)
    cat <<'USAGE'
usage: bench.sh

Contention-gated decode measurement. Starts its own server, measures with a two-length
subtraction, stops the server on exit, and writes a JSON with the contamination verdict.
Configured entirely by environment; see README.md.

  NAME=<tag>     names the result JSON
  BENCH_FORCE=1  bypass the contamination gates -- do not use when they refuse
USAGE
    exit 0 ;;
  "") ;;
  *) echo "  unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

NAME="${NAME:-run}"
OUT="${GLM53_LOG_DIR}/bench_${NAME}_$(date +%H%M).json"
# One die is ~65536 MiB; idle sits near 3 GiB. 6553 = 10%.
DIE_IDLE_MIB="${DIE_IDLE_MIB:-6553}"
# 40 cores per NUMA node. A quiet box for a 1-node measurement is well under that.
LOAD_MAX="${LOAD_MAX:-8}"

_die_used() { npu-smi info 2>&1 | grep -oP '\d+(?=\s*/ 65536)' | sed -n "$(( GLM53_NPU_DEVICE_ID + 1 ))p"; }
_load()     { awk '{print int($1)}' /proc/loadavg; }
_others()   { ps -eo pid,rss,etimes,args --sort=-rss 2>/dev/null \
                | grep -E "launch_server|scheduler_TP" | grep -v "port ${GLM53_PORT}" | grep -vc grep; }

_contention_report() {
  printf '  load=%s  foreign sglang procs=%s  die%s=%s MiB\n' \
    "$(cat /proc/loadavg | cut -d' ' -f1-3)" "$(_others)" "${GLM53_NPU_DEVICE_ID}" "$(_die_used)"
}

echo "== bench '${NAME}' =="
echo "  die ${GLM53_NPU_DEVICE_ID}  port ${GLM53_PORT}  threadpool ${GLM53_THREADPOOL_COUNT}  cpuinfer ${GLM53_CPUINFER}  resident ${GLM53_NUM_GPU_EXPERTS}"

# ---- gate 1: is anyone else on this box? -----------------------------------
echo "before:"; _contention_report
FOREIGN_BEFORE="$(_others)"; LOAD_BEFORE="$(_load)"
if [ "${FOREIGN_BEFORE}" -gt 0 ] || [ "${LOAD_BEFORE}" -gt "${LOAD_MAX}" ]; then
  echo "  REFUSING: ${FOREIGN_BEFORE} foreign sglang processes, load ${LOAD_BEFORE} (max ${LOAD_MAX})." >&2
  echo "  A number taken now is not a slow number, it is not a number. Wait, or set" >&2
  echo "  BENCH_FORCE=1 to record it explicitly marked contaminated." >&2
  [ "${BENCH_FORCE:-0}" = "1" ] || exit 2
  echo "  BENCH_FORCE=1: continuing, result will be marked contaminated." >&2
fi

# ---- gate 2: our die must be free ------------------------------------------
# Retire OUR previous server BEFORE waiting on the die. Waiting first deadlocks a sweep
# against itself: point N+1 waits out the whole timeout on point N's still-running
# process, refuses, and every later point inherits the stall.
# Only ever by port, and with a bracket so the pattern cannot match this script's own
# command line -- `pkill -f 'experiments.sh'` from inside the sweep kills the sweep.
# Never `pkill -f sglang` on a box shared under one OS account.
pkill -f -- "[-]-port ${GLM53_PORT}" 2>/dev/null
sleep 2
pkill -9 -f -- "[-]-port ${GLM53_PORT}" 2>/dev/null

t=0
while [ "$(_die_used)" -ge "${DIE_IDLE_MIB}" ] && [ "${t}" -lt "${DIE_WAIT:-600}" ]; do
  [ "${t}" -eq 0 ] && echo "  waiting for die ${GLM53_NPU_DEVICE_ID} to fall back (HBM lingers 2-3 min after a kill)..."
  sleep 15; t=$((t+15))
done
if [ "$(_die_used)" -ge "${DIE_IDLE_MIB}" ]; then
  echo "  REFUSING: die ${GLM53_NPU_DEVICE_ID} still holds $(_die_used) MiB after ${t}s." >&2
  echo "  Nothing of ours is on it, so someone else took the die." >&2
  exit 2
fi

# ---- launch ----------------------------------------------------------------
# ⛔ Stop the server this script starts, on EVERY exit including Ctrl-C. Without this a
# killed wrapper leaves sglang holding the die, and the next person to run gets
# "NPU out of memory ... 166 MiB free" -- an error that points at the victim rather than
# at the orphan. Only ever by port, and with the bracket so the pattern cannot match this
# script's own command line. Never `pkill -f sglang`: one OS account, several people.
_bench_cleanup() { pkill -f -- "[-]-port ${GLM53_PORT}" 2>/dev/null; return 0; }
trap _bench_cleanup EXIT INT TERM

"${_here}/serve.sh" >/dev/null 2>&1
PID="$(cat "${GLM53_LOG_DIR}/serve.log.pid")"
t=0
until curl -sf -m3 --noproxy '*' "http://127.0.0.1:${GLM53_PORT}/health" >/dev/null 2>&1 \
      || [ "${t}" -ge "${UP_WAIT:-900}" ] || ! kill -0 "${PID}" 2>/dev/null; do sleep 15; t=$((t+15)); done
kill -0 "${PID}" 2>/dev/null || { echo "  server died during startup, see ${GLM53_LOG_DIR}/serve.log" >&2; exit 1; }

# ---- gate 3: did we collide during weight load? ----------------------------
# A neighbour that grabbed the die mid-load shows up as a small avail-mem figure
# here, long before it shows up as a slow benchmark.
AVAIL="$(grep -oP 'Load weight begin\. avail mem=\K[0-9.]+' "${GLM53_LOG_DIR}/serve.log" | tail -1)"
echo "  Load weight begin. avail mem=${AVAIL:-?} GB (healthy is ~60.8 on an idle die)"
if [ -n "${AVAIL}" ] && awk "BEGIN{exit !(${AVAIL} < 58)}"; then
  echo "  WARNING: only ${AVAIL} GB was free when weights started loading -- someone else" >&2
  echo "  was on this die. This is a collision, not a regression." >&2
fi

# ---- measure ---------------------------------------------------------------
"${GLM53_PYTHON}" - "${GLM53_PORT}" "${OUT}" "${NAME}" <<'PY'
import json, os, sys, time, urllib.request
port, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"

def _build_prompt():
    import os, pathlib
    n_items = int(os.environ.get("BENCH_PROMPT_ITEMS", "60"))
    if os.environ.get("BENCH_PROMPT_SYNTHETIC", "0") == "1":
        return " ".join(f"Item {i:04d} is in bay {i%37}." for i in range(n_items))
    want_chars = int(os.environ.get("BENCH_PROMPT_TOKENS", str(n_items * 105 // 10))) * 4
    for cand in (os.environ.get("GLM53_EVAL_DIR", ""),):
        f = pathlib.Path(cand or ".") / "wikitext" / "test.parquet"
        if f.is_file():
            try:
                import pandas as pd
                txt = "\n".join(t for t in pd.read_parquet(f).iloc[:, 0].tolist() if t.strip())
                if len(txt) >= want_chars:
                    return txt[:want_chars]
            except Exception:
                pass
    # No corpus: say so rather than silently falling back to the degenerate filler.
    raise SystemExit("bench: no wikitext corpus for a realistic prompt; set "
                     "GLM53_EVAL_DIR, or BENCH_PROMPT_SYNTHETIC=1 to accept the "
                     "repetitive filler and its overstated routing concentration")


_last_text = ""


def gen(text, n):
    global _last_text
    t0 = time.time()
    req = urllib.request.Request(base + "/generate",
        data=json.dumps({"text": text, "sampling_params":
            {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    _last_text = d.get("text", "")
    return time.time() - t0, d["meta_info"]

# BENCH_PROMPT_ITEMS: ~10.5 tokens per item.  60 (default) is the historical ~630-token
# prompt; a knob that only affects prompt length is needed to measure KT_HOT_TAIL_TOKENS,
# whose N is compared against the prompt length.
# ⚠ The prompt's CONTENT matters here, not just its length. The old synthetic filler
# ("Item 0000 is in bay 0. Item 0001 is in bay 1. ...") was 630 tokens with 48 unique
# ones -- 92.4% repetitive. Expert routing on text like that is far more concentrated
# than on real text, which flatters anything that depends on routing concentration:
# measured hot-32 share is 0.53-0.68 on the synthetic filler against 0.230 held out on
# real text. A decode benchmark run on it overstates dynamic hot-expert residency.
#
# So: real prose, from the same wikitext-2 the perplexity criterion uses, truncated to
# the requested token budget. BENCH_PROMPT_TOKENS sets the length; the default 630
# keeps the historical size so old and new numbers are comparable in length if not in
# content. BENCH_PROMPT_SYNTHETIC=1 restores the old filler for an A/B against history.
prompt = _build_prompt()
gen(prompt, 8)  # warm: a cold prefill lands entirely inside a subtraction-based
                # decode estimate and has manufactured whole conclusions here before.

# Two lengths, then subtract. wall/generated_tokens is NOT the decode rate: it folds
# in a prefill that costs seconds here, because every prefill chunk pays the full CPU
# MoE. On a 630-token prompt that inflated 52.7 ms/token of real decode into an
# apparent 89 ms/token -- and made a healthy system look 40% slow.
#
#   wall(n) = prefill + n * decode      =>  decode = (wall(n2) - wall(n1)) / (n2 - n1)
#
# Both lengths use the identical prompt, so prefill cancels exactly. The warmup above
# is what makes this legitimate: one cold prefill lands entirely in the subtrahend.
rows = []
for ntok in (64, 256):
    walls = []
    for _ in range(3):
        w, m = gen(prompt, ntok)
        walls.append(w)
    walls.sort()
    ct, pt = m["completion_tokens"], m["prompt_tokens"]
    last_text = _last_text
    med = walls[1]
    rows.append({"gen_tokens": ct, "prompt_tokens": pt,
                 "wall_s": {"min": walls[0], "med": med, "max": walls[2]},
                 "naive_tok_s": ct / med, "naive_ms_per_step": med * 1000 / ct})
    print(f"  prompt={pt:5d} gen={ct:4d}  wall med={med:7.2f}s "
          f"(min {walls[0]:.2f} max {walls[2]:.2f})")

(a, b) = rows
dn = b["gen_tokens"] - a["gen_tokens"]
decode_ms = (b["wall_s"]["med"] - a["wall_s"]["med"]) * 1000 / dn
prefill_s = a["wall_s"]["med"] - a["gen_tokens"] * decode_ms / 1000
summary = {"decode_ms_per_token": decode_ms, "decode_tok_s": 1000 / decode_ms,
           "prefill_s": prefill_s, "prompt_tokens": a["prompt_tokens"],
           "prefill_ms_per_prompt_token": prefill_s * 1000 / a["prompt_tokens"],
           # ignore_eos forces the full token budget, so a broken model still produces a
           # number. Keep a sample of what it actually said, and a crude degeneracy
           # check, so a throughput figure can never be read without seeing its output.
           "sample_output": _last_text[:400],
           "output_unique_word_ratio": (
               len(set(_last_text.split())) / max(1, len(_last_text.split()))),
           "prompt_head": prompt[:120]}
print(f"      output: {_last_text[:90]!r}...")
print(f"      unique-word ratio {summary['output_unique_word_ratio']:.2f} "
      f"({'looks degenerate, CHECK IT' if summary['output_unique_word_ratio'] < 0.25 else 'not obviously degenerate'})")
print(f"  --> decode  {decode_ms:6.1f} ms/token   {1000/decode_ms:5.2f} tok/s   "
      f"(from the {a['gen_tokens']}/{b['gen_tokens']} difference)")
print(f"      prefill {prefill_s:6.2f} s for {a['prompt_tokens']} tokens "
      f"({prefill_s*1000/a['prompt_tokens']:.1f} ms/token)")
json.dump({"name": name, "rows": rows, "summary": summary}, open(out, "w"), indent=1)
PY
RC=$?

# ---- gate 4: did anyone arrive while we measured? --------------------------
echo "after:"; _contention_report
FOREIGN_AFTER="$(_others)"; LOAD_AFTER="$(_load)"
VERDICT=clean
if [ "${FOREIGN_AFTER}" -gt "${FOREIGN_BEFORE}" ]; then
  echo "  ⚠ CONTAMINATED: foreign sglang processes went ${FOREIGN_BEFORE} -> ${FOREIGN_AFTER} during the run." >&2
  VERDICT=contaminated
elif [ "${LOAD_AFTER}" -gt $(( LOAD_BEFORE + LOAD_MAX )) ]; then
  echo "  ⚠ CONTAMINATED: load went ${LOAD_BEFORE} -> ${LOAD_AFTER} during the run." >&2
  VERDICT=contaminated
fi
[ "${FOREIGN_BEFORE}" -gt 0 ] && VERDICT=contaminated

"${GLM53_PYTHON}" - "${OUT}" "${VERDICT}" "${LOAD_BEFORE}" "${LOAD_AFTER}" \
  "${FOREIGN_BEFORE}" "${FOREIGN_AFTER}" "${GLM53_THREADPOOL_COUNT}" \
  "${GLM53_CPUINFER}" "${GLM53_NUM_GPU_EXPERTS}" "${AVAIL:-}" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["verdict"] = sys.argv[2]
d["contention"] = {"load_before": sys.argv[3], "load_after": sys.argv[4],
                   "foreign_before": sys.argv[5], "foreign_after": sys.argv[6]}
d["config"] = {"threadpool_count": sys.argv[7], "cpuinfer": sys.argv[8],
               "num_gpu_experts": sys.argv[9], "avail_mem_at_load": sys.argv[10]}
json.dump(d, open(p, "w"), indent=1)
PY
# Streaming is silent when it fails: every exception falls back to the hybrid path, so a
# decode number attributed to "streaming + dynamic hot" can be the hybrid path's number.
# Record what the log says, in the result, so the attribution travels with the figure.
# ⚠ `grep -c ... || echo 0` is wrong: grep already prints "0" when it matches nothing,
# and it ALSO exits 1, so the fallback fires too and the variable becomes "0\n0". That
# then blew up the JSON writer with `invalid literal for int(): '0\n0'` and lost the
# whole result record. Let grep's own zero stand; only supply a default if it cannot run.
_INLINE="$(grep -ac 'inline resident' "${GLM53_LOG_DIR}/serve.log" 2>/dev/null | head -1)"
_FB="$(grep -acE 'streaming failed|hybrid fallback' "${GLM53_LOG_DIR}/serve.log" 2>/dev/null | head -1)"
_INLINE="${_INLINE:-0}"; _FB="${_FB:-0}"
echo "  streaming: inline resident ${_INLINE}, fallbacks ${_FB} (expect >0 and 0 when GLM53_PREFILL_STREAM=1)"
if [ "${GLM53_PREFILL_STREAM:-0}" = "1" ] && [ "${_INLINE}" -eq 0 ]; then
  echo "  ⚠ GLM53_PREFILL_STREAM=1 but streaming never engaged -- this measured the HYBRID path" >&2
  VERDICT="${VERDICT}+stream_never_engaged"
fi
"${GLM53_PYTHON}" - "${OUT}" "${_INLINE}" "${_FB}" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d["streaming"] = {"inline_resident": int(sys.argv[2]), "fallbacks": int(sys.argv[3])}
json.dump(d, open(p, "w"), indent=1)
PY
echo "  verdict: ${VERDICT}    -> ${OUT}"
[ "${VERDICT}" = "clean" ] || echo "  ⚠ Do not compare a contaminated point against anything."
exit "${RC}"
