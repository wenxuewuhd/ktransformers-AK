#!/usr/bin/env bash
# The queued performance experiments, in one pass, for one quiet window.
#
#   ./experiments.sh residency   --kt-num-gpu-experts 8/16/32/40, 8 NUMA
#   ./experiments.sh threads     --kt-cpuinfer 8/16/32/40, 1 NUMA
#   ./experiments.sh profile     one profile of the current config: device vs host
#   ./experiments.sh all
#
# Every point goes through bench.sh, which refuses to measure on a contaminated box.
# That is the point: an unattended sweep is exactly where a neighbour's job gets
# silently folded into a result.
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${_here}/glm53_env.sh"

RESULTS="${GLM53_LOG_DIR}/experiments_$(date +%m%d_%H%M)"
mkdir -p "${RESULTS}"
echo "results -> ${RESULTS}"

# ---------------------------------------------------------------------------
# Q: is decode time driven by the BYTES the CPU reads?
#
# With prefix placement and N resident of 288, a uniform-routing hit rate is N/288,
# so bytes/token scale as (1 - N/288). If wall time does not follow that slope, the
# byte count is not the binding constraint -- and hot-expert placement, which only
# ever changes the byte count, cannot help. Worth knowing before building it.
#
# 8 is the low point, not 0: --kt-num-gpu-experts 0 may not build a resident
# GroupedMatmul at all, and a point on a different code path cannot be read as part
# of the same slope.
# ---------------------------------------------------------------------------
# The order is an A/A bracket, not a sweep: 32 is measured at the start, the middle
# and the end. A shared box is rarely quiet, but a *slope* survives contention that a
# level does not -- provided the contention held still. The three 32-points are how we
# find out. If they agree, the drift is bounded and the slope between 8 and 40 means
# something. If they scatter, nothing in the run does, and no amount of staring at the
# other points will fix it.
RESIDENCY_ORDER="${RESIDENCY_ORDER:-32 8 16 32 40 32}"

exp_residency() {
  echo "== residency sweep (8 NUMA, threads fixed) =="
  echo "   order ${RESIDENCY_ORDER} -- repeats of 32 bracket the sweep as an A/A control"
  local i=0
  for N in ${RESIDENCY_ORDER}; do
    i=$((i+1))
    # 15.46 + 1.009*N GiB of weights; 40 is ~55.8 GiB and the largest that leaves
    # room for KV, the KDA state and graph buffers on a 61.3 GiB die.
    NAME="res${N}_$(printf '%02d' ${i})" GLM53_NUM_GPU_EXPERTS="${N}" \
      "${_here}/bench.sh" 2>&1 | tee "${RESULTS}/residency_${i}_${N}.log"
    echo
  done
}

# ---------------------------------------------------------------------------
# Q: does the CPU MoE scale with threads inside ONE NUMA node?
#
# This is the sweep that was contaminated the first time. One node, 40 cores, is the
# deployment target's shape -- the target image is a container carved from a node
# exactly like this one.
# ---------------------------------------------------------------------------
exp_threads() {
  echo "== thread sweep (1 NUMA node, residency fixed) =="
  for T in 8 16 32 40; do
    NAME="thr${T}" GLM53_THREADPOOL_COUNT=1 GLM53_CPUINFER="${T}" \
      "${_here}/bench.sh" 2>&1 | tee "${RESULTS}/threads_${T}.log"
    echo
  done
}

# ---------------------------------------------------------------------------
# Q: how much of a decode step is device, and where does the host time land?
#
# Do not borrow another configuration's NPU number for this. The routing bookkeeping
# kernels scale with the EXPERT COUNT, not the selected count: MoeInitRoutingV3
# scatters into n_experts buckets, MoeGatingTopK sees n_experts logits, and the
# router MatMulV2 outputs n_experts. A 16-expert measurement understates all three
# for a 288-expert model.
# ---------------------------------------------------------------------------
exp_profile() {
  echo "== profile: device vs host =="
  local prof="${RESULTS}/prof"
  local tools="${SGLANG_REPO}/docs/docs/glm53_npu_support/tools"
  [ -f "${tools}/profile_server_decode.py" ] || { echo "  profile_server_decode.py not found under ${tools}" >&2; return 1; }

  NAME=profile_base "${_here}/bench.sh" 2>&1 | tee "${RESULTS}/profile_bench.log"
  local wall
  wall="$("${GLM53_PYTHON}" - "${GLM53_LOG_DIR}" <<'PY'
import glob, json, os, sys
# summary.decode_ms_per_token, NOT rows[-1]: the per-row figures are the naive
# wall/tokens of one arm and still contain TTFT, which on the streaming path is tens of
# seconds. bench.sh's whole point is the two-length subtraction, and only the summary
# carries its result. (This read was also spelled "ms_per_step_med", a key bench.sh has
# never written, so it raised KeyError and the caller silently dropped --wall-ms.)
f = sorted(glob.glob(os.path.join(sys.argv[1], "bench_profile_base_*.json")))
print((json.load(open(f[-1])).get("summary") or {}).get("decode_ms_per_token", "") if f else "")
PY
)"
  echo "  wall clock: ${wall:-?} ms/step"

  "${GLM53_PYTHON}" "${tools}/profile_server_decode.py" \
      --port "${GLM53_PORT}" --out "${prof}" --steps 20 --concurrency 1 \
    2>&1 | tail -5
  "${_here}/analyze_profile.py" --profile "${prof}" --steps 20 \
      ${wall:+--wall-ms "${wall}"} 2>&1 | tee "${RESULTS}/profile_analysis.txt"
}

# ---------------------------------------------------------------------------
# Q: did hot-expert residency actually reduce the bytes the CPU reads?
#
# Throughput alone is a weak answer -- it moves for many reasons. The direct evidence is
# the per-layer stall at `MoeFinalizeRoutingV2 -> Add`, which IS the CPU wait: at a
# resident hit rate h the CPU reads 633.3 * (1-h)/(1-0.1111) us per layer, of which
# ~143.6 us hides under the resident GEMM. Measured at h=0.109: 489.7 us. Predicted at
# h=0.230, a held-out offline profile: 355 us.
#
# So: the same measurement twice, dynamic residency off then on, reporting throughput
# AND that one gap. If the gap does not move, the placement did not change what the CPU
# reads, whatever else changed.
# ---------------------------------------------------------------------------
exp_hot() {
  echo "== hot-expert residency A/B =="
  local tools="${SGLANG_REPO}/docs/docs/glm53_npu_support/tools"
  local MODE wall
  for MODE in off on; do
    echo "--- KT_DYNAMIC_RESIDENT=${MODE} ---"
    if [ "${MODE}" = "on" ]; then
      export GLM53_PREFILL_STREAM=1 KT_DYNAMIC_RESIDENT=1
    else
      unset KT_DYNAMIC_RESIDENT; export GLM53_PREFILL_STREAM=0
    fi
    NAME="hot_${MODE}" "${_here}/bench.sh" 2>&1 | tee "${RESULTS}/hot_${MODE}.log"
    wall="$("${GLM53_PYTHON}" "${_here}/_last_decode_ms.py" "${GLM53_LOG_DIR}" "hot_${MODE}")"
    "${GLM53_PYTHON}" "${tools}/profile_server_decode.py" --port "${GLM53_PORT}" \
        --out "${RESULTS}/prof_${MODE}" --steps 20 --concurrency 1 \
        --decode-tokens 500 --prefill-settle 25 >/dev/null 2>&1
    "${_here}/analyze_profile.py" --profile "${RESULTS}/prof_${MODE}" --steps 20 \
        ${wall:+--wall-ms "${wall}"} --gap-us 50 > "${RESULTS}/gaps_${MODE}.txt" 2>&1
    echo "  decode ${wall:-?} ms/token"
    grep -E "^device|^host" "${RESULTS}/gaps_${MODE}.txt" | head -3
  done
  echo
  echo "== did the CPU read fewer bytes? =="
  "${GLM53_PYTHON}" "${_here}/_hot_verdict.py" "${RESULTS}"
}

case "${1:-all}" in
  hot)       exp_hot ;;
  residency) exp_residency ;;
  threads)   exp_threads ;;
  profile)   exp_profile ;;
  all)       exp_profile; exp_residency; exp_threads ;;
  *) echo "usage: $0 {hot|residency|threads|profile|all}" >&2; exit 2 ;;
esac

echo
echo "== summary =="
"${GLM53_PYTHON}" - "${GLM53_LOG_DIR}" <<'PY'
import glob, json, os, sys
# Read the SUMMARY, not rows[-1]. rows[] holds per-arm naive wall/token figures with
# TTFT folded in; summary holds the two-length-subtracted decode rate, which is the
# only number in this file worth putting in a table. The previous keys
# ("tok_s_med" / "ms_per_step_med") do not exist in any bench.sh output, so this table
# raised KeyError on every invocation of this script.
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "bench_*.json"))):
    d = json.load(open(f))
    sm = d.get("summary") or {}
    ts, ms = sm.get("decode_tok_s"), sm.get("decode_ms_per_token")
    if ts is None or ms is None:
        continue          # an aborted or gate-refused run writes no summary
    rows.append((d.get("name", os.path.basename(f)), d.get("verdict", "?"),
                 d["config"].get("threadpool_count"), d["config"].get("cpuinfer"),
                 d["config"].get("num_gpu_experts"), ts, ms))
if rows:
    print(f"  {'name':<20s}{'verdict':<14s}{'pools':>6s}{'thr':>5s}{'resident':>9s}"
          f"{'tok/s':>9s}{'ms/step':>9s}")
    for n, v, p, t, e, ts, ms in rows:
        print(f"  {n:<20s}{v:<14s}{str(p):>6s}{str(t):>5s}{str(e):>9s}{ts:9.2f}{ms:9.1f}")
    bad = [n for n, v, *_ in rows if v != "clean"]
    if bad:
        print(f"\n  ⚠ contaminated, do not compare: {', '.join(bad)}")
PY
