#!/usr/bin/env bash
# Launch ONE bs=1 dynamic-hot server in the foreground so sglang's own prints -- in
# particular the per-batch
#     Decode batch, #running-req: 1, ... gen throughput (token/s): NN.NN
# lines -- land straight in this terminal. Then drive it from another shell with ask.sh
# and watch those lines appear.
#
#   ./serve_fg.sh                                    the shared defaults: die 0, port 30013
#   GLM53_NPU_DEVICE_ID=9 GLM53_PORT=30061 ./serve_fg.sh      pick your own on a shared box
#
# ⛔ This box has one OS account and several people on it. The defaults below are the SAME
# ones serve.sh and ask.sh use, so two people running ./serve_fg.sh collide on die 0 and
# port 30013. Check `npu-smi info` and pick a free die before you start.
#
# Ctrl-C stops it. Nothing is left behind on the die except the usual 2-3 min of HBM.
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Which die and which NUMA pair. These two must agree: the pair named here is where the
# weights are placed, and the cores pinned below must belong to that same pair. On this
# box the pairs are (0,1) (2,3) (4,5) (6,7) -- cross-pair reads fall off a 7.4x cliff and
# /sys/devices/system/node/*/distance does NOT show it (PLAN.md section 10).
# Die and port are NOT overridden here: they come from glm53_env.sh so that serve.sh,
# serve_fg.sh and ask.sh all mean the same thing by "the default server". This file used
# to default to die 2 / port 30039 against a documented 30013, which is how one directory
# ended up with three different notions of it -- and how ask.sh, whose whole job is to
# talk to this server, pointed at a port nothing was listening on.
case "${1:-}" in
  -h|--help)
    cat <<'USAGE'
usage: serve_fg.sh

One bs=1 streaming + dynamic-hot server in the FOREGROUND, so sglang's own
"Decode batch ... gen throughput" lines land in this terminal. Drive it from another
shell with ask.sh. Ctrl-C stops it.

  GLM53_NPU_DEVICE_ID=<n>  which die  -- CHECK npu-smi FIRST, this box is shared
  GLM53_PORT=<n>           which port
USAGE
    exit 0 ;;
  "") ;;
  *) echo "[serve_fg] unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

export GLM53_KT_NUMA_NODES="${GLM53_KT_NUMA_NODES:-0,1}"

# ⚠ Pin at launch. sglang and the kt threadpool bind nothing on their own -- every thread
# starts with an affinity mask covering all 320 cores -- and page placement is decided by
# first touch during weight load. Pinning after the server is healthy is too late: it
# binds the threads and leaves the weights on whatever nodes they landed on.
# ⛔ KNOWN BUG, not yet fixed: launch-time pinning breaks the kt threadpool on NUMA nodes
# whose cores do not start at 0. Pinning to nodes 6,7 (cores 240-319) kills the scheduler
# during init with a wall of "Core 7 inside NUMA node 6 not found" and
# "Rank 0 scheduler died during initialization (exit code: -6)". kt enumerates the cores
# of a node assuming absolute ids starting at 0, which only holds for nodes 0 and 1.
# Verified working: nodes 0,1 (cores 0-79) and 2,3 (cores 80-159).
# On nodes 4-7, leave GLM53_PIN_CORES EMPTY -- a single unpinned instance still converges
# via the kernel's autonuma; it is only concurrent instances that need the pin.
case " ${GLM53_KT_NUMA_NODES} " in
  *4*|*5*|*6*|*7*)
    if [ -n "${GLM53_PIN_CORES:-}" ]; then
      echo "[serve_fg] ⚠ GLM53_PIN_CORES with nodes ${GLM53_KT_NUMA_NODES} is the known-broken" >&2
      echo "[serve_fg]   combination; unset it or expect the scheduler to die at init." >&2
    fi ;;
esac

if [ -z "${GLM53_PIN_CORES:-}" ] && case " ${GLM53_KT_NUMA_NODES} " in *4*|*5*|*6*|*7*) false;; *) true;; esac; then
  _p=""
  for n in ${GLM53_KT_NUMA_NODES//,/ }; do
    _p="${_p:+$_p,}$(cat /sys/devices/system/node/node$n/cpulist)"
  done
  export GLM53_PIN_CORES="$_p"
fi

export GLM53_PREFILL_STREAM=1        # dynamic hot rides on the streaming prefill path
export GLM53_NUM_GPU_EXPERTS=32      # K = 32 resident experts
export GLM53_THREADPOOL_COUNT=2      # one subpool per node of the pair
export GLM53_CPUINFER=32             # 32 CPU threads
export GLM53_MAX_RUNNING_REQUESTS=1  # bs = 1
export GLM53_LOG_DIR="${GLM53_LOG_DIR:-${GLM53_ARTIFACT_ROOT:-/var/tmp/glm53}/logs-bs1}"
mkdir -p "${GLM53_LOG_DIR}"

# Resolve die/port for the banner the same way serve.sh will.
: "${GLM53_NPU_DEVICE_ID:=0}" "${GLM53_PORT:=30013}"
export GLM53_NPU_DEVICE_ID GLM53_PORT

cat <<BANNER
== bs=1 dynamic-hot server, foreground ==
   die ${GLM53_NPU_DEVICE_ID}   port ${GLM53_PORT}   numa ${GLM53_KT_NUMA_NODES}   cores ${GLM53_PIN_CORES}
   ⛔ shared box: if die ${GLM53_NPU_DEVICE_ID} is not yours, Ctrl-C now and set
      GLM53_NPU_DEVICE_ID / GLM53_PORT. Check with: npu-smi info
   K=${GLM53_NUM_GPU_EXPERTS} resident, streaming prefill on, ${GLM53_CPUINFER} threads

   Weight load takes ~2 min. Wait for:  The server is fired up and ready to roll!
   Then from ANOTHER shell:             ./ask.sh
   Watch here for:                      Decode batch, ... gen throughput (token/s): NN

   That gen-throughput figure on a Decode batch line is already a decode-only rate --
   it is per decode batch, so no prefill is folded into it. Do not compute
   wall_time/tokens from the client instead: that buries TTFT in the number, and on
   the streaming path TTFT is tens of seconds.
BANNER

exec "${_here}/serve.sh" --foreground
