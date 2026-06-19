#!/usr/bin/env bash
# side-stream / shared-experts-stream 性能探针：固定短 prompt，量 decode TPOT。
# 每个配置一次 boot，跑 REPS 轮 {prefill 探针(max_new=1) + 全量(MAX_NEW)}，
# 每轮 decode ms/tok = (e2e_full - e2e_prefill)/(completion-1)。报 min/median/mean。
# side-stream 收益体现在 TPOT 地板(min/p10)，故重点看 min/median。
#
# 用法（服务已起，PORT 对齐）：
#   TAG=off  PORT=8200 bash tools/p27_sidestream_perf.sh   # 基线 boot(side=0 shared=0)
#   TAG=on   PORT=8200 bash tools/p27_sidestream_perf.sh   # 特性 boot(side=1 shared=1)
# 结果落 /tmp/sidestream_perf_${TAG}.txt，两 boot 后对比：CMP=1 bash tools/p27_sidestream_perf.sh
set -euo pipefail
PORT="${PORT:-8200}"; HOST="${HOST:-127.0.0.1}"
TAG="${TAG:-run}"; REPS="${REPS:-6}"; MAX_NEW="${MAX_NEW:-256}"
PY="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
OUT="/tmp/sidestream_perf_${TAG}.txt"

if [[ "${CMP:-0}" == "1" ]]; then
  "$PY" - <<'PY'
import os, json
def load(t):
    p=f"/tmp/sidestream_perf_{t}.txt"
    return json.load(open(p)) if os.path.exists(p) else None
off, on = load("off"), load("on")
if not off or not on:
    print("[cmp] 缺 off 或 on 结果，先各 boot 跑一次"); raise SystemExit
print("="*64)
print(f"{'metric':>16} {'OFF':>12} {'ON':>12} {'gain':>10}")
print("-"*64)
for k in ("min","median","mean"):
    a,b=off[k],on[k]; g=(a-b)/a*100
    print(f"{k+' ms/tok':>16} {a:>12.2f} {b:>12.2f} {g:>9.1f}%")
print(f"{'min tok/s':>16} {1000/off['min']:>12.2f} {1000/on['min']:>12.2f}")
print("="*64)
print("正收益 = ON 的 ms/tok 更小（gain>0）。side-stream 看 min/median。")
PY
  exit 0
fi

curl -sf -m5 "http://${HOST}:${PORT}/health" >/dev/null || { echo "[perf][ERR] 服务没起 ${HOST}:${PORT}"; exit 1; }
PROMPT="Explain in detail how a transformer neural network processes a sequence of tokens, step by step."
RQ1=/tmp/ssperf_rq1.json; RQ2=/tmp/ssperf_rq2.json
"$PY" - "$PROMPT" "$MAX_NEW" "$RQ1" "$RQ2" <<'PY'
import sys,json
p,mn,rq1,rq2=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
json.dump({"text":p,"sampling_params":{"max_new_tokens":1,"temperature":0}},open(rq1,"w"))
json.dump({"text":p,"sampling_params":{"max_new_tokens":mn,"temperature":0,"ignore_eos":True}},open(rq2,"w"))
PY

echo "[perf][$TAG] warmup 2x ..."
for i in 1 2; do curl -s -m600 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' -d @"$RQ2" -o /dev/null; done

echo "[perf][$TAG] timing ${REPS} reps (MAX_NEW=${MAX_NEW}) ..."
: > /tmp/ssperf_samples.txt
for r in $(seq 1 "$REPS"); do
  A=$(curl -s -m600 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' -d @"$RQ1")
  C=$(curl -s -m600 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' -d @"$RQ2")
  echo "$A" > /tmp/ssperf_a.json; echo "$C" > /tmp/ssperf_c.json
  "$PY" - /tmp/ssperf_a.json /tmp/ssperf_c.json >> /tmp/ssperf_samples.txt <<'PY'
import sys,json
a=json.load(open(sys.argv[1]))["meta_info"]; c=json.load(open(sys.argv[2]))["meta_info"]
e1=a.get("e2e_latency"); e3=c.get("e2e_latency"); nc=c.get("completion_tokens")
if e1 and e3 and nc and nc>1:
    print(f"{(e3-e1)/(nc-1)*1000:.4f}")
PY
  printf "  rep %d: %s ms/tok\n" "$r" "$(tail -1 /tmp/ssperf_samples.txt)"
done

TAG="$TAG" OUT="$OUT" "$PY" - <<'PY'
import os,statistics,json
xs=[float(l) for l in open("/tmp/ssperf_samples.txt") if l.strip()]
xs.sort()
res={"tag":os.environ["TAG"],"n":len(xs),"min":xs[0],"median":statistics.median(xs),
     "mean":statistics.mean(xs),"max":xs[-1],"samples":xs}
json.dump(res,open(os.environ["OUT"],"w"))
print("="*56)
print(f"[perf][{res['tag']}] n={res['n']}  min={res['min']:.2f}  "
      f"median={res['median']:.2f}  mean={res['mean']:.2f} ms/tok")
print(f"           min={1000/res['min']:.2f} tok/s  median={1000/res['median']:.2f} tok/s")
print(f"  -> {os.environ['OUT']}")
print("="*56)
PY
