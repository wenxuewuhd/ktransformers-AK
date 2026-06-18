#!/usr/bin/env python3
"""Paired A/B decode-speed measurement under identical (alternating) contention.

Alternates decode requests between two server ports (e.g. streaming vs no-stream),
short prompt so prefill is cheap and decode dominates. Reports per-round decode
tok/s for each and the paired ratio, plus global CPU busy% at each request.
Usage: paired_decode.py PORT_A PORT_B ROUNDS NEW
"""
import json, sys, time, urllib.request

PA = int(sys.argv[1]); PB = int(sys.argv[2])
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 100
BODY = "/tmp/short_body.json"
src = json.load(open(BODY))

def cpu_busy(dt=0.0):
    def snap():
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        return v[3] + v[4], sum(v)
    a = snap(); time.sleep(0.4); b = snap()
    idle = b[0] - a[0]; tot = b[1] - a[1]
    return 100.0 * (1 - idle / max(tot, 1))

def gen(port):
    req = {"text": src["text"], "sampling_params": {"temperature": 0, "max_new_tokens": NEW}}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=2000)
    out = json.load(r); e2e = time.time() - t0
    m = out["meta_info"]
    return e2e, m["prompt_tokens"], m["completion_tokens"]

# warm both (first call captures graph / pays one-time costs)
for p in (PA, PB):
    try: gen(p)
    except Exception as e: print(f"warm {p} err {e}")

PREFILL_EST = 3.0  # short prompt prefill estimate (s); decode = comp/(e2e-PREFILL_EST)
print(f"{'round':>5} {'cpu%':>5} {'A_tok/s':>8} {'B_tok/s':>8} {'A/B':>5}")
ra, rb = [], []
for k in range(ROUNDS):
    busy = cpu_busy()
    ea, pa, ca = gen(PA)
    eb, pb, cb = gen(PB)
    da = ca / max(ea - PREFILL_EST, 0.1)
    db = cb / max(eb - PREFILL_EST, 0.1)
    ra.append(da); rb.append(db)
    print(f"{k:>5} {busy:>5.0f} {da:>8.2f} {db:>8.2f} {da/db:>5.2f}")
import statistics as st
print(f"median A(port {PA})={st.median(ra):.2f}  B(port {PB})={st.median(rb):.2f}  "
      f"ratio A/B median={st.median([a/b for a,b in zip(ra,rb)]):.2f}")
