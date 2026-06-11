#!/usr/bin/env python3
"""Fire a decode request and sample system state during it, to localize a slow
decode (I/O-fault vs CPU-bound vs stalled-on-sync).

Samples every ~1.5s: CPU busy%, NVMe read MB/s, scheduler-proc read MB/s, RSS.
Run while the server is up. Usage: profile_decode.py PORT BODY NEW
"""
import json, sys, time, threading, urllib.request, glob, os, re

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8013
BODY = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dynres_body.json"
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 120

def cpu_times():
    with open("/proc/stat") as f:
        p = f.readline().split()[1:]
    v = list(map(int, p))
    idle = v[3] + v[4]  # idle + iowait
    total = sum(v)
    return idle, total

def nvme_read_sectors():
    s = 0
    with open("/proc/diskstats") as f:
        for ln in f:
            c = ln.split()
            if len(c) > 6 and re.match(r"nvme\d+n\d+$", c[2]):
                s += int(c[5])
    return s  # sectors (512B)

def sched_pid():
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cl = open(p, "rb").read().replace(b"\0", b" ").decode("utf8", "ignore")
        except Exception:
            continue
        if "sglang" in cl and ("scheduler" in cl or "launch_server" in cl):
            return int(p.split("/")[2])
    return None

def proc_read_bytes(pid):
    try:
        for ln in open(f"/proc/{pid}/io"):
            if ln.startswith("read_bytes:"):
                return int(ln.split()[1])
    except Exception:
        return None

stop = False
samples = []
def sampler():
    pid = sched_pid()
    i0, t0 = cpu_times()
    n0 = nvme_read_sectors()
    p0 = proc_read_bytes(pid) if pid else None
    tlast = time.time()
    while not stop:
        time.sleep(1.5)
        i1, t1 = cpu_times(); n1 = nvme_read_sectors(); p1 = proc_read_bytes(pid) if pid else None
        dt = time.time() - tlast; tlast = time.time()
        busy = 100.0 * (1 - (i1 - i0) / max(t1 - t0, 1))
        nvme_mbps = (n1 - n0) * 512 / 1e6 / dt
        pmbps = ((p1 - p0) / 1e6 / dt) if (p0 is not None and p1 is not None) else -1
        samples.append((busy, nvme_mbps, pmbps))
        i0, t0, n0, p0 = i1, t1, n1, p1

src = json.load(open(BODY))
req = {"text": src["text"], "sampling_params": {"temperature": 0, "max_new_tokens": NEW}}
th = threading.Thread(target=sampler, daemon=True); th.start()
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{PORT}/generate", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}), timeout=2000)
out = json.load(r)
e2e = time.time() - t0
stop = True; th.join(timeout=3)
m = out["meta_info"]
print(f"e2e={e2e:.1f}s prompt={m['prompt_tokens']} completion={m['completion_tokens']} "
      f"raw_tok/s={m['completion_tokens']/e2e:.2f}")
# drop first 2 samples (prefill), report decode-phase stats
ds = samples[2:] if len(samples) > 4 else samples
if ds:
    import statistics as st
    busy = [s[0] for s in ds]; nv = [s[1] for s in ds]; pr = [s[2] for s in ds]
    print(f"decode-phase samples n={len(ds)}")
    print(f"  CPU busy%%   : med={st.median(busy):.0f}  max={max(busy):.0f}  min={min(busy):.0f}")
    print(f"  NVMe MB/s   : med={st.median(nv):.0f}  max={max(nv):.0f}")
    print(f"  sched rd MB/s: med={st.median(pr):.0f}  max={max(pr):.0f}")
print("TEXT_HEAD:", repr(out["text"][:80]))
