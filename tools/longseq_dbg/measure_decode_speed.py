#!/usr/bin/env python3
"""Measure steady-state decode tokens/s for the running sglang server.

Sends a long prompt (triggers streaming prefill + any dynamic residency), then
generates a large fixed number of decode tokens. Reports:
  - client e2e latency, prefill vs decode split (decode tok/s from completion/decode_time)
  - server-reported decode tok/s (parsed by caller from the server log)

Decode SPEED is independent of output quality (a repetition loop decodes at the
same tok/s), so this is valid even when the resident set degrades the text.
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8013
BODY = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dynres_body.json"
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 256

src = json.load(open(BODY))
text = src.get("text")
req = {
    "text": text,
    "sampling_params": {"temperature": 0, "max_new_tokens": NEW},
}
data = json.dumps(req).encode()

# warm: one short call to make sure graphs are captured (separate request)
t0 = time.time()
r = urllib.request.urlopen(
    urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate", data=data,
        headers={"Content-Type": "application/json"}), timeout=1200)
out = json.load(r)
t1 = time.time()
m = out["meta_info"]
comp = m.get("completion_tokens")
prompt = m.get("prompt_tokens")
e2e = t1 - t0
print(f"prompt_tokens={prompt} completion_tokens={comp} e2e={e2e:.2f}s")
# decode tok/s lower bound (includes prefill in e2e): comp / (e2e - prefill_est)
# prefill on streaming ~12s fixed; report both raw and prefill-adjusted
print(f"raw_tok_per_s(incl prefill) = {comp / e2e:.2f}")
for pf in (10, 12, 14):
    dt = e2e - pf
    if dt > 0:
        print(f"  decode_tok_per_s(if prefill={pf}s) = {comp / dt:.2f}")
print("TEXT_HEAD:", repr(out["text"][:120]))
