#!/usr/bin/env python3
"""单卡 DSV4-Flash decode timing 实测。

用 >512 token 的有意义 prompt(触发 KT_PREFILL_STREAM_THRESHOLD=512 流式 prefill +
dynamic 热专家命中),量 prefill TTFT / decode TPOT / tok-s / inter-token 中位。
默认跑 2 发:第 1 发预热(填 KV/触发流式),第 2 发才是稳态 decode 读数。

用法:
    python3 tools/p27_decode_timing.py                 # 默认 8020, max_tokens=400, 2 发
    python3 tools/p27_decode_timing.py --port 8020 --max-tokens 512 --runs 3
    python3 tools/p27_decode_timing.py --big            # prompt 加长到 ~4K token
仅只读打请求,不动服务;会先等 /health=200(最多 20 分钟)。
"""
import argparse, json, os, time, urllib.request, urllib.error

os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)

BASE_CTX = """You are analyzing a single-card MoE inference system. The model is a 43-layer
Mixture-of-Experts transformer with 256 routed experts per layer and top-6 routing, deployed on
a single Ascend 910C(A3) NPU with 61 GB of HBM. Because the full expert weights (about 274 GB in
W8A8) cannot fit in HBM, only 32 "hot" experts per layer are kept resident on the NPU while the
remaining experts are offloaded to CPU DRAM on a 192-core Kunpeng-920 host and computed with an
INT8/MXFP4 kernel. During decode (batch size 1), each generated token activates six experts per
layer; whichever of those six are not resident on the NPU must be fetched from CPU memory and
executed on the CPU, then their outputs are streamed back to the NPU and combined with the
attention output. Measurements show the CPU-side MoE computation is memory-bandwidth bound: the
arithmetic intensity of a batch-1 matrix-vector product is only a few operations per byte, far
below the machine's roofline ridge point, so per-token latency is dominated by how fast expert
weights stream out of DRAM rather than by raw compute. The NPU side handles attention (a Multi-head
Latent Attention variant with a native sparse attention indexer that caps each token's attention
span at 512 selected keys), the router, the shared expert, and the 32 resident experts. The two
sides are pipelined so CPU expert computation for a layer overlaps with NPU work for that layer."""

QUESTION = """

Given this architecture, explain precisely and quantitatively: (1) why speeding up only the NPU
side yields diminishing returns for time-per-output-token once NPU time drops below the CPU-MoE
wall time; (2) which concrete levers actually reduce the CPU-MoE wall time and why each works in
terms of bytes-moved or effective bandwidth; (3) under what condition upgrading the NPU becomes
worthwhile again. Then summarize the whole reasoning in three bullet points."""


def build_prompt(big):
    ctx = BASE_CTX
    if big:
        # 复述背景多份把 prompt 抬到 ~4K token,仍是连贯文本
        ctx = "\n\n".join(
            f"[Section {i+1}] " + BASE_CTX for i in range(5)
        )
    return ctx + QUESTION


def wait_health(url_base, timeout_s=1200):
    hurl = url_base + "/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(hurl)
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        print(f"  [{time.strftime('%H:%M:%S')}] 等 {hurl} …")
        time.sleep(5)
    return False


def one_run(api, model, prompt, max_tokens, tag):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"chat_template_kwargs": {"thinking": False, "high_effort": False}},
    }
    req = urllib.request.Request(
        api, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.time(); t_first = None; last = None; inter = []; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices")
            if ch:
                c = ch[0].get("delta", {}).get("content")
                if c:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    else:
                        inter.append(now - last)
                    last = now; text += c
    t_end = time.time()
    pt = usage.get("prompt_tokens") if usage else None
    ct = usage.get("completion_tokens") if usage else len(inter) + 1
    prefill = (t_first - t0) if t_first else float("nan")
    dwall = (t_end - t_first) if t_first else float("nan")
    print(f"===== {tag} =====")
    print(f"  prompt_tokens     = {pt}   (>512 触发流式 prefill)")
    print(f"  completion_tokens = {ct}")
    print(f"  [PREFILL] TTFT = {prefill*1000:8.0f} ms  => {(pt/prefill) if pt else 0:7.0f} tok/s")
    print(f"  [DECODE ] TPOT = {(dwall/ct*1000):8.2f} ms/tok => {ct/dwall:7.2f} tok/s "
          f"(wall {dwall:.2f}s / {ct} tok)")
    if inter:
        s = sorted(inter); med = s[len(s)//2]
        print(f"  [DECODE ] inter-token 中位 {med*1000:6.2f} ms | p10 {s[len(s)//10]*1000:.1f} "
              f"| p90 {s[len(s)*9//10]*1000:.1f} | min {s[0]*1000:.1f} | max {s[-1]*1000:.1f}")
    print(f"  output[:80] = {text[:80]!r}")
    return ct / dwall if dwall == dwall else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4-Flash-W8A8")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--runs", type=int, default=2, help="总发数;第1发预热,其余取稳态")
    ap.add_argument("--big", action="store_true", help="prompt 抬到 ~4K token")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    api = base + "/v1/chat/completions"
    prompt = build_prompt(args.big)
    print(f"# 目标 {api}  model={args.model}")
    print(f"# 等服务健康 …")
    if not wait_health(base):
        print("!! 20 分钟内 /health 未就绪,退出"); return
    print("# 服务就绪,开始打点\n")

    steady = []
    for i in range(args.runs):
        tag = f"RUN {i+1}/{args.runs}" + ("  (预热,不计)" if i == 0 and args.runs > 1 else "")
        tps = one_run(api, args.model, prompt, args.max_tokens, tag)
        if not (i == 0 and args.runs > 1):
            steady.append(tps)
        print()
    if steady:
        print(f"====== 稳态 decode: {sum(steady)/len(steady):.2f} tok/s "
              f"(min {min(steady):.2f} / max {max(steady):.2f}, n={len(steady)}) ======")
    print("提示:同时在服务端 tmux 看 `gen throughput (token/s)` 交叉验证。")


if __name__ == "__main__":
    main()
