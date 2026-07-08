#!/usr/bin/env python3
"""Capture a deterministic temp=0 generation from a running sglang server as a golden.

Sends the FIXED prompt (tools/bitcompare/prompt.txt) to /generate at temperature=0
(greedy argmax), max_new_tokens=N, requesting per-step token ids and top-k logprobs.
Saves everything to a JSON golden. Compare two goldens with compare.py.

The signals captured, strongest first for a bit-level regression:
  * output_ids            : the generated token-id sequence (temp=0 -> deterministic argmax).
                            Exact equality is the top-level pass/fail gate.
  * chosen_logprobs       : logprob of each chosen token (a precise function of the logits).
  * top_logprobs          : top-k (id, logprob) per step; a fine-grained numeric fingerprint.

Usage:
  python capture_client.py --port 8021 --out golden_baseline_boot1.json [--max-new 64]
"""
import argparse
import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--prompt", default=str(HERE / "prompt.txt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = pathlib.Path(args.prompt).read_text(encoding="utf-8")
    req = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new,
        },
        "return_logprob": True,
        "top_logprobs_num": args.top_logprobs,
        # no input logprobs (huge); output-side is what reflects prefill+decode correctness.
        "logprob_start_len": -1,
        "stream": False,
    }
    url = f"http://{args.host}:{args.port}/generate"
    data = json.dumps(req).encode("utf-8")
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}),
        timeout=1800,
    ) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    meta = body["meta_info"]
    otl = meta.get("output_token_logprobs") or []          # [[logprob, token_id, text_or_null], ...]
    otop = meta.get("output_top_logprobs") or []            # [[[lp, id, txt], ...k], ...]
    output_ids = [int(t[1]) for t in otl]
    chosen_logprobs = [float(t[0]) for t in otl]
    top_logprobs = [[[float(e[0]), int(e[1])] for e in step] for step in otop]

    golden = {
        "prompt_chars": len(prompt),
        "prompt_tokens": int(meta.get("prompt_tokens", -1)),
        "completion_tokens": int(meta.get("completion_tokens", -1)),
        "output_text": body.get("text", ""),
        "output_ids": output_ids,
        "chosen_logprobs": chosen_logprobs,
        "top_logprobs": top_logprobs,
        "request": {"max_new_tokens": args.max_new, "temperature": 0.0,
                    "top_logprobs_num": args.top_logprobs},
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"[capture] prompt_tokens={golden['prompt_tokens']} "
          f"completion_tokens={golden['completion_tokens']} -> {out}")
    print(f"[capture] first 16 output_ids: {output_ids[:16]}")
    print(f"[capture] output_text[:200]: {golden['output_text'][:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
