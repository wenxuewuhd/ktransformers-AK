#!/usr/bin/env python3
"""Build a teacher-forcing sample set for 910B vs 910C logprob cross-check.

Reads one evalscope GPQA predictions jsonl (prompt messages + sampled response),
re-renders the exact chat-template token ids locally (tokenizer md5-verified
identical on both boxes), appends the response token ids, and emits a JSON list
of fixed input_ids sequences. Both machines then prefill the *same* token ids,
so the comparison has zero sampling / tokenizer noise.

Also appends a few synthetic long samples (concatenated GPQA Q+A) to probe
whether cross-machine divergence grows with context length (NSA mechanism
diagnosis; real GPQA-off traffic is <= ~3.2k tokens).
"""
import argparse, importlib.util, json, os, random

# Same renderer the server uses for /v1/chat/completions (serving_chat.py):
# encoding_dsv4.encode_messages(thinking_mode="chat") + tokenizer.encode().
_ENC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../third_party/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py",
)
_spec = importlib.util.spec_from_file_location("encoding_dsv4", _ENC)
encoding_dsv4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encoding_dsv4)


def render(tok, messages, response_text):
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    while msgs and msgs[-1]["role"] == "assistant":  # jsonl 里 messages 末尾带了回答
        msgs.pop()
    if msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": ""})
    real_input = encoding_dsv4.encode_messages(
        msgs, thinking_mode="chat", reasoning_effort=None
    )
    prompt_ids = tok.encode(real_input)
    resp_ids = tok(response_text, add_special_tokens=False)["input_ids"]
    return prompt_ids, resp_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4-Flash-W8A8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=6)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []
    for line in open(args.predictions):
        d = json.loads(line)
        ch = d["model_output"]["choices"][0]
        if d["model_output"].get("error") or ch.get("stop_reason") != "stop":
            continue
        rows.append({
            "index": d["index"],
            "messages": d["messages"],
            "response": ch["message"]["content"],
            "usage": d["model_output"]["usage"],
        })

    # stratify by recorded total tokens
    bins = [(0, 400), (400, 700), (700, 1200), (1200, 4000)]
    rng = random.Random(20260728)
    picked = []
    for lo, hi in bins:
        cand = [r for r in rows if lo <= r["usage"]["total_tokens"] < hi]
        rng.shuffle(cand)
        picked += [(f"bin{lo}", r) for r in cand[: args.per_bin]]

    samples, n_mismatch = [], 0
    for tag, r in picked:
        p_ids, r_ids = render(tok, r["messages"], r["response"])
        rec_in = r["usage"]["input_tokens"]
        if abs(len(p_ids) - rec_in) > 2:
            n_mismatch += 1
            print(f"  !! idx={r['index']} rendered prompt {len(p_ids)} vs recorded {rec_in}")
        samples.append({
            "id": f"{tag}_idx{r['index']}",
            "source_index": r["index"],
            "n_prompt": len(p_ids),
            "n_total": len(p_ids) + len(r_ids),
            "input_ids": p_ids + r_ids,
        })

    # synthetic long: concatenate many Q+A into one user turn + one forced answer
    rows_sorted = sorted(rows, key=lambda r: -r["usage"]["total_tokens"])
    for want, name in [(4500, "syn4k"), (9000, "syn8k")]:
        q_parts, a_parts, tot = [], [], 0
        for r in rows_sorted:
            q = next(m["content"] for m in r["messages"] if m["role"] == "user")
            q_parts.append(q); a_parts.append(r["response"])
            tot += r["usage"]["total_tokens"]
            if tot >= want:
                break
        msgs = [{"role": "user", "content": "\n\n----- NEXT QUESTION -----\n\n".join(q_parts)}]
        p_ids, r_ids = render(tok, msgs, "\n\n".join(a_parts))
        samples.append({
            "id": name, "source_index": -1,
            "n_prompt": len(p_ids), "n_total": len(p_ids) + len(r_ids),
            "input_ids": p_ids + r_ids,
        })

    samples.sort(key=lambda s: s["n_total"])
    json.dump(samples, open(args.out, "w"))
    print(f"wrote {len(samples)} samples -> {args.out}  (prompt-len mismatches: {n_mismatch})")
    for s in samples:
        print(f"  {s['id']:16s} prompt={s['n_prompt']:5d} total={s['n_total']:5d}")


if __name__ == "__main__":
    main()
