#!/usr/bin/env python3
"""Extra synthetic long samples (3k-7.8k, all < 8192 single-chunk) to refine the
|dlogprob|-vs-length curve: does cross-machine divergence grow with context
(NSA compressor mechanism), sampled densely below the 2-chunk crash boundary."""
import argparse, json
from build_samples import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4-Flash-W8A8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []
    for line in open(args.predictions):
        d = json.loads(line)
        ch = d["model_output"]["choices"][0]
        if ch.get("stop_reason") != "stop":
            continue
        rows.append({
            "q": next(m["content"] for m in d["messages"] if m["role"] == "user"),
            "a": ch["message"]["content"],
            "t": d["model_output"]["usage"]["total_tokens"],
        })
    rows.sort(key=lambda r: -r["t"])

    samples = []
    for want, name in [(3000, "syn3k"), (4500, "syn45"), (5500, "syn55"),
                       (6500, "syn65"), (7700, "syn77")]:
        qp, ap_, tot = [], [], 0
        for r in rows[len(samples):]:  # 每个样本换一批题,减少内容重复
            qp.append(r["q"]); ap_.append(r["a"]); tot += r["t"]
            if tot >= want - 300:
                break
        msgs = [{"role": "user", "content": "\n\n----- NEXT QUESTION -----\n\n".join(qp)}]
        p_ids, r_ids = render(tok, msgs, "\n\n".join(ap_))
        full = p_ids + r_ids
        if len(full) > 8100:  # 保守截断,避开 2-chunk 崩溃边界
            full = full[:8100]
        samples.append({"id": name, "source_index": -1, "n_prompt": len(p_ids),
                        "n_total": len(full), "input_ids": full})
        print(f"{name}: total={len(full)}")
    json.dump(samples, open(args.out, "w"))
    print(f"wrote {len(samples)} -> {args.out}")


if __name__ == "__main__":
    main()
