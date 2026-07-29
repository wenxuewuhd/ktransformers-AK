#!/usr/bin/env python3
"""Build the full-198 'final answer letter' sample set.

For every GPQA question: prompt + R1's full CoT response (teacher-forced), plus
the token index of the final answer letter (the X in trailing "ANSWER: X").
letter_probe.py then asks each machine for top-20 logprobs at that position ->
deterministic same-CoT answer accuracy per machine, zero sampling noise.
"""
import argparse, json, re
from build_samples import render  # same server-faithful renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4-Flash-W8A8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    gold = {}
    for line in open(args.reviews):
        d = json.loads(line)
        gold[d["index"]] = d["target"]

    letter_ids = {L: tok(f" {L}", add_special_tokens=False)["input_ids"] for L in "ABCD"}
    assert all(len(v) == 1 for v in letter_ids.values()), letter_ids
    letter_ids = {L: v[0] for L, v in letter_ids.items()}

    samples, skipped = [], []
    for line in open(args.predictions):
        d = json.loads(line)
        ch = d["model_output"]["choices"][0]
        resp = ch["message"]["content"]
        m = list(re.finditer(r"ANSWER:\s*([A-D])\b", resp))
        if not m or ch.get("stop_reason") != "stop":
            skipped.append(d["index"]); continue
        letter_char = m[-1].start(1)

        p_ids, r_ids = render(tok, d["messages"], resp)
        enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == r_ids
        tok_idx = next(i for i, (a, b) in enumerate(enc["offset_mapping"])
                       if a <= letter_char < b)
        full = p_ids + r_ids
        letter_pos = len(p_ids) + tok_idx
        pred = resp[letter_char]
        if full[letter_pos] != letter_ids[pred]:
            skipped.append(d["index"]); continue  # letter not a clean single token here
        samples.append({
            "id": f"q{d['index']}",
            "source_index": d["index"],
            "n_prompt": len(p_ids),
            "n_total": len(full),
            "letter_pos": letter_pos,
            "gold": gold[d["index"]],
            "sampled_pred": pred,
            "input_ids": full,
        })

    out = {"letter_token_ids": letter_ids, "samples": samples}
    json.dump(out, open(args.out, "w"))
    acc = sum(s["gold"] == s["sampled_pred"] for s in samples) / len(samples)
    print(f"wrote {len(samples)} samples (skipped {len(skipped)}: {skipped})")
    print(f"sanity: sampled-R1 accuracy on kept set = {acc:.4f}")


if __name__ == "__main__":
    main()
