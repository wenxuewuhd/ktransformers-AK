#!/usr/bin/env python3
"""Teacher-forced logprob probe: prefill fixed input_ids, dump per-token logprobs.

Pure prefill (max_new_tokens=1, temperature=0), bs=1 serial, fixed order ->
bit-stable on a freshly restarted server. Run the SAME samples file against the
910C server and (via ssh -L tunnel) the 910B server, then diff the outputs.
"""
import argparse, json, os, time, urllib.request


def post(url, payload, timeout):
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8021")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=2, help="also record top-k ids per position")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-len", type=int, default=0,
                    help="skip samples with n_total > this (0 = no limit); "
                         ">8192 (2-chunk prefill) crashes NSA single-mode compress buffer")
    ap.add_argument("--ids", default="", help="comma-separated sample ids to run (subset)")
    args = ap.parse_args()

    samples = json.load(open(args.samples))
    if args.ids:
        want = set(args.ids.split(","))
        samples = [s for s in samples if s["id"] in want]
    if args.max_len:
        skipped = [s["id"] for s in samples if s["n_total"] > args.max_len]
        samples = [s for s in samples if s["n_total"] <= args.max_len]
        if skipped:
            print(f"skipping {skipped} (n_total > {args.max_len})")
    out = {"url": args.url, "tag": args.tag, "results": []}
    for i, s in enumerate(samples):
        t0 = time.time()
        resp = post(args.url, {
            "input_ids": s["input_ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": args.top_k,
            "return_text_in_logprobs": False,
        }, args.timeout)
        mi = resp["meta_info"]
        itl = mi["input_token_logprobs"]  # [(logprob, token_id, None), ...]
        rec = {
            "id": s["id"],
            "n_prompt": s["n_prompt"],
            "n_total": s["n_total"],
            "logprobs": [x[0] for x in itl],
            "token_ids": [x[1] for x in itl],
        }
        if args.top_k:
            # per position: list of top-k (logprob, token_id)
            rec["top"] = [
                [(t[0], t[1]) for t in pos] if pos else []
                for pos in (mi.get("input_top_logprobs") or [])
            ]
        out["results"].append(rec)
        json.dump(out, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)  # 逐条落盘,崩溃不丢已跑样本
        print(f"[{i+1}/{len(samples)}] {s['id']:16s} n={s['n_total']:5d}  {time.time()-t0:6.1f}s", flush=True)

    print(f"wrote {args.out} ({len(out['results'])} samples)")


if __name__ == "__main__":
    main()
