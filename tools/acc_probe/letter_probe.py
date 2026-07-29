#!/usr/bin/env python3
"""Probe top-20 logprobs at the final-answer-letter position for every question.

stdlib-only (runs inside the 910B container too). Output per question: the
top-20 (logprob, token_id) at letter_pos, from which A/B/C/D scores derive.
"""
import argparse, json, os, time, urllib.request


def post(url, payload, timeout):
    req = urllib.request.Request(
        url + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = json.load(open(args.samples))
    out = {"url": args.url, "tag": args.tag,
           "letter_token_ids": data["letter_token_ids"], "results": []}
    done = set()
    if os.path.exists(args.out):  # resume after crash/restart
        try:
            prev = json.load(open(args.out))
            out["results"] = prev["results"]
            done = {r["id"] for r in out["results"]}
            print(f"resuming: {len(done)} already done")
        except Exception:
            pass

    todo = [s for s in data["samples"] if s["id"] not in done]
    for i, s in enumerate(todo):
        t0 = time.time()
        resp = post(args.url, {
            "input_ids": s["input_ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            # 窗口首位置的 lp/top 恒为 None(无上文),提前一位让字母落在 index>=1
            "logprob_start_len": max(0, s["letter_pos"] - 1),
            "top_logprobs_num": 20,
            "return_text_in_logprobs": False,
        }, args.timeout)
        mi = resp["meta_info"]
        itl = mi["input_token_logprobs"]
        itop = mi.get("input_top_logprobs") or []
        # find the entry whose token id == the forced letter token (alignment-proof)
        li = next(j for j, e in enumerate(itl)
                  if e[1] == s["input_ids"][s["letter_pos"]])
        out["results"].append({
            "id": s["id"], "gold": s["gold"], "sampled_pred": s["sampled_pred"],
            "letter_lp": itl[li][0],
            "top20": [(t[0], t[1]) for t in (itop[li] or [])] if li < len(itop) else [],
        })
        json.dump(out, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
        print(f"[{len(done)+i+1}/{len(done)+len(todo)}] {s['id']:6s} n={s['n_total']:5d} {time.time()-t0:6.1f}s",
              flush=True)
    print(f"wrote {args.out} ({len(out['results'])})")


if __name__ == "__main__":
    main()
