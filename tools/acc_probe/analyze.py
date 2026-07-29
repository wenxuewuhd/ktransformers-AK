#!/usr/bin/env python3
"""Diff two probe outputs (910C vs 910B): per-token logprob divergence.

Reads probe.py outputs A and B (same samples file on both), reports per-sample
divergence stats and the |dlogprob| vs position trend (does it grow with
context length -> attention/NSA suspicion; flat small floor -> FP background).
"""
import argparse, json, math


def load(path):
    d = json.load(open(path))
    return {r["id"]: r for r in d["results"]}, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    A, da = load(args.a); B, db = load(args.b)
    ids = [i for i in A if i in B]
    print(f"{args.label_a}={da['url']}  {args.label_b}={db['url']}  samples={len(ids)}\n")
    hdr = f"{'sample':16s} {'ntok':>5s}  {'mean|d|':>8s} {'p99|d|':>8s} {'max|d|':>8s}  {'top1flip%':>9s}  {'1st半':>8s} {'2nd半':>8s}"
    print(hdr)

    rows = []
    for sid in ids:
        ra, rb = A[sid], B[sid]
        assert ra["token_ids"] == rb["token_ids"], f"{sid}: token stream mismatch!"
        la, lb = ra["logprobs"], rb["logprobs"]
        n = min(len(la), len(lb))
        d = [abs(la[i] - lb[i]) for i in range(n)
             if la[i] is not None and lb[i] is not None]
        pos = [i for i in range(n) if la[i] is not None and lb[i] is not None]
        d_sorted = sorted(d)
        mean = sum(d) / len(d)
        p99 = d_sorted[int(0.99 * (len(d) - 1))]
        mx = d_sorted[-1]
        half = len(d) // 2
        m1 = sum(d[:half]) / half
        m2 = sum(d[half:]) / (len(d) - half)

        flips = tot = 0
        ta, tb = ra.get("top") or [], rb.get("top") or []
        for i in range(min(len(ta), len(tb))):
            if ta[i] and tb[i]:
                tot += 1
                if ta[i][0][1] != tb[i][0][1]:
                    flips += 1
        fl = 100.0 * flips / tot if tot else float("nan")

        rows.append(dict(id=sid, n=n, mean=mean, p99=p99, max=mx,
                         flip_pct=fl, first_half=m1, second_half=m2,
                         d=d, pos=pos))
        print(f"{sid:16s} {n:5d}  {mean:8.4f} {p99:8.4f} {mx:8.4f}  {fl:8.2f}%  {m1:8.4f} {m2:8.4f}")

    # global position trend: bucket by absolute position
    buckets = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 10000)]
    print("\n|dlogprob| by absolute position (all samples pooled):")
    for lo, hi in buckets:
        vals = [dv for r in rows for p, dv in zip(r["pos"], r["d"]) if lo <= p < hi]
        if vals:
            print(f"  pos [{lo:5d},{hi:5d}): n={len(vals):7d}  mean={sum(vals)/len(vals):8.4f}  "
                  f"p99={sorted(vals)[int(0.99*(len(vals)-1))]:8.4f}")

    if args.json_out:
        for r in rows:
            r.pop("d"); r.pop("pos")
        json.dump(rows, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
