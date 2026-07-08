#!/usr/bin/env python3
"""Compare two capture goldens for the bit-level regression.

Acceptance tiers (decided empirically in Phase 0 from baseline cross-boot variance):
  * Tier A (bit-exact): output_ids identical AND every logprob bit-identical.
  * Tier B (noise-band): output_ids identical AND max |Δ logprob| <= --logprob-tol.

Exit code 0 = pass at the requested tier, 1 = fail. Prints a diff summary either way.

Usage:
  python compare.py --ref golden_baseline.json --new golden_new.json --tier B --logprob-tol 5e-3
"""
import argparse
import json
import math
import pathlib
import sys


def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def max_abs_logprob_diff(ref, new):
    worst = 0.0
    worst_at = None
    # chosen-token logprobs
    for i, (a, b) in enumerate(zip(ref["chosen_logprobs"], new["chosen_logprobs"])):
        d = abs(a - b)
        if d > worst:
            worst, worst_at = d, f"chosen[{i}]"
    # top-k logprobs (only where the top-k token ids line up; otherwise reported via id diff)
    for i, (sa, sb) in enumerate(zip(ref["top_logprobs"], new["top_logprobs"])):
        amap = {tid: lp for lp, tid in sa}
        for lp, tid in sb:
            if tid in amap:
                d = abs(amap[tid] - lp)
                if d > worst:
                    worst, worst_at = d, f"top[{i}] id={tid}"
    return worst, worst_at


def first_id_divergence(ref, new):
    for i, (a, b) in enumerate(zip(ref["output_ids"], new["output_ids"])):
        if a != b:
            return i, a, b
    if len(ref["output_ids"]) != len(new["output_ids"]):
        return min(len(ref["output_ids"]), len(new["output_ids"])), None, None
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--tier", choices=["A", "B"], default="B")
    ap.add_argument("--logprob-tol", type=float, default=5e-3)
    args = ap.parse_args()

    ref, new = load(args.ref), load(args.new)

    ids_equal = ref["output_ids"] == new["output_ids"]
    div_at, ra, rb = first_id_divergence(ref, new)
    worst, worst_at = max_abs_logprob_diff(ref, new)

    print(f"[compare] ref prompt_tokens={ref.get('prompt_tokens')} "
          f"new prompt_tokens={new.get('prompt_tokens')}")
    print(f"[compare] output_ids equal: {ids_equal} "
          f"(len ref={len(ref['output_ids'])} new={len(new['output_ids'])})")
    if not ids_equal:
        print(f"[compare]   first divergence at step {div_at}: ref={ra} new={rb}")
    print(f"[compare] max |Δ logprob| = {worst:.3e} at {worst_at}")

    if args.tier == "A":
        ok = ids_equal and worst == 0.0
        print(f"[compare] TIER A (bit-exact): {'PASS' if ok else 'FAIL'}")
    else:
        ok = ids_equal and worst <= args.logprob_tol
        print(f"[compare] TIER B (tol={args.logprob_tol:.1e}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
