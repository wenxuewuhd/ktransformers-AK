#!/usr/bin/env python3
"""Verdict analysis: same-CoT greedy answer accuracy per machine.

For each question, each machine's greedy letter = argmax over {A,B,C,D} of the
top-20 logprobs at the final-answer-letter position (teacher-forced identical
CoT). Deterministic, zero sampling noise. Also: letter-level flips between
machines, McNemar on gold-correctness, and per-letter logprob deltas.
"""
import argparse, json, math


def greedy_letter(rec, letter_ids):
    inv = {v: k for k, v in letter_ids.items()}
    best, best_lp = None, -1e30
    for lp, tid in rec["top20"]:
        L = inv.get(tid)
        if L is not None and lp > best_lp:
            best, best_lp = L, lp
    return best


def letter_lps(rec, letter_ids):
    inv = {v: k for k, v in letter_ids.items()}
    out = {}
    for lp, tid in rec["top20"]:
        L = inv.get(tid)
        if L is not None:
            out[L] = lp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="910C"); ap.add_argument("--label-b", default="910B")
    args = ap.parse_args()

    da, db = json.load(open(args.a)), json.load(open(args.b))
    lids = {k: v for k, v in da["letter_token_ids"].items()}
    A = {r["id"]: r for r in da["results"]}
    B = {r["id"]: r for r in db["results"]}
    ids = [i for i in A if i in B]

    accA = accB = accS = flips = 0
    a_only = b_only = 0   # McNemar cells: A correct B wrong / A wrong B correct
    flip_list = []
    margin_deltas = []
    for i in ids:
        ra, rb = A[i], B[i]
        ga = greedy_letter(ra, lids); gb = greedy_letter(rb, lids)
        gold = ra["gold"]
        accA += ga == gold; accB += gb == gold; accS += ra["sampled_pred"] == gold
        if ga != gb:
            flips += 1
            flip_list.append((i, gold, ra["sampled_pred"], ga, gb,
                              letter_lps(ra, lids), letter_lps(rb, lids)))
        if (ga == gold) and (gb != gold): a_only += 1
        if (ga != gold) and (gb == gold): b_only += 1
        # decision margin: top letter lp - runner-up letter lp, per machine
        la, lb = letter_lps(ra, lids), letter_lps(rb, lids)
        if len(la) >= 2 and len(lb) >= 2:
            sa = sorted(la.values(), reverse=True); sb = sorted(lb.values(), reverse=True)
            margin_deltas.append((sa[0] - sa[1], sb[0] - sb[1]))

    n = len(ids)
    print(f"n={n}  (forced CoT = 910C-R1 sampled responses)")
    print(f"greedy-letter accuracy: {args.label_a}={accA/n:.4f} ({accA})   "
          f"{args.label_b}={accB/n:.4f} ({accB})   [sampled R1 ref={accS/n:.4f}]")
    print(f"letter flips between machines: {flips}/{n} ({100*flips/n:.1f}%)")
    print(f"McNemar: {args.label_a}-only-correct={a_only}  {args.label_b}-only-correct={b_only}")
    if a_only + b_only:
        # exact binomial two-sided p
        k, m = min(a_only, b_only), a_only + b_only
        p = sum(math.comb(m, j) for j in range(0, k + 1)) / 2**m * 2
        print(f"         two-sided binomial p = {min(p,1):.3f}")
    if margin_deltas:
        ma = sum(x[0] for x in margin_deltas)/len(margin_deltas)
        mb = sum(x[1] for x in margin_deltas)/len(margin_deltas)
        print(f"decision margin (top-runnerup): {args.label_a}={ma:.3f}  {args.label_b}={mb:.3f}")
    if flip_list:
        print("\nflipped questions (id, gold, R1-sampled, greedyA, greedyB, lpA, lpB):")
        for f in flip_list:
            la = {k: round(v, 3) for k, v in f[5].items()}
            lb = {k: round(v, 3) for k, v in f[6].items()}
            print(f"  {f[0]:6s} gold={f[1]} R1={f[2]}  {args.label_a}:{f[3]} {args.label_b}:{f[4]}")
            print(f"         lp{args.label_a}={la}")
            print(f"         lp{args.label_b}={lb}")


if __name__ == "__main__":
    main()
