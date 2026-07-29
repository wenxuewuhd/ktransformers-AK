#!/usr/bin/env python3
"""Within-machine causal A/B: the 15 bracket-affected GPQA questions, corrupted
(evalscope 1.8.1 choice-preprocess) vs clean (1.9.0) prompt variants, N samples
each at temp=1 on the SAME box. Quantifies the net causal effect of the choice
corruption with zero cross-machine confounds.

Clean prompts come from a 1.9.0 predictions file (byte-identical across boxes,
verified); the corrupted variant re-applies the exact 1.8.1 transform to the
choice lines only (matches how 1.8.1 built prompts: question text untouched).
"""
import argparse, glob, importlib.util, json, os, random, re, time, urllib.request

_ENC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../third_party/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py",
)
_spec = importlib.util.spec_from_file_location("encoding_dsv4", _ENC)
encoding_dsv4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encoding_dsv4)

AFF = [8, 24, 27, 32, 55, 64, 76, 77, 81, 97, 118, 156, 171, 177, 185]


def corrupt_choice(t):  # evalscope 1.8.1 gpqa_adapter.preprocess
    t = t.strip()
    t = t.replace(' [title]', '. ')
    t = re.sub(r'\[.*?\]', '', t)
    t = t.replace('  ', ' ')
    return t


def corrupt_prompt(q):
    out = []
    for ln in q.split('\n'):
        m = re.match(r'^([A-D]\) )(.*)$', ln)
        out.append(m.group(1) + corrupt_choice(m.group(2)) if m else ln)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--pred", required=True, help="a 1.9.0 predictions jsonl (clean prompts)")
    ap.add_argument("--reviews", required=True, help="matching reviews jsonl (gold letters)")
    ap.add_argument("--n", type=int, default=12, help="samples per question per variant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args()

    gold, clean = {}, {}
    for line in open(args.reviews):
        d = json.loads(line)
        if d["index"] in AFF:
            gold[d["index"]] = d["target"]
    for line in open(args.pred):
        d = json.loads(line)
        if d["index"] in AFF:
            clean[d["index"]] = next(
                m["content"] for m in d["messages"] if m["role"] == "user")

    jobs = []
    for i in AFF:
        for v, prompt in (("clean", clean[i]), ("corrupt", corrupt_prompt(clean[i]))):
            for k in range(args.n):
                jobs.append((i, v, k, prompt))
    random.Random(7).shuffle(jobs)  # interleave variants/questions against drift

    res = {}
    if os.path.exists(args.out):
        try:
            res = {tuple(k.split("|")[:3]): v
                   for k, v in json.load(open(args.out))["raw"].items()}
            print(f"resuming with {len(res)} done")
        except Exception:
            pass

    for n_done, (i, v, k, prompt) in enumerate(jobs):
        key = (str(i), v, str(k))
        if key in res:
            continue
        real_input = encoding_dsv4.encode_messages(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            thinking_mode="chat", reasoning_effort=None,
        )
        payload = {
            "text": real_input,
            "sampling_params": {"temperature": 1, "top_p": 1, "max_new_tokens": 8192},
        }
        t0 = time.time()
        req = urllib.request.Request(args.url + "/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=args.timeout).read())
            text = r["text"]
        except Exception as e:
            print(f"!! q{i} {v}#{k}: {e}", flush=True)
            continue
        m = list(re.finditer(r"ANSWER:\s*([A-D])\b", text))
        pred = m[-1].group(1) if m else None
        res[key] = {"pred": pred, "acc": int(pred == gold[i])}
        json.dump({"raw": {"|".join(k2): v2 for k2, v2 in res.items()}},
                  open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
        print(f"[{len(res)}/{len(jobs)}] q{i:<4d} {v:8s}#{k:<2d} -> {pred} "
              f"(gold {gold[i]}) {time.time()-t0:5.1f}s", flush=True)

    # summary
    agg = {}
    for (i, v, k), r in res.items():
        agg.setdefault((i, v), [0, 0])
        agg[(i, v)][0] += r["acc"]; agg[(i, v)][1] += 1
    tot = {"clean": [0, 0], "corrupt": [0, 0]}
    print(f"\n{'idx':>4s} {'clean':>8s} {'corrupt':>8s}")
    for i in AFF:
        row = []
        for v in ("clean", "corrupt"):
            a = agg.get((str(i), v), [0, 0])
            tot[v][0] += a[0]; tot[v][1] += a[1]
            row.append(f"{a[0]}/{a[1]}")
        print(f"{i:4d} {row[0]:>8s} {row[1]:>8s}")
    for v in ("clean", "corrupt"):
        a = tot[v]
        print(f"TOTAL {v}: {a[0]}/{a[1]} = {a[0]/max(1,a[1]):.3f}")


if __name__ == "__main__":
    main()
