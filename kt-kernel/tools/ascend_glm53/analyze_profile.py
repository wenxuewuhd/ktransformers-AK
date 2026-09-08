#!/usr/bin/env python3
"""Split a decode step into device time and host time, from a torch_npu profile.

    analyze_profile.py --profile DIR --steps 20

Why this exists: on the pure-NPU path the INT8 line measured device time and wall
clock agreeing point-for-point -- no host bubble at all. Once a CPU MoE is in the
loop that stops being true, and the gap between the two is exactly the cost of the
offload. This reports:

  device us/step   the wall time the device was busy, i.e. the UNION of every kernel's
                   interval across every stream, divided by steps. Not the sum: with a
                   side stream two kernels overlap and summing double-counts them.
  host us/step     wall clock minus that, if --wall-ms is given
  gaps             where the idle time between consecutive kernels actually lands

The gap histogram is the part worth reading. `wall - device` says how much host time
there is; the gaps say *where*, and whether it is one stall per MoE layer (42 of them,
a fixed per-layer cost) or a few large ones (a serialisation point).

⚠ Gaps are the complement of that union, NOT differences between consecutive kernels
sorted by start time. Sorting a two-stream profile by start time and taking consecutive
differences invents gaps that do not exist: with stream A busy [0,100] and [100,200] and
stream B busy [10,20], the device is busy the whole 200 us, but the naive method reports
an 80 us gap between B's end and A's restart. `KT_SIDE_STREAM=1` is the default here, so
this profile is always two-stream and the naive method was always wrong on it.

Deliberately does NOT attribute kernels to layer families. attribute_kernels.py does
that by per-step call count, and a CPU MoE changes the call counts, so its rules no
longer describe this model -- it will assert rather than mislead, which is correct,
but it also means it cannot help here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


def _find_kernel_csv(root: str) -> str:
    hits = glob.glob(os.path.join(root, "**", "kernel_details.csv"), recursive=True)
    if not hits:
        raise SystemExit(
            f"no kernel_details.csv under {root}. Profile with:\n"
            f"  python tools/profile_server_decode.py --port PORT --out {root} "
            f"--steps 20 --concurrency 1"
        )
    return max(hits, key=os.path.getsize)


def _col(fieldnames, *candidates):
    """Column names drift between CANN releases; match case/space-insensitively."""
    norm = {f.lower().replace(" ", "").replace("_", ""): f for f in fieldnames}
    for c in candidates:
        k = c.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="directory written by profile_server_decode.py")
    ap.add_argument("--steps", type=int, required=True, help="decode steps captured")
    ap.add_argument("--wall-ms", type=float, default=None,
                    help="measured wall-clock ms/step, to derive host time")
    ap.add_argument("--gap-us", type=float, default=50.0,
                    help="report gaps at least this large (default 50us)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    path = _find_kernel_csv(args.profile)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")

    fn = rows[0].keys()
    c_dur = _col(fn, "Duration(us)", "Duration us", "duration")
    c_start = _col(fn, "Start Time(us)", "Start Time", "start")
    c_name = _col(fn, "Name", "Op Name", "kernel name")
    c_stream = _col(fn, "Stream ID", "StreamId", "stream")
    if c_dur is None:
        raise SystemExit(f"no duration column in {path}; saw {list(fn)}")

    def fnum(s):
        try:
            return float(str(s).strip().replace("\t", "").replace(",", ""))
        except ValueError:
            return None

    durs = [(fnum(r[c_dur]), r.get(c_name, "?")) for r in rows]
    durs = [(d, n) for d, n in durs if d is not None]
    total_us = sum(d for d, _ in durs)
    print(f"profile   {path}")
    print(f"kernels   {len(durs)}  ({len(durs)/args.steps:.0f} per step)")
    print(f"kernel-time sum {total_us/args.steps:9.1f} us/step  "
          f"({total_us/args.steps/1000:.2f} ms)  ⚠ double-counts overlap; see 'device busy' below")
    if args.wall_ms is not None:
        print(f"wall            {args.wall_ms*1000:9.1f} us/step  ({args.wall_ms:.2f} ms)")

    by_name: dict[str, list[float]] = {}
    for d, n in durs:
        by_name.setdefault(str(n).strip(), []).append(d)
    print(f"\ntop {args.top} kernels by total device time per step:")
    print(f"  {'kernel':<44s} {'calls/step':>10s} {'us/call':>9s} {'us/step':>9s}")
    for name, ds in sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[: args.top]:
        ds_sorted = sorted(ds)
        print(f"  {name[:44]:<44s} {len(ds)/args.steps:10.1f} "
              f"{ds_sorted[len(ds)//2]:9.1f} {sum(ds)/args.steps:9.1f}")

    if c_start is None:
        print("\nno start-time column, cannot compute gaps")
        return 0
    if c_stream is None:
        print("\n⚠ no Stream ID column: gaps below assume a single stream and will be "
              "wrong if this profile has more than one.")

    ev = []
    for r in rows:
        st, d = fnum(r[c_start]), fnum(r[c_dur])
        if st is not None and d is not None:
            ev.append((st, st + d, str(r.get(c_name, "?")).strip(),
                       str(r.get(c_stream, "?")).strip()))
    ev.sort()

    # Merge the busy intervals across ALL streams, then read the holes. The kernel named
    # for a gap is the last one to *end* before it, which is the one the device was
    # waiting on -- not simply the previous row in start order.
    merged = []          # (start, end, name_of_last_kernel_to_end)
    for st, en, nm, _sid in ev:
        if merged and st <= merged[-1][1]:
            if en > merged[-1][1]:
                merged[-1] = (merged[-1][0], en, nm)
        else:
            merged.append((st, en, nm))
    busy_us = sum(e - b for b, e, _ in merged)
    print(f"\ndevice busy (union of {len({e[3] for e in ev})} streams) "
          f"{busy_us/args.steps:9.1f} us/step  ({busy_us/args.steps/1000:.2f} ms)")
    if args.wall_ms is not None:
        idle = args.wall_ms * 1000 - busy_us / args.steps
        print(f"device idle                             {idle:9.1f} us/step  "
              f"({idle/1000:.2f} ms)   {100*idle/(args.wall_ms*1000):.1f}% of the step")

    gaps = []
    for i in range(len(merged) - 1):
        g = merged[i + 1][0] - merged[i][1]
        if g >= args.gap_us:
            # name the kernel that starts the next busy run, for the "before" column
            after = next((nm for st, _en, nm, _s in ev if st == merged[i + 1][0]), "?")
            gaps.append((g, merged[i][2], after))
    tot_gap = sum(g for g, _, _ in gaps)
    print(f"\ngaps >= {args.gap_us:.0f}us between consecutive kernels:")
    print(f"  count {len(gaps)}  ({len(gaps)/args.steps:.1f} per step)  "
          f"total {tot_gap/args.steps:.1f} us/step")
    if gaps:
        agg: dict[tuple[str, str], list[float]] = {}
        for g, a, b in gaps:
            agg.setdefault((a[:26], b[:26]), []).append(g)
        print(f"  {'after':<28s} {'before':<28s} {'n/step':>7s} {'us/step':>9s}")
        for (a, b), gs in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[: args.top]:
            print(f"  {a:<28s} {b:<28s} {len(gs)/args.steps:7.1f} {sum(gs)/args.steps:9.1f}")
        print("\n  A gap that repeats ~42 times per step is one per MoE layer: a fixed "
              "per-layer\n  host cost. A few large gaps instead means one serialisation point.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
