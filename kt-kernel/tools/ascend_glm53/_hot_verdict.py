#!/usr/bin/env python3
"""Compare the per-layer CPU stall between the dynamic-residency off and on runs.

The gap after MoeFinalizeRoutingV2 is the sync() that waits for the CPU experts, once
per MoE layer, so it is the CPU read time directly -- a better witness than throughput
that the resident set now holds hotter experts.

Two readings, and which one is valid depends on whether the box was quiet:

  absolute  at 150 GB/s node-local, the CPU reads 633.3*(1-h)/(1-0.1111) us per layer
            and ~143.6 us of that hides under the resident GEMM. Inverting gives h.
            ⚠ Only meaningful when the stall is at or below the 633.3 us roofline. Under
            contention the effective bandwidth is lower, the stall exceeds it, and the
            inversion returns a negative "hit rate" -- which is the arithmetic telling
            you the model does not apply, not a measurement.

  ratio     stall + hidden is proportional to bytes, which is proportional to (1-h).
            The unknown bandwidth cancels between two arms measured under the same
            contention, so h_on follows from h_off and the ratio alone. This survives a
            busy box, which is the only kind we get.

Pass --h-off to name the off-arm hit rate (default 0.109, the measured prefix value).
"""
import argparse, os, sys

# Per-layer CPU bytes depend on how many experts are resident, so the roofline and the
# uniform-hit baseline are functions of K -- they are not constants. The old fixed pair
# was the K=32 case, and it was being applied to arms that ran at K=28.
# Model constants: true for GLM-5.3-Flash wherever it runs.
N_EXPERTS = 288
TOP_K = 8
MIB_PER_EXPERT = 12.75          # MXFP4: 3 * 2048 * 4096 * 0.53125 bytes
N_MOE_LAYERS = 42

# ⚠ MEASUREMENTS FROM ONE MACHINE, not properties of the model. The projection below is
# only as portable as these three are, and on different hardware it will be confidently
# wrong rather than obviously wrong. Override from the environment when you move.
#   BW_GB_S    node-local host memory bandwidth at 32 threads
#   HIDDEN_US  per-layer CPU time hidden under the resident GEMM and its neighbours
#   DEVICE_MS  NPU time per decode step for the deployed configuration -- measured with
#              --disable-shared-experts-fusion, which is what we deploy; the 31.10 figure
#              taken with fusion ON is 2.2 ms optimistic for this deployment
BW_GB_S = float(os.environ.get("HOT_BW_GB_S", "150.0"))
HIDDEN_US = float(os.environ.get("HOT_HIDDEN_US", "143.6"))
DEVICE_MS = float(os.environ.get("HOT_DEVICE_MS", "33.348"))


def roofline_us(k):
    """us the CPU needs per layer at K resident experts and a uniform-routing hit rate."""
    mb = TOP_K * (N_EXPERTS - k) / N_EXPERTS * MIB_PER_EXPERT * 1.048576
    return mb / BW_GB_S * 1000.0


def uniform_hit(k):
    return k / N_EXPERTS


def stall_us_per_layer(path):
    try:
        text = open(path).read()
    except OSError:
        return None
    for line in text.splitlines():
        if "MoeFinalizeRouting" in line and "Add" in line:
            f = line.split()
            try:
                per_step, n = float(f[-1]), float(f[-2])
                return per_step / n if n else None
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--h-off", type=float, default=0.109,
                    help="off-arm resident hit rate (0.109 = measured prefix value at K=32)")
    ap.add_argument("--k", type=int, default=32,
                    help="resident experts in the arms being compared; the roofline and "
                          "the uniform-hit baseline both depend on it")
    a = ap.parse_args()

    off = stall_us_per_layer(os.path.join(a.results, "gaps_off.txt"))
    on = stall_us_per_layer(os.path.join(a.results, "gaps_on.txt"))
    if off is None or on is None:
        missing = " ".join(m for m, v in (("off", off), ("on", on)) if v is None)
        print(f"  could not read the gap report for: {missing}")
        return 1

    print(f"  per-layer CPU stall   off {off:7.1f} us   on {on:7.1f} us   {100*(on-off)/off:+.1f}%")

    ROOFLINE_US = roofline_us(a.k)
    UNIFORM_HIT = uniform_hit(a.k)
    print(f"  model at K={a.k}: roofline {ROOFLINE_US:.1f} us/layer, "
          f"uniform hit {UNIFORM_HIT:.3f}, device {DEVICE_MS:.2f} ms")
    quiet = max(off, on) <= ROOFLINE_US
    if quiet:
        inv = lambda us: 1 - (us + HIDDEN_US) / ROOFLINE_US * (1 - UNIFORM_HIT)
        print(f"  implied resident hit  off {inv(off):7.3f}      on {inv(on):7.3f}   (absolute, box was quiet)")
    else:
        print(f"  ⚠ stall exceeds the {ROOFLINE_US:.0f} us roofline, so effective bandwidth was")
        print(f"    below 150 GB/s -- the box was contended and the absolute inversion")
        print(f"    does not apply. Using the ratio, which cancels the bandwidth:")
        ratio = (on + HIDDEN_US) / (off + HIDDEN_US)
        h_on = 1 - ratio * (1 - a.h_off)
        print(f"    bytes ratio on/off    {ratio:.4f}")
        print(f"    implied hit  off {a.h_off:.3f} (given)   on {h_on:.3f} (inferred)")
        cpu = ROOFLINE_US * (1 - h_on) / (1 - UNIFORM_HIT)
        dec = DEVICE_MS + max(0.0, cpu - HIDDEN_US) * N_MOE_LAYERS / 1000
        print(f"    projected on a quiet box: cpu {cpu:.1f} us/layer, decode {dec:.1f} ms, {1000/dec:.1f} tok/s")
        print(f"    ⚠ assumes the hidden {HIDDEN_US:.0f} us is unchanged. With fewer resident")
        print(f"      experts there is less resident GEMM to hide behind, so the true h_on")
        print(f"      is if anything higher than this -- the estimate is conservative.")
    print(f"  held-out reference (K=32): prefix 0.109 -> static profile 0.230")
    return 0


if __name__ == "__main__":
    sys.exit(main())
