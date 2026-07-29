#!/usr/bin/env python3
"""NPU sampler kernel audit: does torch.multinomial(+softmax) on THIS box's
torch_npu/CANN produce samples faithful to the given distribution?

Replicates sampler.py _sample_from_logits simple case exactly:
    probs = torch.softmax(logits, dim=-1); torch.multinomial(probs, 1)
on synthetic logits with known ground truth. Any box whose empirical
frequencies deviate from its own softmax (beyond sampling noise) has a broken
sampling kernel -> temp=1 eval accuracy drops with clean teacher-forced logprobs.
"""
import argparse, json, math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--n", type=int, default=2_000_000, help="draws per case")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.manual_seed(1234)
    dev = args.device
    V = args.vocab
    cases = {}

    g = torch.Generator().manual_seed(7)
    # case A: sharp, CoT-like (top1 ~0.6)
    la = torch.full((V,), -14.0)
    la[:2000] = torch.randn(2000, generator=g) * 2.0
    la[0] = 6.0; la[1] = 4.4; la[2] = 3.8
    cases["sharp"] = la
    # case B: flat-ish, decision-point-like (top1 ~0.25)
    lb = torch.full((V,), -12.0)
    lb[:5000] = torch.randn(5000, generator=g) * 1.2
    lb[:8] = torch.tensor([2.2, 1.9, 1.7, 1.5, 0.8, 0.4, 0.1, -0.2])
    cases["flat"] = lb
    # case C: long tail heavy (stress low-prob region)
    lc = torch.randn(V, generator=g) * 0.5
    lc[0] = 5.0
    cases["tail"] = lc

    report = {"device": dev, "n": args.n, "torch": torch.__version__}
    try:
        report["torch_npu"] = torch_npu.__version__
    except Exception:
        pass

    for name, logits_cpu in cases.items():
        logits = logits_cpu.to(dev, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        # softmax fidelity vs fp64 CPU reference
        ref = torch.softmax(logits_cpu.to(torch.float64), dim=-1)
        p_cpu = probs.cpu().to(torch.float64)
        sm_err = (p_cpu - ref).abs().max().item()

        counts = torch.zeros(V, dtype=torch.int64, device=dev)
        pb = probs.unsqueeze(0).expand(args.batch, V)
        done = 0
        while done < args.n:
            b = min(args.batch, args.n - done)
            idx = torch.multinomial(pb[:b], num_samples=1).view(-1)
            counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))
            done += b
        c = counts.cpu().to(torch.float64)
        p = ref  # ground truth = fp64 softmax of same logits
        n = float(args.n)

        # chi-square over top-64 + tail bucket
        order = torch.argsort(p, descending=True)
        top = order[:64]
        exp_top = p[top] * n
        obs_top = c[top]
        exp_tail = n - exp_top.sum()
        obs_tail = n - obs_top.sum()
        chi2 = ((obs_top - exp_top) ** 2 / exp_top).sum().item()
        if exp_tail.item() > 5:
            chi2 += ((obs_tail - exp_tail) ** 2 / exp_tail).item()
        dof = 64  # 65 buckets - 1
        # per-token z for top-8
        z = ((obs_top[:8] - exp_top[:8]) /
             (exp_top[:8] * (1 - p[top[:8]] * 1.0)).sqrt()).tolist()
        report[name] = {
            "softmax_max_abs_err_vs_fp64": sm_err,
            "chi2_top64_plus_tail": chi2,
            "dof": dof,
            "top1_p_true": p[top[0]].item(),
            "top1_p_emp": (c[top[0]] / n).item(),
            "top8_z": [round(v, 2) for v in z],
        }
        print(f"{name:6s} chi2={chi2:9.1f} (dof={dof})  "
              f"top1 true={p[top[0]]:.5f} emp={c[top[0]]/n:.5f}  "
              f"softmax_err={sm_err:.2e}")
        print(f"       top8 z-scores: {[round(v,2) for v in z]}")

    json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
