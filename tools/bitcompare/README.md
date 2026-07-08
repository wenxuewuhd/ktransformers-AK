# Bit-level regression harness (DSV4-Flash single-card 910C)

Proves the clean-code pass does **not** change inference behaviour: same fixed prompt at
`temperature=0` must produce the same output token-ids (and, per the tier below, the same
logprobs) before and after the cleanup.

## Pieces
- `make_prompt.py` → writes `prompt.txt`, a fixed deterministic ~1.8K-token prompt. Long
  enough to trigger streaming prefill (threshold 512) + inline-resident depool + tail-layer
  dynamic residency (the code paths the cleanup touches most), short enough to stay in one
  chunk (< `chunked_prefill_size` 8192, avoiding the 坑⑯ NSA cross-chunk crash).
- `run_server.sh` → boots the production launcher (`tools/p27_launch_ds4flash_npu.sh`) on a
  spare port/device with this box's model + mxfp4 GGUF paths, so the capture goes through the
  exact production code path. Box-specific env it sets:
  - `KT_NSA_COMPRESSOR_MODE=single` — this box has the CANN 9.0.0 public 18-arg compressor op;
    the code defaults to the private 19-arg split ABI, which fails graph capture here.
  - `KT_THREADPOOL_COUNT=1`, `KT_CPUINFER=32` — single-NUMA, 40-core host (launcher's 8-NUMA
    `8/128` defaults bind to non-existent NUMA nodes 1..7 → `alloc N from other numa` fallbacks).
  - `PORT=8021`, `NPU_DEVICE_ID=0` — avoid the main service.
- `capture_client.py` → POSTs the fixed prompt to `/generate` at `temperature=0`,
  `max_new_tokens=64`, `return_logprob`, `top_logprobs_num=20`; saves output_ids +
  chosen/step top-k logprobs to a golden JSON.
- `compare.py` → diffs two goldens. Tier A = output_ids identical AND all logprobs bit-equal;
  Tier B = output_ids identical AND `max|Δ logprob| <= --logprob-tol`.

## Procedure
```bash
# 1. boot baseline (pre-cleanup tree)
bash tools/bitcompare/run_server.sh    # wait for "Uvicorn running on ... :8021"
# 2. capture golden
python3 tools/bitcompare/capture_client.py --port 8021 --out tools/bitcompare/goldens/baseline.json
# 3. (after cleanup) reboot cleaned tree, recapture, compare
python3 tools/bitcompare/capture_client.py --port 8021 --out tools/bitcompare/goldens/cleaned.json
python3 tools/bitcompare/compare.py --ref .../baseline.json --new .../cleaned.json --tier <A|B>
```

## Determinism findings (why the tier is what it is)
- **Same-boot, run-to-run: bit-exact (Tier A).** Two identical requests in one boot →
  output_ids identical, `max|Δ logprob| = 0.0`. So the forward has no run-to-run nondeterminism
  under this config (bs=1, serial, `max-running-requests 1`).
- **Cross-boot: also bit-exact (Tier A).** Rebooting the server and recapturing the same
  prompt → output_ids identical, `max|Δ logprob| = 0.0` vs the first boot. This **contradicts**
  the old doc claim that "this stack isn't reproducible across boots even at greedy": under this
  config (temp=0 greedy → argmax, bs=1, `max-running-requests 1`) it is fully cross-boot
  reproducible. Greedy has no RNG, and the forward has no cross-boot layout nondeterminism here.
- **Acceptance tier for the cleanup: A (strict bit-exact).** After the clean-up, a reboot +
  recapture must be `torch.equal` to `goldens/baseline.json` (output_ids AND all logprobs).
  `goldens/baseline.json` = boot-1 capture; `baseline_boot1_b.json` (same-boot) and
  `baseline_boot2_a.json` (cross-boot) are the determinism witnesses, all bit-identical.
