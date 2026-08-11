# DSv4 Single-NPU Upstreaming Plan (based on sgl-project mainline)

> Updated: 2026-08-11. Route decided: the sglang side builds on **sgl-project mainline**
> (upstream NPU DSv4 verified working by us), NOT on the kvcache-ai/sglang fork.
> This is the single execution reference: full PR list, ordering, and per-phase tasks.
> Chinese version: `UPSTREAM_opensource_plan.md` (kept in sync manually).

---

## 1. Established facts (inventory — do not re-derive)

### 1.1 Where our changes live (three places)

| Location | Size | Content | Upstream destination |
|---|---|---|---|
| Main repo `kt-kernel/` | 48 files / +7018 lines | Ascend vendor backend (callback worker, `vendors/ascend_npu.h`), K920 KML int8/int4 prefill GEMM, llamafile MXFP4 ARM path, Python-side experts_base/loader | kvcache-ai/ktransformers (K1/K2) |
| Main repo `third_party/llamafile/` (in-tree) | 2 files | ARM82 GEMV software-prefetch optimization (2.4× on K920) | Ships with K2 |
| Main repo `script/`, `doc/`, `tools/` | ~350 files | Almost all internal (p27 scripts, patch series, handoffs) — **not upstreamed**; only curated launch scripts + newly written docs go | K3 |
| sglang fork (`wenxuewuhd/sglang-dsv4@dsv4_release`) | 76 files / +14158 lines | = sgl-project snapshot of 2026-04-09 + 3 commits (Yijie Zhu, superseded by mainline) + **46 commits (ours)** | Ported, then submitted to sgl-project (S1–S4) |
| `third_party/llama.cpp` submodule | ~150 lines of **uncommitted** working-tree changes | See 1.3 | No PR anywhere; handled inside K1 |

### 1.2 Upstream status

- **sgl-project mainline NPU DSv4 is ready**: PR #28980 (Jun 30, MTP) + #31931 (Jul 28: PD
  disaggregation, chunked prefill with compression state preserved across chunks, fused
  compressor, dual-stream decode). **Verified working by us (2026-08).**
- Environment: CANN 9.0.0 + torch_npu 2.10.0; official images
  `quay.io/ascend/cann:9.0.0-910b` / `9.0.0-a3`.
- Both quant formats supported: ModelSlim and compressed-tensors are two front-ends over the
  same NPU kernels (`NPUCompressedTensorsW8A8Int8[DynamicMoE]`, etc.). The PRs were only
  validated with ModelSlim W8A8.
- Code layout was restructured: `hardware_backend/npu/` (`dsv4/`,
  `attention/ascend_dsv4_backend.py`, `quantization/`, `graph_runner/`),
  `layers/moe/moe_runner/ascend.py`. The port must map onto this new structure.
- kvcache-ai/ktransformers already ships GPU DSv4-Flash (single RTX 5090 + CPU MoE); their
  kt-kernel MXFP4 exists only on the x86 AMX path — **our llamafile/ARM MXFP4 and the Ascend
  backend are net-new**, no conflicts.
- A standalone `kvcache-ai/kt-kernel` repo does not exist (404); kt-kernel lives only inside
  the ktransformers main repo tree.

### 1.3 llama.cpp uncommitted changes = two independent patches

| Patch | Content | Nature | Disposition |
|---|---|---|---|
| ① C side, ~130 lines (`ggml.h/.c`, `ggml-quants.c/.h`, `ggml-common.h`) | `block_mxfp4` + `dequantize_row_mxfp4` + `ggml_vec_dot_mxfp4_q8_0` (NEON) + type_traits registration id=39; includes K920 software-prefetch tuning | **Required in production** (llamafile CPU MoE dispatches through ggml type_traits); backport from master + our tuning | Commit as patch in Phase 0; ship with K1 as vendor-into-kt-kernel or build patch — maintainers decide at RFC |
| ② gguf-py, 21 lines (2-line MXFP4 enum + 19-line NumPy 2 compat) | Only serves internal `tools/` scripts that sys.path the submodule's gguf-py | **Redundant**: the production loader uses pip gguf ≥ 0.17 (0.18 has MXFP4 + NumPy 2 compat built in) | Point the scripts at pip gguf, then discard |

**No PR to ggerganov/llama.cpp**: b3173 is a historical tag that cannot take PRs, and master
already has MXFP4 (we picked id=39 precisely to align with it).

### 1.4 Upstream CI and accuracy requirements

- The only PR gate: `.github/workflows/kt-kernel-tests.yml`. Triggers when a PR touches
  `kt-kernel/**`, requires a maintainer-applied `run-ci` label, non-draft. Runs on a
  self-hosted **x86 AMX** runner (`kt-cpu`): `install.sh build` →
  `test/run_suite.py --hw cpu --suite default`.
- **Accuracy is operator-level only**: random weights, single MoE layer forward vs a plain
  torch reference, relative-error threshold (AMX int8: `mean(|out−ref|)/mean(|ref|) < 0.05`).
  There is no model-level (GPQA-style) CI anywhere; model-level accuracy is PR evidence
  culture (screenshots in the PR description).
- Test registration: `test/ci/ci_register.py`; `HWBackend` has only CPU/CUDA/AMD (CUDA/AMD
  are placeholders). NPU tests cannot run in their CI — follow the placeholder precedent,
  real runs stay local as PR evidence.
- Commit format is enforced (CONTRIBUTING regex): `[type](scope): msg`, type ∈ feat|fix|docs|…;
  every commit carries `Signed-off-by` (DCO); no Co-Authored-By.

---

## 2. Full PR list and ordering

### 2.1 sgl-project/sglang (1 RFC + 4 PRs, serial stack)

Open the **RFC issue** first: single-card 910B + CPU heterogeneous DSv4-Flash (vs. today's
16-device A3 all-NPU), with measured numbers, the S1–S4 split, and how kt-kernel should be
taken as an optional dependency.

| # | Topic | Content | Size | Depends on |
|---|---|---|---|---|
| S1 | KT-EP wrapper: CPU MoE offload core | NPU-side kt_ep_wrapper (submit/sync/merge overlapping NPU MoE), kt-kernel integration, per-layer GGUF path resolution, basic `gpu_experts_mask` (first N experts on NPU) | M | — |
| S2 | Expert placement & remap | prefix/frequency masks, logical→physical expert remap (grouped matmul + checkpoint loading), mask cloned together with resident weights (NSA-stall fix) | M | S1 |
| S3 | Streaming prefill + depool + dynamic residency | Streaming weight pool, slot time-sharing + reservation, dynamic resident gather, on-the-fly conversion chain (MXFP4→int8 NZ), startup warmup | L | S1, S2 |
| S4 | Single-card deployment docs + wrap-up | Single-card heterogeneous guide under `docs/.../ascend-npus/`, parameter reference, misc | S | S1–S3 |

### 2.2 kvcache-ai/ktransformers (1 issue + 3 PRs)

Open an **issue** first: announce single-card Ascend support, link the sgl-project RFC, and
let maintainers decide the llama.cpp option (vendor into kt-kernel vs build patch).

| # | Topic | Content | Size | Depends on |
|---|---|---|---|---|
| K1 | kt-kernel Ascend NPU backend | `cpu_backend/ascend_callback_worker.{cpp,h}`, `vendors/ascend_npu.h`, cpuinfer/ext_bindings wiring, CMake/install.sh Ascend branch, Python experts_base/loader, **llama.cpp C-side MXFP4 disposition (vendor/patch)** | M-L | llama.cpp decision |
| K2 | ARM/K920 CPU operators | KML int8/int4 prefill GEMM, llamafile MXFP4 path (`operators/llamafile/moe.hpp`), the 2 `third_party/llamafile` files, **new MXFP4 operator accuracy test (per_commit style)** | M | K1 (can stack on same branch) |
| K3 | Docs + launch scripts | `doc/en/DeepSeek-V4-Flash-Ascend.md` (mirroring the GPU doc's format), `script/` launchers, pinned companion sglang mainline version | S | K1, K2 merged; S series landed |

### 2.3 third_party repos: zero PRs

| Repo | Disposition |
|---|---|
| ggerganov/llama.cpp | No PR (see 1.3). Contributing the K920 prefetch tuning to master is an optional nice-to-have, out of scope |
| kvcache-ai/sglang | No PR. NPU users consume sgl-project mainline directly |
| Mozilla llamafile | No PR. We modify ktransformers' in-tree copy; ships with K2 |
| pybind11 / custom_flashinfer | Untouched |

### 2.4 Ordering overview

```
Phase 0 (preservation + CI pre-flight, local)
   │
   ├─► RFC issue (sgl-project) ───► S1 ► S2 ► S3 ► S4      (serial stack)
   └─► issue (kvcache-ai) ────────► K1 ► K2 ──────► K3
                                    (K1/K2 run parallel to the S series; K3 waits for both)
```

- The two tracks are mostly parallel; the only hard serialization point is **K3**
  (its docs must name the companion sglang version).
- The S series is a serial stack; K1/K2 can stack on one branch.
- Total: **7 PRs + 2 issues**.

---

## 3. Phase-by-phase execution plan

### Phase 0 — Preservation + CI pre-flight (local, no new machines, ~2 days)

- [ ] **P0.1 Commit llama.cpp patch ①**: turn the ~130 C-side lines into a patch file
      committed to this repo (the only asset at risk of loss — one `git submodule update`
      wipes it).
- [ ] **P0.2 Dispose of patch ②**: switch the `tools/` scripts that sys.path the submodule's
      gguf-py (`batch_convert_*_mp.py`, `ascendc_mxfp4/test_*`, …) to pip gguf; zero out the
      submodule working tree.
- [ ] **P0.3 Upstream manifest**: per-file disposition — upstream / not-upstream / needs
      cleanup (p27 markers, internal paths, codenames).
- [ ] **P0.4 CI pre-flight (ARM side)**: on the K920 box, pass `install.sh build` +
      `run_suite.py --hw cpu --suite default`.
- [ ] **P0.5 CI pre-flight (x86 compile gate)**: laptop (MateBook KLV-WX9, Whiskey Lake-U)
      via WSL2 + Ubuntu, `install.sh --manual` forcing AVX512=ON and AMX=ON for a full
      compile — proves the ARM/Ascend changes are cleanly arch-gated and the x86 build
      survives. ⚠️ This machine covers the compile gate only: no AVX512/AMX, ≤16 GB RAM, the
      runtime accuracy tests would OOM — left to the first CI run or a later x86 server.
- [ ] **P0.6 New MXFP4 operator accuracy test**: rewrite the existing MXFP4 reference check
      (cosine 0.99994 reconciliation) into upstream's `register_cpu_ci` + pytest pattern
      (random weights → MXFP4 → forward vs torch reference → threshold), skipping cleanly on
      x86; NPU parts follow the placeholder precedent.
- [ ] **P0.7 GPQA evidence pack**: shape the external accuracy numbers into PR-pasteable
      form (see §4).

### Phase B — Port the 46 commits onto sgl-project mainline (the big one, 1.5–2 weeks)

- [ ] **B0** Fork sgl-project, branch from a pinned ref after #31931; **regroup the 46
      commits by feature** (not a mechanical rebase) into the four S1–S4 series. Yijie Zhu's
      3 base commits are superseded by mainline and disappear.
- [ ] **B1** Map onto the new structure: the old code was concentrated in
      `models/deepseek_v4.py`; it now splits across `hardware_backend/npu/dsv4/`,
      `moe_runner/ascend.py`, and quantization schemes. Fuse our MXFP4-CPU + NPU int8
      conversion chain with upstream's NPUW8A8/compressed-tensors schemes. kt-kernel is
      decoupled from sglang — **unchanged**.
- [ ] **B2** Single-card regression (910B/A3, CANN 9.0.0):
      - GPQA holds at the 70.88% baseline; decode holds 19–22.5 tok/s;
      - chunked prefill can finally be enabled (upstream fixed cross-chunk) — verify + long
        prompt regression;
      - **MTP × KT CPU-offload compatibility** (new upside, and the hardest interaction —
        its own line item);
      - re-test long-context mid-position needle retrieval on the new base (the old base's
        NSA block selection dropped mid-context).
- [ ] **B3** Environment unification: rebuild + validate our AscendC custom ops under CANN
      9.0.0 on the 910B box (known core-type pitfall; fix = static + always_inline).

### Phase C — Submission (1 week + review cycles)

- [ ] **C1** Open both issues (sgl-project RFC + kvcache-ai); finish C2 prep while waiting
      for maintainer responses.
- [ ] **C2** Send PRs in the §2 order; run every PR through the §4 checklist; each
      sgl-project PR attaches local verification evidence (GPQA / throughput / msprof).
- [ ] **C3** Review follow-through: first CI run (needs the maintainer's `run-ci` label),
      review feedback, K3 wrap-up.

---

## 4. Per-PR exit checklist

**Commit conventions**: `[type](scope): msg` (regex-enforced); `Signed-off-by` on every
commit (DCO); no Co-Authored-By.

**Code cleanup**: zero p27/phase/task-N/Fx markers (filenames, functions, logs, comments,
docs); internal paths and codenames removed; script hygiene (print→logging etc., per the
established open-source script standard).

**CI (any PR touching `kt-kernel/**`)**: x86 compile gate pre-flighted locally; existing
AMX/AVX2 suites unchanged; new operators ship with per_commit accuracy tests that skip
cleanly on x86.

**Accuracy evidence pack (external numbers — never mix baselines)**:
- GPQA-Diamond thinking-off: **70.88% (evalscope 1.9.1)**; the evalscope 1.8.1-era numbers
  (68.99/67.53/68.13) are void — do not cite.
- 910B, three runs: 69.19 / 72.73 / 73.23 → mean **71.72% / SD 1.80pp**.
- Perf (19–22.5 tok/s, cpu_moe 16 ms) measured on A3; accuracy measured on 910B — label
  accordingly, never blend.
- When comparing against upstream numbers, state the baseline difference: their 86.87% is
  **thinking-on, 16-device A3, all-NPU**.

---

## 5. Machine assignments

| Machine | Role | Status |
|---|---|---|
| K920 + 910B (this box) | ARM-side CI pre-flight, MXFP4 operator tests, B2 single-card regression, GPQA/perf evidence | Available (B3 needs a CANN 9.0.0 container) |
| MateBook laptop (WSL2) | x86 compile-gate pre-flight (compile only, all flags forced) | WSL2 setup pending |
| 8× 910B | Upstream native-path reference (already verified OK; reusable for A/B experiments) | Verified by user |
| x86 AMX server (optional) | Full local pass of runtime AMX/AVX2 accuracy tests | If unavailable, first CI run covers it |

---

## 6. Risk register

1. **B1 port effort dominates**: large structural divergence — this is re-homing, not
   rebasing; budget 1.5–2 weeks.
2. **Mainline moves fast**: pin a ref throughout; catch up to HEAD once before merging.
3. **transformers/hf_hub pins**: we previously hit an upstream `@strict` incompatibility;
   confirm the pin combination when setting up the environment.
4. **MTP × KT CPU-offload interaction unknown**: draft/verify batch shapes vs the CPU MoE
   submission path — dedicated B2 line item.
5. **compressed-tensors on upstream DSv4 is unproven** (PRs only tested ModelSlim): smoke
   both of our checkpoints (A = compressed-tensors, B = ModelSlim) — surface any gap early.
6. **The `run-ci` label is maintainer-controlled**: RFC first to establish contact, so PRs
   don't sit with CI never triggered.
