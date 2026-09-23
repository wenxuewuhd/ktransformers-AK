"""KT_FULLSET_LOAD: resident experts get CPU weights too (so they can be demoted).

Origin: dsv41 stream-prefill F2-3 (full-set load for runtime demotion).

What is judged, and against what:

* ``TestSwitchLogic`` -- ``NativeMoEWrapper._cpu_expert_subset_enabled`` on a
  bare instance. Expected values are literal truth tables written from the old
  rule (``SUPPORTS_EXPERT_SUBSET and num_gpu_experts > 0``), not computed by
  the function under test.

* ``TestRealBackend`` -- the compiled ``kt_kernel_ext`` loading a tiny
  synthetic MXFP4 checkpoint, one subprocess per arm (CPUInfer is a
  process-wide singleton, and reading a BufferB that was never allocated may
  take the process down rather than raise). Each child reports:
    - the ``expert_ids`` the loader was actually called with,
    - ``skip_gpu_expert_weights`` read back from the very ``MOEConfig`` object
      the backend was constructed from (``amx.MOEConfig`` is wrapped, not
      re-derived),
    - for each expert, the bytes the C++ backend hands back from its BufferB
      via ``write_weight_scale_to_buffer``, compared to the checkpoint bytes
      the test itself generated (weights) or to 2**(e-127) computed here
      (scales). A resident expert whose bytes round-trip proves its BufferB
      was allocated *and* loaded -- stronger than any flag.
  CPU experts are the positive control: they must round-trip in every arm,
  otherwise the comparison itself is wrong and the test says so.

Which kt is tested: whatever ``kt_kernel`` the interpreter imports
(``PYTHONPATH``); every child prints the ``amx.py`` it imported. Mutation arm:
run the same file against the pre-change site (``kt_site_v2``) or a copy with
the switch reverted -- ``test_on_*`` must fail there.

Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<kt site> python -m pytest -q test/test_fullset_load.py
     (or: python test/test_fullset_load.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HIDDEN, INTER, N_EXP, TOPK, LAYER = 1024, 512, 8, 2, 0
GROUP = 32
# Resident (accelerator) experts: a non-prefix, non-contiguous mask on purpose.
RESIDENT = (1, 4, 6)
CPU = tuple(e for e in range(N_EXP) if e not in RESIDENT)
SEED = 20260923


# ---------------------------------------------------------------------------
# Switch logic (no C++ load)
# ---------------------------------------------------------------------------


class _SubsetLoader:
    SUPPORTS_EXPERT_SUBSET = True


class _PlainLoader:
    pass


def _bare_wrapper(loader, num_gpu_experts):
    from kt_kernel.utils.amx import NativeMoEWrapper

    w = NativeMoEWrapper.__new__(NativeMoEWrapper)
    w.loader = loader
    w.num_gpu_experts = num_gpu_experts
    return w


class _Env:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.pop("KT_FULLSET_LOAD", None)
        if self.value is not None:
            os.environ["KT_FULLSET_LOAD"] = self.value

    def __exit__(self, *exc):
        os.environ.pop("KT_FULLSET_LOAD", None)
        if self.old is not None:
            os.environ["KT_FULLSET_LOAD"] = self.old


# (loader, num_gpu_experts) -> old rule, written out literally.
OLD_TABLE = [
    (_SubsetLoader, 3, True),
    (_SubsetLoader, 0, False),
    (_PlainLoader, 3, False),
    (_PlainLoader, 0, False),
]


class TestSwitchLogic(unittest.TestCase):
    def test_off_matches_old_rule(self):
        for env in (None, "", "0"):
            with _Env(env):
                for loader, ngpu, want in OLD_TABLE:
                    with self.subTest(env=env, loader=loader.__name__, ngpu=ngpu):
                        self.assertIs(_bare_wrapper(loader(), ngpu)._cpu_expert_subset_enabled(), want)

    def test_on_disables_subset(self):
        with _Env("1"):
            for loader, ngpu, _ in OLD_TABLE:
                with self.subTest(loader=loader.__name__, ngpu=ngpu):
                    self.assertIs(_bare_wrapper(loader(), ngpu)._cpu_expert_subset_enabled(), False)

    def test_on_rejects_unparseable_value(self):
        # A typo must not silently mean "subset".
        for bad in ("true", "yes", "2", " 1"):
            with _Env(bad):
                with self.subTest(value=bad), self.assertRaises(ValueError):
                    _bare_wrapper(_SubsetLoader(), 3)._cpu_expert_subset_enabled()


# ---------------------------------------------------------------------------
# Real backend, one subprocess per arm
# ---------------------------------------------------------------------------


def _write_checkpoint(dirpath: Path) -> None:
    import torch
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(SEED)
    t = {}
    base = f"model.layers.{LAYER}.ffn.experts"
    shapes = {"w1": (INTER, HIDDEN), "w3": (INTER, HIDDEN), "w2": (HIDDEN, INTER)}
    for e in range(N_EXP):
        for proj, (n, k) in shapes.items():
            t[f"{base}.{e}.{proj}.weight"] = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
            # ue8m0 exponents kept in bf16's normal range.
            t[f"{base}.{e}.{proj}.scale"] = torch.randint(110, 140, (n, k // GROUP), generator=g, dtype=torch.uint8)
    save_file(t, str(dirpath / "model.safetensors"))


CHILD = textwrap.dedent(
    r"""
    import json, os, sys
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
    os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"
    import torch
    for _n in ("zeros", "empty", "full"):
        _o = getattr(torch, _n)
        def _nopin(*a, __o=_o, **kw):
            kw.pop("pin_memory", None); return __o(*a, **kw)
        setattr(torch, _n, _nopin)
    from safetensors import safe_open
    import kt_kernel.experts_base as _eb
    _eb._ensure_ascend_callback_worker = lambda: None
    import kt_kernel.utils.amx as amx
    from kt_kernel.experts import KTMoEWrapper
    assert "torch_npu" not in sys.modules

    ckpt, H, I, E, K, L = sys.argv[1], *map(int, sys.argv[2:7])
    resident = [int(x) for x in sys.argv[7].split(",")]
    out = {"amx_file": amx.__file__, "env": os.environ.get("KT_FULLSET_LOAD")}

    # Spy 1: the MOEConfig object the backend is built from.
    _configs = []
    _Real = amx.MOEConfig
    def _spy_cfg(*a, **kw):
        c = _Real(*a, **kw); _configs.append(c); return c
    amx.MOEConfig = _spy_cfg
    # Spy 2: what the loader was asked to read.
    _calls = []
    _Loader = amx.MXFP4SafeTensorLoader
    _orig = _Loader.load_experts
    def _spy_load(self, base_key, *a, **kw):
        r = _orig(self, base_key, *a, **kw)
        ids = kw.get("expert_ids")
        _calls.append(None if ids is None else sorted(int(x) for x in ids))
        return r
    _Loader.load_experts = _spy_load

    mask = torch.zeros(E, dtype=torch.bool); mask[resident] = True
    w = KTMoEWrapper(layer_idx=L, num_experts=E, num_experts_per_tok=K, hidden_size=H,
                     moe_intermediate_size=I, gpu_experts_mask=mask, cpuinfer_threads=4,
                     threadpool_count=2, weight_path=ckpt, chunked_prefill_size=16,
                     method="MXFP4", numa_nodes=[0, 1])
    w.load_weights(torch.arange(E, dtype=torch.int64))
    out["loader_expert_ids"] = _calls
    out["skip_gpu_expert_weights"] = [bool(c.skip_gpu_expert_weights) for c in _configs]
    out["mask_after_load"] = [int(e) for e in torch.nonzero(w.gpu_experts_mask).flatten().tolist()]
    out["backend"] = type(w.moe).__name__
    if hasattr(w, "lacks_cpu_weights"):
        out["lacks_cpu_weights"] = [bool(w.lacks_cpu_weights(e)) for e in range(E)]
        out["cpu_weight_expert_ids"] = list(w.cpu_weight_expert_ids())
        # The answer is about what was allocated at load, not the live mask: a
        # later mask rewrite (a swap) must not change it.
        saved = w.gpu_experts_mask.clone()
        w.gpu_experts_mask.copy_(~saved)
        out["lacks_after_mask_flip"] = [bool(w.lacks_cpu_weights(e)) for e in range(E)]
        w.gpu_experts_mask.copy_(saved)

    # Expected bytes, from the checkpoint the parent wrote (not from kt).
    f = safe_open(os.path.join(ckpt, "model.safetensors"), "pt")
    base = f"model.layers.{L}.ffn.experts"
    def exp_scale(e_u8):
        return torch.pow(2.0, e_u8.to(torch.float32) - 127.0).to(torch.bfloat16).flatten()
    probe = [int(x) for x in sys.argv[8].split(",")]
    rt = {}
    for e in probe:
        w13 = torch.full((I * H,), 0xAA, dtype=torch.uint8)          # 2 * I*H/2 bytes
        w13s = torch.zeros(2 * I * H // 32, dtype=torch.bfloat16)
        w2 = torch.full((H * I // 2,), 0xAA, dtype=torch.uint8)
        w2s = torch.zeros(H * I // 32, dtype=torch.bfloat16)
        w.submit_write_weight_scale_to_buffer(1, e, [w13.data_ptr()], [w13s.data_ptr()],
                                              [w2.data_ptr()], [w2s.data_ptr()])
        w.sync_write_weight_scale_to_buffer()
        g = lambda p, s: f.get_tensor(f"{base}.{e}.{p}.{s}")
        want_w13 = torch.cat([g("w1", "weight").view(torch.uint8).flatten(), g("w3", "weight").view(torch.uint8).flatten()])
        want_w13s = torch.cat([exp_scale(g("w1", "scale").view(torch.uint8)), exp_scale(g("w3", "scale").view(torch.uint8))])
        want_w2 = g("w2", "weight").view(torch.uint8).flatten()
        want_w2s = exp_scale(g("w2", "scale").view(torch.uint8))
        rt[str(e)] = {
            "w13": bool(torch.equal(w13, want_w13)),
            "w13_scale": bool(torch.equal(w13s.view(torch.int16), want_w13s.view(torch.int16))),
            "w2": bool(torch.equal(w2, want_w2)),
            "w2_scale": bool(torch.equal(w2s.view(torch.int16), want_w2s.view(torch.int16))),
            "bytes_checked": int(w13.numel() + 2 * w13s.numel() + w2.numel() + 2 * w2s.numel()),
            # Distinguishing power: the same bytes must NOT match a neighbour's
            # checkpoint (else "equal" would hold for any loaded expert).
            "w13_matches_other_expert": bool(torch.equal(w13, torch.cat([
                f.get_tensor(f"{base}.{(e + 1) % E}.w1.weight").view(torch.uint8).flatten(),
                f.get_tensor(f"{base}.{(e + 1) % E}.w3.weight").view(torch.uint8).flatten()]))),
        }
    out["roundtrip"] = rt
    print("RESULT " + json.dumps(out), flush=True)
    """
)


def _run_child(ckpt: Path, env_value, probe):
    env = dict(os.environ)
    env.pop("KT_FULLSET_LOAD", None)
    if env_value is not None:
        env["KT_FULLSET_LOAD"] = env_value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    args = [str(ckpt), str(HIDDEN), str(INTER), str(N_EXP), str(TOPK), str(LAYER),
            ",".join(map(str, RESIDENT)), ",".join(map(str, probe))]
    p = subprocess.run([sys.executable, "-c", CHILD, *args], env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if p.returncode != 0 or line is None:
        raise AssertionError(
            f"child (KT_FULLSET_LOAD={env_value!r}, probe={probe}) rc={p.returncode}, no result.\n"
            f"--- stdout tail ---\n{p.stdout[-2000:]}\n--- stderr tail ---\n{p.stderr[-2000:]}"
        )
    return json.loads(line[len("RESULT "):])


def _all_ok(rt_entry):
    return all(rt_entry[k] for k in ("w13", "w13_scale", "w2", "w2_scale"))


class TestRealBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import kt_kernel.utils.amx as amx  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest(f"kt_kernel_ext not importable: {exc}")
        if amx.AMXFP4_KGroup_MOE is None:  # pragma: no cover
            raise unittest.SkipTest("AMXFP4_KGroup_MOE not compiled in")
        cls._tmp = tempfile.TemporaryDirectory(prefix="kt_fullset_")
        cls.ckpt = Path(cls._tmp.name)
        _write_checkpoint(cls.ckpt)
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _get(self, env_value, probe):
        key = (env_value, tuple(probe))
        if key not in self.results:
            self.results[key] = _run_child(self.ckpt, env_value, probe)
            r = self.results[key]
            print(f"\n[child KT_FULLSET_LOAD={env_value!r}] amx={r['amx_file']} backend={r['backend']} "
                  f"loader_ids={r['loader_expert_ids']} skip={r['skip_gpu_expert_weights']} "
                  f"rt={ {e: _all_ok(v) for e, v in r['roundtrip'].items()} }", file=sys.stderr)
        return self.results[key]

    # -- switch off: the old subset path ------------------------------------
    def test_off_subset_path_unchanged(self):
        r = self._get(None, CPU)
        self.assertEqual(r["loader_expert_ids"], [list(CPU)])
        self.assertEqual(r["skip_gpu_expert_weights"], [True])
        self.assertEqual(r["mask_after_load"], list(RESIDENT))
        # Positive control: offloaded experts round-trip byte for byte.
        self.assertEqual(sorted(int(e) for e in r["roundtrip"]), list(CPU))
        for e, v in r["roundtrip"].items():
            self.assertTrue(_all_ok(v), f"CPU expert {e} did not round-trip: {v}")
            self.assertGreater(v["bytes_checked"], 0)
            self.assertFalse(v["w13_matches_other_expert"], "comparison has no distinguishing power")
        if "lacks_cpu_weights" in r:
            self.assertEqual([e for e in range(N_EXP) if r["lacks_cpu_weights"][e]], list(RESIDENT))
            self.assertEqual(r["cpu_weight_expert_ids"], list(CPU))
            self.assertEqual(r["lacks_after_mask_flip"], r["lacks_cpu_weights"])

    def test_off_zero_is_off(self):
        r = self._get("0", CPU)
        self.assertEqual(r["loader_expert_ids"], [list(CPU)])
        self.assertEqual(r["skip_gpu_expert_weights"], [True])

    # -- switch on: full set ------------------------------------------------
    def test_on_loads_full_set(self):
        r = self._get("1", tuple(range(N_EXP)))
        self.assertEqual(r["loader_expert_ids"], [None], "loader must be asked for every expert (no subset)")
        self.assertEqual(r["skip_gpu_expert_weights"], [False])
        self.assertEqual(r["mask_after_load"], list(RESIDENT), "routing mask must still mark the residents")
        self.assertEqual(r["backend"], "AMXFP4_KGroup_MOE")
        self.assertIn("lacks_cpu_weights", r)
        self.assertEqual(r["lacks_cpu_weights"], [False] * N_EXP)
        self.assertEqual(r["cpu_weight_expert_ids"], list(range(N_EXP)))
        self.assertEqual(r["lacks_after_mask_flip"], [False] * N_EXP)

    def test_on_every_expert_has_cpu_weights(self):
        r = self._get("1", tuple(range(N_EXP)))
        # Coverage: all N_EXP experts probed, residents included.
        self.assertEqual(sorted(int(e) for e in r["roundtrip"]), list(range(N_EXP)))
        for e in CPU:  # positive control
            self.assertTrue(_all_ok(r["roundtrip"][str(e)]), f"CPU expert {e}: {r['roundtrip'][str(e)]}")
        for e in RESIDENT:  # the point of the switch
            self.assertTrue(_all_ok(r["roundtrip"][str(e)]), f"resident expert {e}: {r['roundtrip'][str(e)]}")
        for e, v in r["roundtrip"].items():
            self.assertFalse(v["w13_matches_other_expert"], f"expert {e}: comparison has no distinguishing power")


if __name__ == "__main__":
    unittest.main(verbosity=2)
