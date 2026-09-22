"""run_pinned_forward_sync's inline path (DSV4.1 offload §4.22 P2).

The NPU graph path runs one MoE layer per host-func callback: submit + sync.
The submit is a thread hand-off to the TaskQueue worker, measured at ~10.4 us
median round trip (tq_bench), which is ~0.42 ms per token over 40 layers.
``CPUInfer::run_inline`` runs the task on the calling thread instead.

Inlining is only sound while this layer is the entire content of the queue, so
what is tested here is the *decision*, from real state rather than from the
assumption that the graph path never defers:

  * the switch off (or a build without the feature): the call sequence is
    submit + sync, exactly as before -- the point of a default-off switch;
  * switch on, nothing pending anywhere: run_inline + sync(0);
  * switch on but the PREVIOUS layer left a deferred task (``incremental``):
    back to submit, because an inline run would jump that queued task;
  * switch on but THIS layer will leave one (a deferred expert id >= 0 in the
    real buffer, not merely ``max_deferred_experts_per_token > 0``): back to
    submit, and the sync still allows the 1 pending task.

Pure CPU, no weights, no device: the wrapper is built with ``__new__`` and
handed recording stand-ins, the same shape as test_sync_cpu_only.py. The module
is loaded from the source tree so that this runs against the checkout even
where the built wheel is not installed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "python" / "experts_base.py"


def _load_experts_base(inline_enabled):
    """experts_base from the tree, with kt_kernel_ext stubbed.

    ``inline_enabled`` is None for a build that predates the probe (the
    attribute is simply absent), otherwise the bool it returns.
    """
    ext = types.SimpleNamespace()
    if inline_enabled is not None:
        ext.cpuinfer_inline_enabled = lambda: inline_enabled
    pkg = types.ModuleType("kt_kernel")
    pkg.kt_kernel_ext = ext
    pkg.__path__ = []  # a package, so "from kt_kernel import kt_kernel_ext" works
    saved = sys.modules.get("kt_kernel")
    sys.modules["kt_kernel"] = pkg
    sys.modules["kt_kernel.kt_kernel_ext"] = ext
    try:
        spec = importlib.util.spec_from_file_location("kt_experts_base_ut", SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("kt_kernel", None)
        else:
            sys.modules["kt_kernel"] = saved
        sys.modules.pop("kt_kernel.kt_kernel_ext", None)
    return mod


class _RecordingCpuInfer:
    """submit / run_inline / sync, in the order they were called."""

    def __init__(self):
        self.calls = []

    def submit(self, task):
        self.calls.append(("submit", task))

    def run_inline(self, task):
        self.calls.append(("run_inline", task))
        return True

    def sync(self, allow_n_pending=0):
        self.calls.append(("sync", allow_n_pending))


class _NoInlineCpuInfer:
    """A CPUInfer from a build without run_inline (hasattr must decide)."""

    def __init__(self):
        self.calls = []

    def submit(self, task):
        self.calls.append(("submit", task))

    def sync(self, allow_n_pending=0):
        self.calls.append(("sync", allow_n_pending))


class _FakeMoE:
    def __init__(self):
        self.tasks = []

    def forward_task(self, *args):
        task = ("task", len(self.tasks), args[-1])  # args[-1] = incremental
        self.tasks.append(task)
        return task


def _buffers(bs=1, hidden=16, k=4, depth=2, deferred_at=None):
    """What KExpertsCPUBuffer.get_buffer returns, in plain CPU tensors.

    ``deferred_at`` puts a real deferred expert id into this layer's slot --
    the state bit the decision has to read.
    """
    zeros = lambda *shape, dtype=torch.float32: [torch.zeros(*shape, dtype=dtype) for _ in range(depth)]
    deferred = [torch.full((bs, k), -1, dtype=torch.long) for _ in range(depth)]
    if deferred_at is not None:
        deferred[deferred_at][0, 0] = 3
    return (
        zeros(bs, hidden, dtype=torch.bfloat16),
        [torch.zeros((bs, k), dtype=torch.long) for _ in range(depth)],
        deferred,
        zeros(bs, k),
        zeros(bs, hidden, dtype=torch.bfloat16),
        [torch.full((1,), bs, dtype=torch.int32) for _ in range(depth)],
        zeros(bs, hidden, dtype=torch.bfloat16),
    )


def _wrapper(mod, cpu_infer, *, layer_idx=4, max_deferred=0, buffers=None):
    base = mod.BaseMoEWrapper

    class _W(base):
        def load_weights(self, physical_to_logical_map_cpu):  # pragma: no cover - unused
            raise NotImplementedError

        def load_weights_from_tensors(self, *a):  # pragma: no cover - unused
            raise NotImplementedError

    w = _W.__new__(_W)
    w.layer_idx = layer_idx
    w.num_experts_per_tok = 4
    w.max_deferred_experts_per_token = max_deferred
    w.cpu_infer = cpu_infer
    w.moe = _FakeMoE()
    w._check_qlen_fits_cpp_buffers = lambda hidden_states: None
    base._layer_has_pending_deferred.clear()
    mod.KExpertsCPUBuffer.capture_buffers = {1: buffers if buffers is not None else _buffers()}
    return w


def _run(w):
    w.run_pinned_forward_sync(torch.zeros((1, 16), dtype=torch.bfloat16), 0)


def test_the_switch_off_keeps_the_old_submit_then_sync():
    mod = _load_experts_base(False)
    infer = _RecordingCpuInfer()
    w = _wrapper(mod, infer)
    _run(w)
    assert [c[0] for c in infer.calls] == ["submit", "sync"]
    assert infer.calls[-1][1] == 0


def test_a_build_without_run_inline_is_not_asked_for_it():
    mod = _load_experts_base(True)  # switch on, but the CPUInfer has no such method
    infer = _NoInlineCpuInfer()
    w = _wrapper(mod, infer)
    _run(w)
    assert [c[0] for c in infer.calls] == ["submit", "sync"]


def test_a_build_without_the_probe_answers_false():
    mod = _load_experts_base(None)
    infer = _RecordingCpuInfer()
    w = _wrapper(mod, infer)
    _run(w)
    assert [c[0] for c in infer.calls] == ["submit", "sync"]


def test_the_switch_on_and_nothing_pending_runs_the_task_inline():
    mod = _load_experts_base(True)
    infer = _RecordingCpuInfer()
    w = _wrapper(mod, infer)
    _run(w)
    assert [c[0] for c in infer.calls] == ["run_inline", "sync"]
    # the very task kt built, unchanged, and the sync still drains to 0
    assert infer.calls[0][1] == w.moe.tasks[0]
    assert infer.calls[1][1] == 0
    assert mod.BaseMoEWrapper._layer_has_pending_deferred[4] is False


def test_a_task_left_by_the_previous_layer_refuses_the_inline_run():
    """``incremental``: an inline run would execute before that queued task."""
    mod = _load_experts_base(True)
    infer = _RecordingCpuInfer()
    w = _wrapper(mod, infer)
    mod.BaseMoEWrapper._layer_has_pending_deferred[3] = True  # the previous layer
    _run(w)
    assert [c[0] for c in infer.calls] == ["submit", "sync"]
    assert w.moe.tasks[0][2] is True, "and the task is still built as incremental"


def test_a_deferred_expert_in_this_slot_refuses_the_inline_run():
    """The state bit is read from the buffer, not inferred from the config:
    max_deferred > 0 alone is not a deferral, a real id >= 0 is."""
    mod = _load_experts_base(True)
    infer = _RecordingCpuInfer()
    # layer 4 -> slot 0 (layer_idx % buffer_depth)
    w = _wrapper(mod, infer, max_deferred=2, buffers=_buffers(deferred_at=0))
    _run(w)
    assert [c[0] for c in infer.calls] == ["submit", "submit", "sync"]
    assert infer.calls[-1][1] == 1, "the deferred task is left pending on purpose"
    assert mod.BaseMoEWrapper._layer_has_pending_deferred[4] is True


def test_the_same_config_without_a_real_deferred_id_does_inline():
    mod = _load_experts_base(True)
    infer = _RecordingCpuInfer()
    w = _wrapper(mod, infer, max_deferred=2, buffers=_buffers(deferred_at=None))
    _run(w)
    assert [c[0] for c in infer.calls] == ["run_inline", "sync"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
