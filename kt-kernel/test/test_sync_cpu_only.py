"""BaseMoEWrapper.sync_cpu_only / has_pending_deferred (M3 side-stream overlap).

Pure CPU, no model weights, no device: the wrapper is built with __new__ and
handed a recording stand-in for the CPUInfer singleton.
"""

import pytest
import torch

import kt_kernel.experts_base as eb
from kt_kernel.experts_base import BaseMoEWrapper


class _RecordingCpuInfer:
    def __init__(self):
        self.sync_calls = []
        self.other_calls = []

    def sync(self, allow_n_pending=0):
        self.sync_calls.append(allow_n_pending)

    def __getattr__(self, name):  # any other CPUInfer entry point is a failure here
        def _rec(*a, **kw):
            self.other_calls.append(name)

        return _rec


class _Wrapper(BaseMoEWrapper):
    """Concrete enough to instantiate; __init__ is deliberately not run."""

    def load_weights(self, physical_to_logical_map_cpu):  # pragma: no cover - unused
        raise NotImplementedError

    def load_weights_from_tensors(self, gate_proj, up_proj, down_proj, physical_to_logical_map_cpu):
        raise NotImplementedError  # pragma: no cover - unused


@pytest.fixture
def wrapper():
    w = _Wrapper.__new__(_Wrapper)
    w.layer_idx = 7
    w.num_experts_per_tok = 6
    w.cpu_infer = _RecordingCpuInfer()
    BaseMoEWrapper._layer_has_pending_deferred.clear()
    yield w
    BaseMoEWrapper._layer_has_pending_deferred.clear()


def test_drain_allowance_follows_the_deferred_flag(wrapper):
    wrapper.sync_cpu_only()
    assert wrapper.cpu_infer.sync_calls == [0]

    BaseMoEWrapper._layer_has_pending_deferred[7] = True
    wrapper.sync_cpu_only()
    assert wrapper.cpu_infer.sync_calls == [0, 1]

    # ... and it is *this* layer's flag, not a neighbour's
    BaseMoEWrapper._layer_has_pending_deferred[7] = False
    BaseMoEWrapper._layer_has_pending_deferred[6] = True
    wrapper.sync_cpu_only()
    assert wrapper.cpu_infer.sync_calls == [0, 1, 0]


def test_no_device_wait_and_no_other_cpuinfer_entry_point(wrapper, monkeypatch):
    """The whole point of the method: it must not synchronize the accelerator."""

    def _boom(device):  # pragma: no cover - only runs on regression
        raise AssertionError("sync_cpu_only must not call _wait_device")

    monkeypatch.setattr(eb, "_wait_device", _boom)
    monkeypatch.setattr(eb, "_is_capture_mode", lambda: pytest.fail("capture probe must not be consulted"))
    wrapper.sync_cpu_only()
    assert wrapper.cpu_infer.sync_calls == [0]
    assert wrapper.cpu_infer.other_calls == []
    # negative control: the guard above really does fire when something waits
    with pytest.raises(AssertionError):
        eb._wait_device(torch.device("cpu"))


def test_has_pending_deferred_reads_and_targets_a_layer(wrapper):
    assert wrapper.has_pending_deferred() is False
    BaseMoEWrapper._layer_has_pending_deferred[7] = True
    BaseMoEWrapper._layer_has_pending_deferred[6] = False
    assert wrapper.has_pending_deferred() is True
    assert wrapper.has_pending_deferred(6) is False
    assert wrapper.has_pending_deferred(1234) is False  # unseen layer, not a KeyError


def test_sync_cpu_only_returns_nothing_and_copies_nothing(wrapper):
    assert wrapper.sync_cpu_only() is None
    assert wrapper.cpu_infer.other_calls == []
