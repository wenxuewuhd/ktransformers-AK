"""set_capture_probe: a host framework's capture probe decides whether _wait_device syncs.

CPU only: kt_kernel_ext is stubbed and torch.npu is faked, so this runs without
the compiled extension or an NPU. Run: python -m pytest test/test_capture_probe.py
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def eb(monkeypatch):
    """A fresh experts_base loaded from the source tree against a stub extension."""
    pkg = types.ModuleType("kt_kernel")
    pkg.__path__ = []
    pkg.kt_kernel_ext = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "kt_kernel", pkg)
    spec = importlib.util.spec_from_file_location(
        "kt_capture_probe_under_test", os.path.join(_HERE, "..", "python", "experts_base.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # sglang is not importable here, so the default probe answers False
    monkeypatch.setattr(mod, "_sglang_is_capture_mode", lambda: False)
    return mod


class FakeNPU:
    """torch.npu stand-in whose own capture test is unreliable (always False)."""

    def __init__(self):
        self.syncs = 0

    def is_current_stream_capturing(self):
        return False

    def synchronize(self, device=None):
        self.syncs += 1


@pytest.fixture
def npu(monkeypatch):
    fake = FakeNPU()
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    return fake


DEV = types.SimpleNamespace(type="npu")


def test_default_is_sglang_compatible(eb, npu):
    assert eb.get_capture_probe() is None
    eb._wait_device(DEV)
    assert npu.syncs == 1  # not capturing -> sync, as before


def test_installed_probe_suppresses_sync_during_capture(eb, npu):
    capturing = {"on": True}
    eb.set_capture_probe(lambda: capturing["on"])
    eb._wait_device(DEV)
    assert npu.syncs == 0
    capturing["on"] = False
    eb._wait_device(DEV)
    assert npu.syncs == 1
    eb.set_capture_probe(None)  # restore default
    assert eb.get_capture_probe() is None


def test_probe_errors_propagate(eb, npu):
    def broken():
        raise RuntimeError("probe cannot tell")

    eb.set_capture_probe(broken)
    with pytest.raises(RuntimeError):
        eb._wait_device(DEV)
    assert npu.syncs == 0  # never "assume not capturing" and sync
    with pytest.raises(TypeError):
        eb.set_capture_probe(42)


def test_mutation_arm_probe_ignored_is_red(eb, npu):
    """A _wait_device that only consults sglang (the pre-patch code) syncs under capture."""
    eb.set_capture_probe(lambda: True)
    mutant_syncs_before = npu.syncs

    def mutant_wait_device(device):
        if device.type == "npu":
            if torch.npu.is_current_stream_capturing():
                return
            if eb._sglang_is_capture_mode():
                return
            torch.npu.synchronize(device)

    mutant_wait_device(DEV)
    assert npu.syncs == mutant_syncs_before + 1  # mutant would sync on a captured stream
    eb._wait_device(DEV)
    assert npu.syncs == mutant_syncs_before + 1  # real code does not
