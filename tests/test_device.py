"""Tests for loud device resolution and honest timing."""

import pytest
import torch

from tinylm.device import Device, Error


def test_cpu_resolves_and_unknown_names_are_rejected():
    assert Device("cpu").resolve() == torch.device("cpu")
    for bad in ("tpu", "", "cuda:x", "mps:1"):
        with pytest.raises(Error, match="device"):
            Device(bad).resolve()


def test_unavailable_accelerators_fail_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    with pytest.raises(Error, match="MPS is built but not available"):
        Device("mps").resolve()
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    with pytest.raises(Error, match="not built"):
        Device("mps").resolve()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(Error, match="CUDA"):
        Device("cuda").resolve()


def test_silent_cpu_fallback_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(Error, match="fallback"):
        Device("mps").resolve()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert Device("mps").resolve() == torch.device("mps")


def test_sync_and_release_are_safe_on_cpu_and_do_not_raise():
    Device("cpu").sync()
    Device("cpu").release()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_sync_and_release_run_on_the_real_accelerator():
    device = Device("mps")
    x = torch.ones(1000, 1000, device=device.resolve())
    (x @ x).sum()
    device.sync()
    device.release()
