import pytest
import torch

from app.pipelines.device import DeviceSelector


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "expected"),
    [
        ("cpu", True, True, "cpu"),
        ("auto", True, False, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
    ],
)
def test_device_selection_honors_config(monkeypatch, requested, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert DeviceSelector(requested).resolve().type == expected


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_explicit_unavailable_device_fails(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="not available"):
        DeviceSelector(requested).resolve()
