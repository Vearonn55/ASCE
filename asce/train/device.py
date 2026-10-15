"""Pick a torch device, with an MPS sanity check and a CPU fallback."""

from __future__ import annotations

import torch


def mps_is_usable() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        return False
    try:
        value = torch.ones(2, 2, device="mps")
        _ = value @ value
        return True
    except Exception:
        return False


def resolve_device(prefer: str = "auto") -> torch.device:
    """``auto`` uses MPS when a tiny matmul succeeds, else CPU.

    Pass ``cpu`` to force the fallback path used to sanity-check MPS numerics.
    """

    choice = (prefer or "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice in ("auto", "mps") and mps_is_usable():
        return torch.device("mps")
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "mps" and not mps_is_usable():
        return torch.device("cpu")
    return torch.device("cpu")


def sanity_check_device() -> dict:
    """Compare a small conv on CPU and, when present, MPS.

    Large absolute differences mean the MPS path is not trustworthy for an
    experiment and it should be re-run with ``--device cpu``.
    """

    layer = torch.nn.Conv1d(1, 2, kernel_size=3, padding=1)
    data = torch.randn(1, 1, 16)
    with torch.no_grad():
        cpu_out = layer(data)
        report: dict = {
            "resolved": str(resolve_device("auto")),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "mps_usable": mps_is_usable(),
            "max_abs_diff": None,
        }
        if report["mps_usable"]:
            layer_mps = layer.to("mps")
            mps_out = layer_mps(data.to("mps")).cpu()
            report["max_abs_diff"] = float((cpu_out - mps_out).abs().max())
    return report
