"""Device detection and array backend utility — CPU vs CUDA GPU.

Checks for NVIDIA CUDA availability via cupy (preferred) or torch (fallback).
Provides a seamless array backend dispatcher (cupy when GPU is available, numpy when CPU only)
so that code runs accelerated on GPU when available and falls back gracefully to CPU for
environments without a GPU.

Usage:
    from src.statistical_research.device_utils import get_device, device_summary, get_array_module, to_numpy

    device = get_device()         # "cuda" | "cpu"
    xp = get_array_module()       # returns cupy module if CUDA available, else numpy module
    arr = xp.zeros((100, 100))    # runs on GPU if available, CPU if not
    np_arr = to_numpy(arr)        # safe conversion back to numpy ndarray
"""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np


def _check_cupy() -> tuple[bool, str]:
    """Return (available, version_or_reason)."""
    try:
        import cupy  # type: ignore[import-untyped]

        return True, cupy.__version__
    except ImportError:
        return False, "cupy not installed"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _check_torch() -> tuple[bool, str]:
    """Return (available, device_name_or_reason)."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, "torch present but CUDA not available"
    except ImportError:
        return False, "torch not installed"


def _nvidia_smi_name() -> str:
    """Best-effort GPU name from nvidia-smi (no Python GPU lib required)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def get_device() -> str:
    """Return 'cuda' if a usable GPU is detected, else 'cpu'."""
    cupy_ok, _ = _check_cupy()
    if cupy_ok:
        return "cuda"
    torch_ok, _ = _check_torch()
    if torch_ok:
        return "cuda"
    return "cpu"


def device_summary() -> str:
    """Return a one-line human-readable device description."""
    cupy_ok, cupy_info = _check_cupy()
    torch_ok, torch_info = _check_torch()

    if cupy_ok:
        gpu_name = _nvidia_smi_name()
        return f"CUDA (cupy {cupy_info}) — {gpu_name}"

    if torch_ok:
        return f"CUDA (torch) — {torch_info}"

    gpu_name = _nvidia_smi_name()
    if gpu_name != "unknown":
        return (
            f"CPU (GPU detected: {gpu_name}, but no cupy/torch installed — "
            "install cupy-cuda12x to enable GPU acceleration)"
        )

    return "CPU (no GPU detected)"


def get_array_module(prefer_gpu: bool = True) -> Any:
    """Return cupy module if CUDA is available and preferred, else numpy module."""
    if prefer_gpu and get_device() == "cuda":
        try:
            import cupy  # type: ignore[import-untyped]

            return cupy
        except Exception:  # noqa: BLE001
            pass
    return np


def to_numpy(arr: Any) -> np.ndarray:
    """Safely convert any array (cupy, numpy, pandas, list) to a numpy ndarray."""
    if arr is None:
        return np.array([])
    if hasattr(arr, "get"):
        # CuPy array -> NumPy array
        return arr.get()
    if hasattr(arr, "to_numpy"):
        # Pandas Series / Index -> NumPy array
        return arr.to_numpy()
    return np.asarray(arr)


def gpu_memory_free_bytes() -> int:
    """Return free GPU memory in bytes, or 0 if unavailable."""
    try:
        import cupy  # type: ignore[import-untyped]

        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:  # noqa: BLE001
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0]) * 1024 * 1024
    except Exception:  # noqa: BLE001
        pass
    return 0
