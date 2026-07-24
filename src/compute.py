"""Portable compute backend: CUDA GPU when available, CPU otherwise.

Every pipeline in this repository must produce the *same research numbers* on
every machine, and must run on machines with or without a GPU and with widely
different amounts of memory.  Those two goals interact, so this module states
the policy explicitly and provides the primitives that enforce it.

Determinism policy
------------------
Floating-point reductions (``cumsum``, ``sum``, ``mean``) are not associative.
A GPU scan and a CPU scan over the same ``float64`` array can differ in the
last bits, and different NumPy builds can differ from each other too.  For a
frozen research artefact that is unacceptable: the feature matrix, the
labels, and every partition verdict derived from them must not depend on which
machine executed them.

Therefore:

* **Research-artefact numerics run on the deterministic CPU path.**  They are
  computed with a single, fixed implementation regardless of the host.  This
  is what makes the committed feature matrix reproducible between a CPU-only
  laptop and a CUDA workstation.
* **The GPU is used for work whose result cannot move**, and for exploratory or
  diagnostic work that no verdict depends on.  Callers opt in explicitly.
* **CPU parallelism is used where it is provably exact** - that is, where the
  work splits into independent units whose individual results are unchanged by
  the split and whose collection order is restored.  ``parallel_map`` below is
  order-preserving for exactly this reason.

Selecting a backend
-------------------
The ``PROJECT_COMPUTE`` environment variable overrides detection:

``auto`` (default)
    Use CUDA when a working GPU array module is importable, otherwise CPU.
``cpu``
    Never use the GPU.  Set this to reproduce a colleague's CPU-only results.
``gpu``
    Require CUDA; raise if it is unavailable rather than silently degrading.

Usage::

    from src.compute import compute_plan, get_array_module, parallel_map, to_numpy

    print(compute_plan().describe())      # one-line environment banner
    xp = get_array_module()               # cupy when enabled, else numpy
    results = parallel_map(fn, items)     # order-preserving CPU parallelism
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import psutil

from .resources import machine_memory_gb, memory_tier

T = TypeVar("T")
R = TypeVar("R")

COMPUTE_MODE_ENV = "PROJECT_COMPUTE"
VALID_COMPUTE_MODES = ("auto", "cpu", "gpu")

#: Fraction of free device memory a single batch may occupy.
DEVICE_MEMORY_SAFETY_FRACTION = 0.5
#: Fraction of available host memory a single batch may occupy.
HOST_MEMORY_SAFETY_FRACTION = 0.25
#: Workers are capped here so a many-core host does not thrash on memory.
MAX_CPU_WORKERS = 8


def compute_mode() -> str:
    """Return the requested backend mode from the environment (never cached)."""

    mode = os.environ.get(COMPUTE_MODE_ENV, "auto").strip().lower() or "auto"
    if mode not in VALID_COMPUTE_MODES:
        raise ValueError(
            f"{COMPUTE_MODE_ENV} must be one of {VALID_COMPUTE_MODES}, received {mode!r}"
        )
    return mode


@lru_cache(maxsize=1)
def _probe_cupy() -> tuple[bool, str]:
    """Return (usable, detail). Cached: probing imports is not free."""

    try:
        import cupy  # type: ignore[import-untyped]

        # Importing cupy succeeds on hosts with no driver; a tiny allocation is
        # the cheapest honest proof that the runtime actually works.
        cupy.zeros(1)
        return True, f"cupy {cupy.__version__}"
    except ImportError:
        return False, "cupy not installed"
    except Exception as exc:  # noqa: BLE001 - any runtime/driver failure means unusable
        return False, f"cupy present but unusable ({type(exc).__name__})"


@lru_cache(maxsize=1)
def _probe_torch() -> tuple[bool, str]:
    """Return (usable, detail) for a torch CUDA build. Cached."""

    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return False, "torch not installed"
    try:
        if torch.cuda.is_available():
            return True, f"torch {torch.__version__} ({torch.cuda.get_device_name(0)})"
        return False, "torch present but CUDA unavailable"
    except Exception as exc:  # noqa: BLE001
        return False, f"torch present but unusable ({type(exc).__name__})"


@lru_cache(maxsize=1)
def _nvidia_smi_name() -> str:
    """Best-effort GPU name from nvidia-smi. Cached; absent tool is normal."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    return "unknown"


def gpu_available() -> bool:
    """True when a GPU array module is importable and its runtime works."""

    return _probe_cupy()[0] or _probe_torch()[0]


def get_device() -> str:
    """Return the active device, ``"cuda"`` or ``"cpu"``, honouring the override."""

    mode = compute_mode()
    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        if not gpu_available():
            raise RuntimeError(
                f"{COMPUTE_MODE_ENV}=gpu was requested but no usable CUDA device was found "
                f"({_probe_cupy()[1]}; {_probe_torch()[1]}). "
                f"Unset {COMPUTE_MODE_ENV} to fall back to CPU automatically."
            )
        return "cuda"
    return "cuda" if gpu_available() else "cpu"


def get_array_module(prefer_gpu: bool = True) -> Any:
    """Return ``cupy`` when the GPU backend is active and preferred, else ``numpy``.

    ``prefer_gpu=False`` is the explicit request for the deterministic CPU path
    and is what research-artefact numerics use.
    """

    if not prefer_gpu:
        return np
    if get_device() != "cuda":
        return np
    usable, _ = _probe_cupy()
    if not usable:
        return np
    import cupy  # type: ignore[import-untyped]

    return cupy


def to_numpy(array: Any) -> np.ndarray:
    """Convert any array-like (cupy, numpy, pandas, sequence) to a NumPy array."""

    if array is None:
        return np.array([])
    if hasattr(array, "get") and type(array).__module__.startswith("cupy"):
        return array.get()
    if hasattr(array, "to_numpy"):
        return array.to_numpy()
    return np.asarray(array)


def gpu_memory_free_bytes() -> int:
    """Free device memory in bytes, or 0 when no usable GPU is present."""

    if get_device() != "cuda":
        return 0
    if _probe_cupy()[0]:
        try:
            import cupy  # type: ignore[import-untyped]

            free, _total = cupy.cuda.runtime.memGetInfo()
            return int(free)
        except Exception:  # noqa: BLE001
            pass
    if _probe_torch()[0]:
        try:
            import torch  # type: ignore[import-untyped]

            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except Exception:  # noqa: BLE001
            pass
    return 0


def cpu_worker_count(requested: int | None = None) -> int:
    """Worker count for CPU parallelism, bounded by cores, memory tier and cap.

    A 4 GB host is not given eight workers just because it reports eight
    logical cores; memory, not cores, is the binding constraint here.
    """

    if requested is not None:
        if requested < 1:
            raise ValueError("requested worker count must be at least 1")
        return requested
    cores = os.cpu_count() or 1
    total_gb, available_gb = machine_memory_gb()
    memory_workers = max(int(available_gb // 1.0), 1)
    workers = min(cores, memory_workers, MAX_CPU_WORKERS)
    if memory_tier(total_gb) == "low":
        workers = min(workers, 2)
    return max(workers, 1)


def parallel_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int | None = None,
    ordered: bool = True,
) -> list[R]:
    """Apply ``function`` across ``items`` using threads, preserving input order.

    This is safe for research numerics only because each unit of work is
    independent and its own result is unaffected by the split; restoring input
    order then makes the output identical to the serial computation.  Do not
    use it to split a single reduction.

    Falls back to a serial loop when one worker is selected, which keeps
    single-core and memory-constrained hosts predictable.
    """

    materialised: Sequence[T] = list(items)
    if not materialised:
        return []
    worker_count = cpu_worker_count(workers)
    if worker_count == 1 or len(materialised) == 1:
        return [function(item) for item in materialised]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        if ordered:
            return list(pool.map(function, materialised))
        return list(pool.map(function, materialised))


def plan_batches(
    item_count: int,
    bytes_per_item: int,
    *,
    device: str | None = None,
    max_batches: int = 512,
) -> list[tuple[int, int]]:
    """Split ``item_count`` into (start, stop) batches that fit available memory.

    ``bytes_per_item`` is the caller's estimate of peak working bytes for one
    item.  On the GPU the budget comes from free device memory, which is what
    lets a 4 GB card process the same workload a 32 GB host does - in more,
    smaller batches rather than by failing.
    """

    if item_count < 0:
        raise ValueError("item_count must not be negative")
    if bytes_per_item <= 0:
        raise ValueError("bytes_per_item must be positive")
    if item_count == 0:
        return []

    active = device or get_device()
    if active == "cuda":
        free_bytes = gpu_memory_free_bytes()
        budget = int(free_bytes * DEVICE_MEMORY_SAFETY_FRACTION)
    else:
        budget = 0
    if budget <= 0:
        available_bytes = int(psutil.virtual_memory().available)
        budget = int(available_bytes * HOST_MEMORY_SAFETY_FRACTION)

    batch_size = max(int(budget // bytes_per_item), 1)
    batch_size = min(batch_size, item_count)
    if max_batches > 0:
        minimum_size = max(-(-item_count // max_batches), 1)
        batch_size = max(batch_size, minimum_size)
    return [
        (start, min(start + batch_size, item_count)) for start in range(0, item_count, batch_size)
    ]


@dataclass(frozen=True)
class ComputePlan:
    """Resolved description of the machine this process is running on."""

    mode: str
    device: str
    device_detail: str
    cpu_workers: int
    total_memory_gb: float
    available_memory_gb: float
    memory_tier: str
    gpu_free_memory_gb: float

    def describe(self) -> str:
        line = (
            f"compute mode {self.mode} -> device {self.device} ({self.device_detail}); "
            f"{self.cpu_workers} CPU worker(s); memory tier {self.memory_tier} "
            f"(total {self.total_memory_gb:.1f} GB, available {self.available_memory_gb:.1f} GB)"
        )
        if self.device == "cuda":
            line += f"; GPU free {self.gpu_free_memory_gb:.2f} GB"
        return line


def compute_plan() -> ComputePlan:
    """Inspect the host and return the resolved compute plan."""

    device = get_device()
    cupy_ok, cupy_detail = _probe_cupy()
    torch_ok, torch_detail = _probe_torch()
    if device == "cuda":
        detail = cupy_detail if cupy_ok else torch_detail
        gpu_name = _nvidia_smi_name()
        if gpu_name != "unknown":
            detail = f"{detail}, {gpu_name}"
    else:
        gpu_name = _nvidia_smi_name()
        if gpu_name != "unknown" and not (cupy_ok or torch_ok):
            detail = f"no GPU array module installed (device present: {gpu_name})"
        elif compute_mode() == "cpu":
            detail = "CPU forced by environment"
        else:
            detail = "no CUDA device detected"
    total_gb, available_gb = machine_memory_gb()
    return ComputePlan(
        mode=compute_mode(),
        device=device,
        device_detail=detail,
        cpu_workers=cpu_worker_count(),
        total_memory_gb=total_gb,
        available_memory_gb=available_gb,
        memory_tier=memory_tier(total_gb),
        gpu_free_memory_gb=gpu_memory_free_bytes() / 2**30,
    )


def device_summary() -> str:
    """One-line human-readable description of the active compute device."""

    return compute_plan().describe()
