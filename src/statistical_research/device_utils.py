"""Compatibility shim: the compute backend now lives in :mod:`src.compute`.

The backend was moved up a package because it is used outside the statistical
research branch (the POI research modules, the execution layer, and the
notebooks all need the same device and memory policy).  This module re-exports
the public names so existing imports keep working; prefer importing from
``src.compute`` in new code.
"""

from __future__ import annotations

from ..compute import (
    ComputePlan,
    compute_mode,
    compute_plan,
    cpu_worker_count,
    device_summary,
    get_array_module,
    get_device,
    gpu_available,
    gpu_memory_free_bytes,
    parallel_map,
    plan_batches,
    to_numpy,
)

__all__ = [
    "ComputePlan",
    "compute_mode",
    "compute_plan",
    "cpu_worker_count",
    "device_summary",
    "get_array_module",
    "get_device",
    "gpu_available",
    "gpu_memory_free_bytes",
    "parallel_map",
    "plan_batches",
    "to_numpy",
]
