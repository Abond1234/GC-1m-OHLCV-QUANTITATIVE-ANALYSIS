"""Machine-adaptive memory planning for the research pipelines.

Every full-population pipeline in this repository should run end to end on
any machine.  This module inspects physical memory at run time and returns a
processing plan: single-pass on machines with headroom, chronological
date-chunked processing as the fallback when the estimated peak does not fit.

The tier thresholds follow the project's operating guidance: machines with at
least 16 GB of total memory normally run every stage single-pass; 8 GB
machines fall back to chunked processing for the heavy Branch A stages; the
plan also respects *currently available* memory, so a loaded 16 GB machine
degrades gracefully instead of being killed by the operating system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import psutil

HIGH_MEMORY_TIER_GB = 16.0
MEDIUM_MEMORY_TIER_GB = 8.0
AVAILABLE_MEMORY_SAFETY_FRACTION = 0.6
MAX_CHUNKS = 16


@dataclass(frozen=True)
class MemoryPlan:
    """A processing decision derived from machine memory and workload size."""

    total_gb: float
    available_gb: float
    tier: str
    budget_gb: float
    estimated_peak_gb: float
    chunk_count: int

    @property
    def chunked(self) -> bool:
        return self.chunk_count > 1

    def describe(self) -> str:
        mode = f"chunked x{self.chunk_count}" if self.chunked else "single-pass"
        return (
            f"memory tier {self.tier} (total {self.total_gb:.1f} GB, "
            f"available {self.available_gb:.1f} GB, budget {self.budget_gb:.1f} GB); "
            f"estimated peak {self.estimated_peak_gb:.1f} GB -> {mode}"
        )


def machine_memory_gb() -> tuple[float, float]:
    """Total and currently available physical memory in GB."""

    memory = psutil.virtual_memory()
    return memory.total / 2**30, memory.available / 2**30


def memory_tier(total_gb: float) -> str:
    if total_gb >= HIGH_MEMORY_TIER_GB:
        return "high"
    if total_gb >= MEDIUM_MEMORY_TIER_GB:
        return "medium"
    return "low"


def plan_processing(
    estimated_peak_gb: float,
    *,
    total_gb: float | None = None,
    available_gb: float | None = None,
    safety_fraction: float = AVAILABLE_MEMORY_SAFETY_FRACTION,
    max_chunks: int = MAX_CHUNKS,
) -> MemoryPlan:
    """Decide between single-pass and chunked processing for a workload.

    ``estimated_peak_gb`` is the caller's estimate of single-pass peak memory.
    The budget is a safety fraction of available memory; the chunk count is
    the smallest split whose per-chunk peak fits the budget, capped so a
    pathological estimate cannot explode the chunk count.
    """

    if estimated_peak_gb <= 0:
        raise ValueError("estimated_peak_gb must be positive")
    if total_gb is None or available_gb is None:
        measured_total, measured_available = machine_memory_gb()
        total_gb = measured_total if total_gb is None else total_gb
        available_gb = measured_available if available_gb is None else available_gb
    budget_gb = max(available_gb * safety_fraction, 0.25)
    chunk_count = min(max(math.ceil(estimated_peak_gb / budget_gb), 1), max_chunks)
    return MemoryPlan(
        total_gb=float(total_gb),
        available_gb=float(available_gb),
        tier=memory_tier(total_gb),
        budget_gb=float(budget_gb),
        estimated_peak_gb=float(estimated_peak_gb),
        chunk_count=int(chunk_count),
    )


def chronological_chunks(dates, chunk_count: int) -> list:
    """Split sorted unique dates into contiguous chunks, whole days intact."""

    import numpy as np

    unique = np.unique(np.asarray(dates))
    if chunk_count <= 1 or len(unique) <= 1:
        return [unique]
    return [chunk for chunk in np.array_split(unique, chunk_count) if len(chunk)]
