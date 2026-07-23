"""Tests for the portable compute backend.

These cover the two guarantees the project depends on: the backend can always
be pinned to a known device so two machines can be made to agree, and the CPU
parallelism helper never changes a result relative to the serial path.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from src.compute import (
    COMPUTE_MODE_ENV,
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


def _env(mode: str | None):
    environment = dict(os.environ)
    environment.pop(COMPUTE_MODE_ENV, None)
    if mode is not None:
        environment[COMPUTE_MODE_ENV] = mode
    return mock.patch.dict(os.environ, environment, clear=True)


class ComputeModeTests(unittest.TestCase):
    def test_defaults_to_auto_when_unset(self):
        with _env(None):
            self.assertEqual(compute_mode(), "auto")

    def test_accepts_documented_modes_case_insensitively(self):
        for raw, expected in (("CPU", "cpu"), (" gpu ", "gpu"), ("Auto", "auto")):
            with _env(raw):
                self.assertEqual(compute_mode(), expected)

    def test_rejects_unknown_mode(self):
        with _env("tpu"), self.assertRaises(ValueError):
            compute_mode()

    def test_cpu_mode_forces_cpu_device_and_numpy(self):
        with _env("cpu"):
            self.assertEqual(get_device(), "cpu")
            self.assertIs(get_array_module(), np)
            self.assertEqual(gpu_memory_free_bytes(), 0)

    def test_gpu_mode_raises_when_no_device_present(self):
        """Explicitly requesting the GPU must fail loudly, not degrade silently."""
        with _env("gpu"), mock.patch("src.compute.gpu_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                get_device()
            self.assertIn(COMPUTE_MODE_ENV, str(ctx.exception))

    def test_auto_mode_uses_gpu_only_when_available(self):
        with _env("auto"), mock.patch("src.compute.gpu_available", return_value=False):
            self.assertEqual(get_device(), "cpu")
        with _env("auto"), mock.patch("src.compute.gpu_available", return_value=True):
            self.assertEqual(get_device(), "cuda")

    def test_prefer_gpu_false_always_returns_numpy(self):
        """The deterministic research path must never be handed a device module."""
        with _env("auto"), mock.patch("src.compute.gpu_available", return_value=True):
            self.assertIs(get_array_module(prefer_gpu=False), np)


class ToNumpyTests(unittest.TestCase):
    def test_converts_common_inputs(self):
        self.assertEqual(to_numpy(None).size, 0)
        np.testing.assert_array_equal(to_numpy([1, 2, 3]), np.array([1, 2, 3]))
        source = np.arange(4.0)
        np.testing.assert_array_equal(to_numpy(source), source)

    def test_converts_pandas_series(self):
        import pandas as pd

        np.testing.assert_array_equal(to_numpy(pd.Series([1.0, 2.0])), np.array([1.0, 2.0]))


class CpuWorkerTests(unittest.TestCase):
    def test_detected_count_is_at_least_one(self):
        self.assertGreaterEqual(cpu_worker_count(), 1)

    def test_explicit_request_is_honoured(self):
        self.assertEqual(cpu_worker_count(3), 3)

    def test_rejects_non_positive_request(self):
        with self.assertRaises(ValueError):
            cpu_worker_count(0)

    def test_low_memory_host_is_capped(self):
        with mock.patch("src.compute.machine_memory_gb", return_value=(4.0, 3.0)):
            self.assertLessEqual(cpu_worker_count(), 2)


class ParallelMapTests(unittest.TestCase):
    def test_preserves_input_order(self):
        items = list(range(50))
        self.assertEqual(parallel_map(lambda x: x * x, items, workers=4), [x * x for x in items])

    def test_matches_serial_result_exactly(self):
        rng = np.random.default_rng(20260723)
        blocks = [rng.normal(size=500) for _ in range(16)]

        def statistic(block: np.ndarray) -> float:
            return float(block.mean())

        serial = [statistic(block) for block in blocks]
        parallel = parallel_map(statistic, blocks, workers=4)
        # Bit-for-bit: splitting independent units must not perturb any value.
        self.assertEqual(parallel, serial)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(parallel_map(lambda x: x, []), [])

    def test_single_worker_falls_back_to_serial(self):
        self.assertEqual(parallel_map(lambda x: x + 1, [1, 2, 3], workers=1), [2, 3, 4])


class PlanBatchesTests(unittest.TestCase):
    def test_batches_cover_every_item_exactly_once(self):
        batches = plan_batches(1000, bytes_per_item=1024)
        self.assertEqual(batches[0][0], 0)
        self.assertEqual(batches[-1][1], 1000)
        # Pairwise over consecutive batches, so the offset sequence is shorter
        # by one by construction.
        for (_, previous_stop), (next_start, _) in zip(batches, batches[1:], strict=False):
            self.assertEqual(previous_stop, next_start)
        self.assertEqual(sum(stop - start for start, stop in batches), 1000)

    def test_zero_items_produces_no_batches(self):
        self.assertEqual(plan_batches(0, bytes_per_item=8), [])

    def test_small_device_budget_produces_more_batches(self):
        """A 4 GB card must split the work rather than fail on it."""
        with mock.patch("src.compute.get_device", return_value="cuda"):
            with mock.patch("src.compute.gpu_memory_free_bytes", return_value=8 * 1024**2):
                small = plan_batches(10_000, bytes_per_item=4096, max_batches=0)
            with mock.patch("src.compute.gpu_memory_free_bytes", return_value=8 * 1024**3):
                large = plan_batches(10_000, bytes_per_item=4096, max_batches=0)
        self.assertGreater(len(small), len(large))
        self.assertEqual(sum(stop - start for start, stop in small), 10_000)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            plan_batches(10, bytes_per_item=0)
        with self.assertRaises(ValueError):
            plan_batches(-1, bytes_per_item=8)


class ComputePlanTests(unittest.TestCase):
    def test_plan_reports_the_resolved_environment(self):
        with _env("cpu"):
            plan = compute_plan()
        self.assertIsInstance(plan, ComputePlan)
        self.assertEqual(plan.device, "cpu")
        self.assertEqual(plan.mode, "cpu")
        self.assertGreaterEqual(plan.cpu_workers, 1)
        self.assertGreater(plan.total_memory_gb, 0.0)
        self.assertIn(plan.memory_tier, {"low", "medium", "high"})

    def test_describe_is_a_useful_one_liner(self):
        with _env("cpu"):
            text = device_summary()
        self.assertIn("device cpu", text)
        self.assertIn("CPU worker", text)
        self.assertEqual(text.count("\n"), 0)

    def test_gpu_available_returns_a_bool_without_raising(self):
        self.assertIsInstance(gpu_available(), bool)


if __name__ == "__main__":
    unittest.main()
