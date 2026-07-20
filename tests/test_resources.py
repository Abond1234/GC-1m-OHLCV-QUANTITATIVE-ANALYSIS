"""Tests for the machine-adaptive memory planner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.resources import (
    MemoryPlan,
    chronological_chunks,
    machine_memory_gb,
    memory_tier,
    plan_processing,
)


class MemoryPlanTests(unittest.TestCase):
    def test_tiers(self) -> None:
        self.assertEqual(memory_tier(32.0), "high")
        self.assertEqual(memory_tier(16.0), "high")
        self.assertEqual(memory_tier(12.0), "medium")
        self.assertEqual(memory_tier(7.9), "low")

    def test_high_memory_machine_runs_single_pass(self) -> None:
        plan = plan_processing(4.2, total_gb=32.0, available_gb=24.0)
        self.assertEqual(plan.tier, "high")
        self.assertFalse(plan.chunked)
        self.assertEqual(plan.chunk_count, 1)

    def test_eight_gb_machine_chunks_the_heavy_stage(self) -> None:
        plan = plan_processing(4.2, total_gb=8.0, available_gb=2.4)
        self.assertEqual(plan.tier, "medium")
        self.assertTrue(plan.chunked)
        self.assertGreaterEqual(plan.chunk_count, 3)
        self.assertLessEqual(plan.chunk_count, 16)

    def test_loaded_sixteen_gb_machine_degrades_gracefully(self) -> None:
        plan = plan_processing(4.2, total_gb=16.0, available_gb=3.0)
        self.assertEqual(plan.tier, "high")
        self.assertTrue(plan.chunked)

    def test_chunk_count_is_capped(self) -> None:
        plan = plan_processing(500.0, total_gb=8.0, available_gb=1.0)
        self.assertEqual(plan.chunk_count, 16)

    def test_invalid_estimate_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_processing(0.0, total_gb=8.0, available_gb=4.0)

    def test_measured_memory_is_positive(self) -> None:
        total, available = machine_memory_gb()
        self.assertGreater(total, 0.0)
        self.assertGreater(available, 0.0)
        self.assertGreaterEqual(total, available)

    def test_describe_mentions_mode(self) -> None:
        plan = plan_processing(1.0, total_gb=32.0, available_gb=20.0)
        self.assertIn("single-pass", plan.describe())
        chunked = plan_processing(40.0, total_gb=8.0, available_gb=4.0)
        self.assertIn("chunked", chunked.describe())


class ChronologicalChunkTests(unittest.TestCase):
    def test_chunks_preserve_order_and_cover_everything(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=10).to_numpy()
        chunks = chronological_chunks(dates, 3)
        recombined = np.concatenate(chunks)
        np.testing.assert_array_equal(recombined, np.unique(dates))
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertLess(earlier[-1], later[0])

    def test_single_chunk_when_not_chunked(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=5).to_numpy()
        chunks = chronological_chunks(dates, 1)
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], np.unique(dates))

    def test_whole_days_stay_intact(self) -> None:
        dates = np.repeat(pd.bdate_range("2023-01-02", periods=6).to_numpy(), 3)
        chunks = chronological_chunks(dates, 2)
        seen = set()
        for chunk in chunks:
            for date in chunk:
                self.assertNotIn(date, seen)
                seen.add(date)


if __name__ == "__main__":
    unittest.main()
