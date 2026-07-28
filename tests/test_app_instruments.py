"""Instrument registry, availability probing, and the external data contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.datalayer.instruments import (
    EXTERNAL_REQUIRED_COLUMNS,
    REGISTRY,
    Instrument,
    available_instruments,
    data_available,
    external_path,
    load_instrument_bars,
)
from src.app.datalayer.paths import research_bars_path

_HAS_RESEARCH_BARS = research_bars_path().exists()


def _external_frame(n=120, date="2024-06-03"):
    utc = pd.date_range("2024-06-03 12:00:00+00:00", periods=n, freq="min")
    return pd.DataFrame(
        {
            "ts_event_utc": utc,
            "open": np.linspace(100.0, 101.0, n),
            "high": np.linspace(100.2, 101.2, n),
            "low": np.linspace(99.8, 100.8, n),
            "close": np.linspace(100.1, 101.1, n),
            "volume": np.full(n, 25.0),
            "trade_date_ny": pd.Timestamp(date),
            "minute_of_day_ny": np.arange(480, 480 + n, dtype=np.int64),
            "continuous_segment_id": np.ones(n, dtype=np.int64),
            "rolling_atr_20m": np.full(n, 0.5),
        }
    )


class RegistryTests(unittest.TestCase):
    def test_registry_carries_correct_contract_economics(self):
        self.assertEqual(REGISTRY["GC"].spec.dollars_per_point, 100.0)
        self.assertEqual(REGISTRY["MGC"].spec.dollars_per_point, 10.0)
        self.assertEqual(REGISTRY["NQ"].spec.dollars_per_point, 20.0)
        self.assertEqual(REGISTRY["NQ"].spec.tick_size, 0.25)
        self.assertEqual(REGISTRY["ES"].spec.dollars_per_point, 50.0)
        self.assertEqual(REGISTRY["ES"].spec.tick_size, 0.25)

    def test_only_gc_is_research_grade(self):
        research_grade = [s for s, inst in REGISTRY.items() if inst.research_grade]
        self.assertEqual(research_grade, ["GC"])

    def test_every_instrument_documents_itself(self):
        for instrument in REGISTRY.values():
            self.assertIsInstance(instrument, Instrument)
            self.assertTrue(instrument.notes)
            self.assertIn(instrument.source, ("research", "external"))


class AvailabilityTests(unittest.TestCase):
    def test_unknown_symbol_is_unavailable(self):
        self.assertFalse(data_available("DOGE"))

    def test_external_without_file_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(data_available("NQ", Path(tmp)))

    @unittest.skipUnless(_HAS_RESEARCH_BARS, "research bars parquet not present")
    def test_research_products_available_with_local_data(self):
        symbols = [inst.symbol for inst in available_instruments()]
        self.assertIn("GC", symbols)
        self.assertIn("MGC", symbols)

    def test_external_becomes_available_when_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = external_path("NQ", Path(tmp))
            path.parent.mkdir(parents=True, exist_ok=True)
            _external_frame().to_parquet(path)
            self.assertTrue(data_available("NQ", Path(tmp)))


class ExternalContractTests(unittest.TestCase):
    def test_unknown_instrument_raises(self):
        with self.assertRaises(KeyError):
            load_instrument_bars("DOGE")

    def test_missing_file_raises_with_contract_pointer(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_instrument_bars("ES")
        self.assertIn("Adding an instrument", str(ctx.exception))

    def test_missing_columns_refused_loudly(self):
        import src.app.datalayer.instruments as instruments_module

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "NQ_1m.parquet"
            frame = _external_frame().drop(columns=["rolling_atr_20m"])
            frame.to_parquet(path)
            original = instruments_module.external_path
            instruments_module.external_path = lambda symbol, root=None: path
            try:
                with self.assertRaises(ValueError) as ctx:
                    load_instrument_bars("NQ")
                self.assertIn("rolling_atr_20m", str(ctx.exception))
            finally:
                instruments_module.external_path = original

    def test_valid_external_file_loads_sorted_bars(self):
        import src.app.datalayer.instruments as instruments_module

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "NQ_1m.parquet"
            frame = _external_frame().iloc[::-1]  # deliberately unsorted
            frame.to_parquet(path)
            original = instruments_module.external_path
            instruments_module.external_path = lambda symbol, root=None: path
            try:
                store = instruments_module.load_instrument_bars("NQ")
            finally:
                instruments_module.external_path = original
            self.assertEqual(store.n_bars, 120)
            self.assertTrue((np.diff(store.minute_ny) == 1).all())
            self.assertEqual(len(EXTERNAL_REQUIRED_COLUMNS), 10)


if __name__ == "__main__":
    unittest.main()
