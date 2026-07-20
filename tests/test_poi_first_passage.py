from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.poi_first_passage import evaluate_first_passage


def _bars(highs, lows, closes=None, start="2024-01-03 09:00", segment=None, symbols=None):
    ny = pd.date_range(start, periods=len(highs), freq="min", tz="America/New_York")
    closes = np.asarray(
        closes if closes is not None else (np.asarray(highs) + np.asarray(lows)) / 2
    )
    segment = np.asarray(segment if segment is not None else np.ones(len(highs), dtype=int))
    symbols = symbols if symbols is not None else ["GCG4"] * len(highs)
    return pd.DataFrame(
        {
            "bar_id": np.arange(len(highs)),
            "ts_event_utc": ny.tz_convert("UTC"),
            "trade_date_ny": pd.Timestamp("2024-01-03"),
            "symbol": symbols,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "minute_of_day_ny": ny.hour * 60 + ny.minute,
            "continuous_segment_id": segment,
        }
    )


def _opportunity(side="long", entry=100.0, stop=99.0, entry_bar=0, model="boundary_touch"):
    return pd.DataFrame(
        {
            "true_trade_opportunity_id": ["T1"],
            "true_poi_id": ["P1"],
            "true_retest_id": ["R1"],
            "trade_date_ny": [pd.Timestamp("2024-01-03")],
            "retest_bar_id": [0],
            "entry_bar_id": [entry_bar],
            "entry_price": [entry],
            "stop_price": [stop],
            "trade_side": [side],
            "entry_model": [model],
        }
    )


class FirstPassageTests(unittest.TestCase):
    def test_target_before_stop(self):
        result = evaluate_first_passage(
            _opportunity(),
            _bars([100.2, 102.2, 100], [99.8, 100, 98.8]),
            target_rs=(2,),
            max_holding_minutes=2,
        )
        self.assertTrue(bool(result.loc[0, "target_hit_before_stop"]))

    def test_stop_before_target(self):
        result = evaluate_first_passage(
            _opportunity(),
            _bars([100.2, 100.5, 102.2], [99.8, 98.8, 100]),
            target_rs=(2,),
            max_holding_minutes=2,
        )
        self.assertTrue(bool(result.loc[0, "stop_hit_before_target"]))

    def test_same_bar_and_entry_bar_ambiguity_conservative(self):
        result = evaluate_first_passage(
            _opportunity(), _bars([102.2, 100], [98.8, 100]), target_rs=(2,), max_holding_minutes=1
        )
        self.assertTrue(bool(result.loc[0, "same_bar_ambiguity"]))
        self.assertTrue(bool(result.loc[0, "entry_bar_ambiguity"]))
        self.assertTrue(bool(result.loc[0, "stop_hit_before_target"]))
        self.assertEqual(result.loc[0, "realized_r"], -1.0)

    def test_ambiguity_excluded(self):
        result = evaluate_first_passage(
            _opportunity(),
            _bars([102.2, 100], [98.8, 100]),
            target_rs=(2,),
            max_holding_minutes=1,
            ambiguity_treatment="exclude",
        )
        self.assertTrue(pd.isna(result.loc[0, "realized_r"]))

    def test_forced_or_time_exit_before_either(self):
        result = evaluate_first_passage(
            _opportunity(),
            _bars([100.2, 100.4], [99.8, 99.7], closes=[100, 100.25]),
            target_rs=(2,),
            max_holding_minutes=1,
        )
        self.assertTrue(bool(result.loc[0, "forced_exit_before_either"]))

    def test_no_entry_after_noon(self):
        bars = _bars([100.2, 100.3], [99.8, 99.7], start="2024-01-03 12:01")
        result = evaluate_first_passage(_opportunity(), bars, target_rs=(2,))
        self.assertTrue(bool(result.loc[0, "invalid_path"]))

    def test_segment_boundary_invalidates_path(self):
        result = evaluate_first_passage(
            _opportunity(),
            _bars([100.2, 102.2], [99.8, 100], segment=[1, 2]),
            target_rs=(2,),
            max_holding_minutes=1,
        )
        self.assertTrue(bool(result.loc[0, "invalid_path"]))

    def test_contract_change_invalidates_path(self):
        bars = _bars([100.2, 102.2], [99.8, 100], symbols=["GCG4", "GCJ4"])
        result = evaluate_first_passage(_opportunity(), bars, target_rs=(2,), max_holding_minutes=1)
        self.assertTrue(bool(result.loc[0, "invalid_path"]))

    def test_short_side_is_symmetric(self):
        opportunity = _opportunity(side="short", entry=100.0, stop=101.0)
        result = evaluate_first_passage(
            opportunity,
            _bars([100.2, 100.0, 101.2], [99.8, 97.8, 99.0]),
            target_rs=(2,),
            max_holding_minutes=2,
        )
        self.assertTrue(bool(result.loc[0, "target_hit_before_stop"]))
        self.assertEqual(result.loc[0, "realized_r"], 2.0)

    def test_next_bar_market_entry(self):
        result = evaluate_first_passage(
            _opportunity(entry_bar=1, model="next_bar_confirmation"),
            _bars([99, 100.2, 102.2], [98, 99.8, 100]),
            target_rs=(2,),
            max_holding_minutes=1,
        )
        self.assertTrue(bool(result.loc[0, "entry_filled"]))
        self.assertEqual(result.loc[0, "time_to_entry_minutes"], 1)


if __name__ == "__main__":
    unittest.main()
