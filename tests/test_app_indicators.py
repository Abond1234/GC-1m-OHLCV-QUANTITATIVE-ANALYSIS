"""Indicator engine correctness: warmups, known values, and registry sanity."""

from __future__ import annotations

import unittest

import numpy as np

from src.app.analysis import indicators as ind


class IndicatorMathTests(unittest.TestCase):
    def test_sma_known_values_and_warmup(self):
        out = ind.sma([1, 2, 3, 4, 5], 3)
        self.assertTrue(np.isnan(out[:2]).all())
        np.testing.assert_allclose(out[2:], [2.0, 3.0, 4.0])

    def test_sma_short_input_is_all_nan(self):
        self.assertTrue(np.isnan(ind.sma([1.0, 2.0], 5)).all())

    def test_ema_seeds_with_sma_then_smooths(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = ind.ema(x, 3)
        self.assertTrue(np.isnan(out[:2]).all())
        self.assertAlmostEqual(out[2], 2.0)  # SMA seed
        self.assertAlmostEqual(out[3], 0.5 * 4.0 + 0.5 * 2.0)  # alpha = 2/(n+1) = 0.5
        self.assertAlmostEqual(out[4], 0.5 * 5.0 + 0.5 * 3.0)

    def test_bollinger_bands_are_symmetric_about_mid(self):
        x = np.linspace(100, 110, 40) + np.sin(np.arange(40))
        bands = ind.bollinger(x, 10, 2.0)
        valid = ~np.isnan(bands["mid"])
        np.testing.assert_allclose(
            bands["upper"][valid] - bands["mid"][valid],
            bands["mid"][valid] - bands["lower"][valid],
        )
        self.assertTrue((bands["upper"][valid] >= bands["lower"][valid]).all())

    def test_true_range_covers_gaps(self):
        # Second bar gaps above the first close: TR must use |high - prev close|.
        high = np.array([10.0, 15.0])
        low = np.array([9.0, 14.5])
        close = np.array([9.5, 14.8])
        tr = ind.true_range(high, low, close)
        self.assertAlmostEqual(tr[0], 1.0)
        self.assertAlmostEqual(tr[1], 15.0 - 9.5)

    def test_atr_is_simple_mean_of_true_range(self):
        high = np.array([10.0, 11.0, 12.0, 13.0])
        low = np.array([9.0, 10.0, 11.0, 12.0])
        close = np.array([9.5, 10.5, 11.5, 12.5])
        out = ind.atr(high, low, close, 2)
        tr = ind.true_range(high, low, close)
        self.assertAlmostEqual(out[1], np.mean(tr[:2]))
        self.assertAlmostEqual(out[3], np.mean(tr[2:4]))

    def test_rsi_bounds_and_direction(self):
        up = ind.rsi(np.arange(30, dtype=float), 14)
        self.assertAlmostEqual(up[-1], 100.0)
        down = ind.rsi(np.arange(30, 0, -1, dtype=float), 14)
        self.assertLess(down[-1], 1.0)
        mixed = ind.rsi(100 + np.sin(np.arange(60)), 14)
        valid = mixed[~np.isnan(mixed)]
        self.assertTrue(((valid >= 0) & (valid <= 100)).all())

    def test_rolling_volatility_flat_series_is_zero(self):
        out = ind.rolling_volatility(np.full(20, 50.0), 5)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0)

    def test_obv_accumulates_signed_volume(self):
        close = np.array([10.0, 11.0, 10.5, 10.5, 12.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        np.testing.assert_allclose(ind.obv(close, volume), [0.0, 200.0, -100.0, -100.0, 400.0])

    def test_registry_and_compute_agree(self):
        n = 60
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        ohlc = {
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": rng.uniform(50, 150, n),
        }
        for key, definition in ind.INDICATORS.items():
            self.assertIn(definition.pane, ("price", "osc", "volume"))
            for p in definition.params:
                self.assertTrue(p.lo <= p.default <= p.hi)
            if definition.source is not None:
                continue  # data-sourced (VWAP variants): nothing to compute here
            params = {p.name: p.default for p in definition.params}
            series = ind.compute(key, params, ohlc)
            self.assertEqual(set(series), set(definition.outputs))
            for arr in series.values():
                self.assertEqual(len(arr), n)

    def test_instance_label_shows_parameters(self):
        inst = ind.IndicatorInstance(1, "ema", {"n": 21})
        self.assertEqual(inst.display_label(), "EMA 21")
        inst2 = ind.IndicatorInstance(2, "vwap_day")
        self.assertEqual(inst2.display_label(), "VWAP day")


if __name__ == "__main__":
    unittest.main()
