"""Display-timeframe resampling and the ViewMap 1m <-> displayed-bar mapping."""

from __future__ import annotations

import unittest

import numpy as np

from src.app.datalayer.timeframe import (
    TIMEFRAMES,
    ViewMap,
    bucket_labels,
    resample_window,
    sample_first,
    sample_last,
)


class _Bars:
    """Minimal BarStore stand-in with the attributes the resampler reads."""

    def __init__(self, minute, dates, seg):
        n = len(minute)
        self.minute_ny = np.asarray(minute, dtype=np.int64)
        self.trade_date = np.asarray(dates)
        self.segment = np.asarray(seg)
        self.open = np.arange(n, dtype=np.float64) + 100.0
        self.high = self.open + 0.5
        self.low = self.open - 0.5
        self.close = self.open + 0.2
        self.volume = np.full(n, 10.0)


def _day(n, start_minute=420, date="2024-06-03", seg=1):
    return (
        np.arange(start_minute, start_minute + n),
        np.full(n, np.datetime64(date)),
        np.full(n, seg),
    )


class ResampleTests(unittest.TestCase):
    def test_ohlcv_reductions_match_naive_loop(self):
        m, d, s = _day(23)
        bars = _Bars(m, d, s)
        rs = resample_window(bars, 0, 22, 5)
        # minutes 420..442: quotient buckets 84(420-424), 85, 86, 87, 88(440-442)
        self.assertEqual(len(rs.open), 5)
        for b in range(len(rs.open)):
            lo, hi = rs.bucket_starts[b], rs.bucket_ends[b]
            sl = slice(lo, hi + 1)
            self.assertEqual(rs.open[b], bars.open[sl][0])
            self.assertEqual(rs.close[b], bars.close[sl][-1])
            self.assertEqual(rs.high[b], bars.high[sl].max())
            self.assertEqual(rs.low[b], bars.low[sl].min())
            self.assertEqual(rs.volume[b], bars.volume[sl].sum())

    def test_bucket_breaks_at_segment_change(self):
        m, d, s = _day(10)
        s = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
        bars = _Bars(m, d, s)
        rs = resample_window(bars, 0, 9, 5)
        # 420-422 seg1 | 423-424 seg2 | 425-429 seg2
        self.assertEqual(list(rs.bucket_starts), [0, 3, 5])
        self.assertEqual(list(rs.segment), [1, 2, 2])

    def test_bucket_breaks_at_date_change_with_equal_quotient(self):
        # Two dates, same minute values -> same quotient, must still break.
        m1, d1, s1 = _day(5, start_minute=420, date="2024-06-03")
        m2, d2, s2 = _day(5, start_minute=420, date="2024-06-04")
        bars = _Bars(np.concatenate([m1, m2]), np.concatenate([d1, d2]), np.concatenate([s1, s2]))
        rs = resample_window(bars, 0, 9, 60)
        self.assertEqual(list(rs.bucket_starts), [0, 5])

    def test_daily_is_one_bucket_per_date_segment_run(self):
        m1, d1, s1 = _day(30, date="2024-06-03")
        m2, d2, s2 = _day(30, date="2024-06-04")
        bars = _Bars(np.concatenate([m1, m2]), np.concatenate([d1, d2]), np.concatenate([s1, s2]))
        rs = resample_window(bars, 0, 59, TIMEFRAMES["1D"])
        self.assertEqual(len(rs.open), 2)

    def test_midnight_wrap_buckets_by_quotient_change(self):
        # 18:00 evening open: minutes 1438,1439 then 0,1 next NY-clock day but the
        # SAME trade date; quotient changes at the wrap so the bucket splits.
        m = np.array([1438, 1439, 0, 1])
        d = np.full(4, np.datetime64("2024-06-03"))
        bars = _Bars(m, d, np.ones(4))
        rs = resample_window(bars, 0, 3, 5)
        self.assertEqual(list(rs.bucket_starts), [0, 2])

    def test_labels_day_prefix_rules(self):
        m1, d1, s1 = _day(10, date="2024-06-03")
        m2, d2, s2 = _day(10, date="2024-06-04")
        bars = _Bars(np.concatenate([m1, m2]), np.concatenate([d1, d2]), np.concatenate([s1, s2]))
        rs = resample_window(bars, 0, 19, 5)
        single = bucket_labels(bars, resample_window(bars, 0, 9, 5), 5, multi_day=False)
        self.assertEqual(single[0], "07:00")
        multi = bucket_labels(bars, rs, 5, multi_day=True)
        self.assertEqual(multi[0], "06-03 07:00")
        daily = bucket_labels(bars, resample_window(bars, 0, 19, 1440), 1440, multi_day=True)
        self.assertEqual(list(daily), ["06-03", "06-04"])

    def test_display_sampling(self):
        m, d, s = _day(10)
        bars = _Bars(m, d, s)
        rs = resample_window(bars, 0, 9, 5)
        series = np.arange(10, dtype=np.float64)
        self.assertEqual(list(sample_last(series, rs)), [4.0, 9.0])
        self.assertEqual(list(sample_first(series, rs)), [0.0, 5.0])

    def test_window_offset_indices_are_global(self):
        m, d, s = _day(20)
        bars = _Bars(m, d, s)
        rs = resample_window(bars, 5, 14, 5)
        self.assertEqual(rs.bucket_starts[0], 5)
        self.assertEqual(rs.bucket_ends[-1], 14)


class ViewMapTests(unittest.TestCase):
    def test_identity_round_trips(self):
        vm = ViewMap.identity(100, 149)
        self.assertEqual(vm.view_start, 100)
        self.assertEqual(vm.view_end, 149)
        self.assertEqual(vm.n_display, 50)
        self.assertEqual(vm.global_to_local(123), 23)
        self.assertAlmostEqual(vm.global_to_local_f(123), 23.0)
        self.assertEqual(vm.local_to_global_start(23), 123)
        self.assertEqual(vm.local_to_global_end(23), 123)
        self.assertAlmostEqual(vm.local_f_to_global(23.0), 123.0)
        self.assertEqual(vm.last_complete_local(123), 23)

    def _vm_5m(self):
        m, d, s = _day(20)
        bars = _Bars(m, d, s)
        rs = resample_window(bars, 0, 19, 5)
        return ViewMap.from_resampled(rs, 5)

    def test_fractional_position_within_bucket(self):
        vm = self._vm_5m()
        # minute 3 of the first 5-minute bucket: l - 0.5 + (3 + 0.5)/5
        self.assertAlmostEqual(vm.global_to_local_f(3), -0.5 + 3.5 / 5)
        # round-trip
        self.assertAlmostEqual(vm.local_f_to_global(vm.global_to_local_f(13)), 13.0)

    def test_last_complete_local(self):
        vm = self._vm_5m()
        self.assertEqual(vm.last_complete_local(3), -1)  # first bucket still forming
        self.assertEqual(vm.last_complete_local(4), 0)  # exactly complete
        self.assertEqual(vm.last_complete_local(7), 0)  # mid second bucket
        self.assertEqual(vm.last_complete_local(19), 3)

    def test_minute_positions_clipped(self):
        vm = self._vm_5m()
        xs = vm.minute_positions(15, 10)  # runs past the window end
        self.assertEqual(len(xs), 10)
        self.assertLessEqual(xs.max(), float(vm.n_display))
        self.assertTrue(np.all(np.diff(xs[:5]) > 0))

    def test_freeplay_next_bucket_mapping(self):
        vm = self._vm_5m()
        # entry = clicked bucket's last minute + 1 == next bucket's first minute
        self.assertEqual(vm.local_to_global_end(0) + 1, vm.local_to_global_start(1))
        # at the final bucket it walks off the window end (caller guards)
        self.assertEqual(vm.local_to_global_end(3) + 1, vm.view_end + 1)

    def test_dst_length_day_buckets_align_by_clock(self):
        # A 23-hour clock day: minutes jump 119 -> 180 (02:00-02:59 missing).
        m = np.concatenate([np.arange(60, 120), np.arange(180, 240)])
        d = np.full(120, np.datetime64("2024-03-11"))
        bars = _Bars(m, d, np.ones(120))
        rs = resample_window(bars, 0, 119, 60)
        self.assertEqual(len(rs.open), 2)  # hour quotients 1 and 3; the missing 02:xx
        vm = ViewMap.from_resampled(rs, 60)  # hour collapses without misaligning buckets
        self.assertEqual(vm.global_to_local(60), 1)


class TimeframeTableTests(unittest.TestCase):
    def test_registry_shape(self):
        self.assertEqual(TIMEFRAMES["1m"], 1)
        self.assertEqual(TIMEFRAMES["1D"], 1440)
        self.assertEqual(list(TIMEFRAMES), ["1m", "5m", "15m", "30m", "1H", "4H", "1D"])


if __name__ == "__main__":
    unittest.main()
