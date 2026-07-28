"""Partition governance for the app: Development and Validation only.

The Final-test partition (2025-01-01 onward) must never be explored descriptively
per the project research governance, so the app never loads or renders it. The
date cap here is the hard boundary; `assert_dev_val_only` is the guard that fails
loudly if a Final-test row ever reaches the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Development is through 2023-12-31, Validation is the 2024 calendar year, and the
# Final-test partition begins 2025-01-01 (see baselines.py partition boundaries).
EVAL_CAP_DATE = pd.Timestamp("2024-12-31")
FINAL_TEST_START = pd.Timestamp("2025-01-01")
BARS_DATE_COLUMN = "trade_date_ny"


def assert_dev_val_only(trade_dates: np.ndarray | pd.Series) -> None:
    """Raise if any trade date falls in the locked Final-test partition."""

    dates = pd.to_datetime(pd.Series(trade_dates)).dt.normalize()
    if (dates >= FINAL_TEST_START).any():
        leaked = dates[dates >= FINAL_TEST_START].min()
        raise ValueError(
            "Final-test partition is locked and must not be loaded or rendered; "
            f"found a row dated {leaked.date()} (>= {FINAL_TEST_START.date()})."
        )
