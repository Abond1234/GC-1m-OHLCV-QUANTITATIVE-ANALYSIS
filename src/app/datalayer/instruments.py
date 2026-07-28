"""Instrument registry and multi-asset bar loading for the simulator.

One place declares every instrument the app knows how to chart: display name,
CME/exchange contract economics (dollars per point, tick size), and where its
1-minute bars come from. Two data sources exist:

* ``research`` - a product inside the repository's research bar table
  (``research_bars_gc_mgc_1m.parquet``). These load through ``BarStore.load``
  and therefore inherit the Development+Validation cap; the Final-test
  partition is never read.
* ``external`` - a per-instrument parquet dropped under
  ``data/processed/instruments/<SYMBOL>_1m.parquet`` by the user. These are
  not research data (no partitions apply), but the schema contract is strict
  and enforced loudly: the file must carry exactly the bar-store columns the
  simulator's engine consumes. See the app README's "Adding an instrument".

Only instruments whose data is actually present are offered in the UI;
registered-but-dataless instruments (NQ, ES, BTCUSD until you supply bars)
surface as "no local data" so the pathway is visible without pretending.

Research validity note: the strategy catalog, replay markers, and the Edge
Context panel are GC-validated evidence. The app disables them - with the
reason on screen - for every other instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..analysis.account import InstrumentSpec
from .bar_store import BarStore
from .paths import processed_dir, research_bars_path

EXTERNAL_DIR_NAME = "instruments"

# Columns an external instrument file must provide (the resident bar-store
# contract minus ``product``, which is implied by the file).
EXTERNAL_REQUIRED_COLUMNS = (
    "ts_event_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_date_ny",
    "minute_of_day_ny",
    "continuous_segment_id",
    "rolling_atr_20m",
)


@dataclass(frozen=True)
class Instrument:
    """One chartable asset: economics plus its data source."""

    symbol: str
    display: str
    spec: InstrumentSpec  # dollars per point + tick size (account model reuse)
    source: str  # "research" | "external"
    research_grade: bool  # True only for the GC signal instrument
    notes: str


REGISTRY: dict[str, Instrument] = {
    "GC": Instrument(
        "GC",
        "GC - Gold futures",
        InstrumentSpec("GC", 100.0, 0.10),
        "research",
        True,
        "The research signal instrument. Strategy catalog, replay, and Edge "
        "Context evidence are validated on GC only.",
    ),
    "MGC": Instrument(
        "MGC",
        "MGC - Micro gold",
        InstrumentSpec("MGC", 10.0, 0.10),
        "research",
        False,
        "Micro gold from the research table (Dev+Val capped). Charting and "
        "free-play practice only: no research trial is run, per the frozen "
        "Section 6 verdict that MGC transfer work is not permitted.",
    ),
    "NQ": Instrument(
        "NQ",
        "NQ - Nasdaq-100 futures",
        InstrumentSpec("NQ", 20.0, 0.25),
        "external",
        False,
        "Requires data/processed/instruments/NQ_1m.parquet.",
    ),
    "ES": Instrument(
        "ES",
        "ES - S&P 500 futures",
        InstrumentSpec("ES", 50.0, 0.25),
        "external",
        False,
        "Requires data/processed/instruments/ES_1m.parquet.",
    ),
    "BTCUSD": Instrument(
        "BTCUSD",
        "BTCUSD - Bitcoin",
        InstrumentSpec("BTCUSD", 1.0, 1.0),
        "external",
        False,
        "Spot/perp bitcoin priced at 1 dollar per point per unit. Requires "
        "data/processed/instruments/BTCUSD_1m.parquet.",
    ),
}


def external_dir(root: Path | None = None) -> Path:
    base = processed_dir() if root is None else Path(root) / "data" / "processed"
    return base / EXTERNAL_DIR_NAME


def external_path(symbol: str, root: Path | None = None) -> Path:
    return external_dir(root) / f"{symbol}_1m.parquet"


def data_available(symbol: str, root: Path | None = None) -> bool:
    """Whether the instrument's bars exist locally (cheap filesystem checks)."""

    instrument = REGISTRY.get(symbol)
    if instrument is None:
        return False
    if instrument.source == "research":
        return research_bars_path().exists()
    return external_path(symbol, root).exists()


def available_instruments(root: Path | None = None) -> list[Instrument]:
    """Registry order, filtered to instruments with local data."""

    return [inst for inst in REGISTRY.values() if data_available(inst.symbol, root)]


def load_instrument_bars(
    symbol: str,
    *,
    date_floor: pd.Timestamp | None = None,
) -> BarStore:
    """Load an instrument's 1m bars into the simulator's bar store.

    Research products go through ``BarStore.load`` (Dev+Val capped). External
    files are validated against the strict column contract and refused loudly
    on any mismatch - a wrong file must never render as a slightly different
    chart.
    """

    instrument = REGISTRY.get(symbol)
    if instrument is None:
        raise KeyError(f"unknown instrument {symbol!r}")
    if instrument.source == "research":
        return BarStore.load(research_bars_path(), product=symbol, date_floor=date_floor)

    path = external_path(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"no local data for {symbol}: expected {path}. See the app README's "
            f"'Adding an instrument' for the column contract."
        )
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(path).names)
    missing = [name for name in EXTERNAL_REQUIRED_COLUMNS if name not in schema_names]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns {missing}; the external "
            f"instrument contract is {list(EXTERNAL_REQUIRED_COLUMNS)}"
        )
    frame = pq.read_table(path, columns=list(EXTERNAL_REQUIRED_COLUMNS)).to_pandas(
        ignore_metadata=True
    )
    frame = frame.assign(product=symbol)
    if date_floor is not None:
        frame = frame[frame["trade_date_ny"] >= pd.Timestamp(date_floor)]
    frame = frame.sort_values("ts_event_utc").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows")
    return BarStore(
        ts=frame["ts_event_utc"].to_numpy(),
        open=frame["open"].to_numpy(dtype="float64"),
        high=frame["high"].to_numpy(dtype="float64"),
        low=frame["low"].to_numpy(dtype="float64"),
        close=frame["close"].to_numpy(dtype="float64"),
        volume=frame["volume"].to_numpy(dtype="float64"),
        atr20=frame["rolling_atr_20m"].to_numpy(dtype="float64"),
        segment=frame["continuous_segment_id"].to_numpy(),
        trade_date=frame["trade_date_ny"].to_numpy(),
        minute_ny=frame["minute_of_day_ny"].to_numpy(dtype="int64"),
    )
