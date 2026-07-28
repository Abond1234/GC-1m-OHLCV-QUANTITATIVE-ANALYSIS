"""Dollar account model over simulated free-play trades.

Converts the blotter's R-multiple outcomes into account currency using CME
contract economics (GC $100/point, MGC $10/point, 0.10 tick), builds a
compounding equity curve in entry order, and summarises the session the way a
trading-desk review would: win rate, expectancy, profit factor, drawdown.

Everything here is Qt-free and unit-tested. The numbers describe simulated
fills on historical Development+Validation data - they are practice accounting,
not live results and not a validated strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract economics for one listed instrument."""

    name: str
    dollars_per_point: float
    tick_size: float


GC_SPEC = InstrumentSpec("GC", 100.0, 0.10)
MGC_SPEC = InstrumentSpec("MGC", 10.0, 0.10)

SIZING_MODES = ("risk_percent", "fixed_contracts")


@dataclass(frozen=True)
class AccountSettings:
    """User-configurable account: balance, sizing policy, instrument."""

    starting_balance: float = 100_000.0
    sizing_mode: str = "risk_percent"
    risk_percent: float = 1.0  # of current equity, when sizing_mode == "risk_percent"
    fixed_contracts: int = 1
    instrument: InstrumentSpec = GC_SPEC


def contracts_for(settings: AccountSettings, stop_points: float, equity: float) -> int:
    """Position size for one trade under the account's sizing policy (min 1).

    Risk-percent mode: the largest whole number of contracts whose loss at the
    initial stop stays within ``risk_percent`` of current equity.
    """

    if settings.sizing_mode == "fixed_contracts":
        return max(1, int(settings.fixed_contracts))
    risk_dollars = max(0.0, float(equity)) * settings.risk_percent / 100.0
    per_contract = float(stop_points) * settings.instrument.dollars_per_point
    if not math.isfinite(per_contract) or per_contract <= 0:
        return 1
    return max(1, int(risk_dollars / per_contract))


def trade_dollars(
    settings: AccountSettings, gross_r: float, stop_points: float, contracts: int
) -> float:
    """Realized P&L in dollars: R multiple x initial risk per contract x size."""

    return (
        float(gross_r) * float(stop_points) * settings.instrument.dollars_per_point * int(contracts)
    )


@dataclass(frozen=True)
class EquityCurve:
    """Per-trade dollar accounting in entry order, plus the running curve."""

    entry_positions: np.ndarray  # global 1m bar index per trade (sorted)
    trade_ids: np.ndarray
    trade_dates: np.ndarray  # NY trade date per trade
    contracts: np.ndarray
    pnl: np.ndarray  # per-trade $
    equity: np.ndarray  # after each trade, from starting_balance
    drawdown: np.ndarray  # equity minus running peak (<= 0)

    @property
    def n_trades(self) -> int:
        return len(self.pnl)


def build_equity_curve(trades, settings: AccountSettings, trade_dates) -> EquityCurve:
    """Walk placed trades in entry order, sizing each from equity at entry.

    ``trades`` is any iterable of PlacedTrade-shaped objects (``entry_position``,
    ``id``, ``stop_points``, ``result.gross_r``); ``trade_dates`` maps a global
    bar index to its NY trade date (the BarStore ``trade_date`` array).
    """

    ordered = sorted(trades, key=lambda t: int(t.entry_position))
    n = len(ordered)
    entry_positions = np.empty(n, dtype=np.int64)
    trade_ids = np.empty(n, dtype=np.int64)
    dates = np.empty(n, dtype="datetime64[ns]")
    contracts = np.empty(n, dtype=np.int64)
    pnl = np.empty(n, dtype=np.float64)
    equity = np.empty(n, dtype=np.float64)
    running = settings.starting_balance
    for i, trade in enumerate(ordered):
        entry = int(trade.entry_position)
        size = contracts_for(settings, float(trade.stop_points), running)
        dollars = trade_dollars(
            settings, float(trade.result.gross_r), float(trade.stop_points), size
        )
        running += dollars
        entry_positions[i] = entry
        trade_ids[i] = int(trade.id)
        dates[i] = np.datetime64(trade_dates[entry], "ns")
        contracts[i] = size
        pnl[i] = dollars
        equity[i] = running
    peak = (
        np.maximum.accumulate(np.concatenate(([settings.starting_balance], equity)))[1:]
        if n
        else equity
    )
    drawdown = equity - peak if n else equity
    return EquityCurve(
        entry_positions=entry_positions,
        trade_ids=trade_ids,
        trade_dates=dates,
        contracts=contracts,
        pnl=pnl,
        equity=equity,
        drawdown=drawdown,
    )


def daily_pnl(curve: EquityCurve) -> tuple[np.ndarray, np.ndarray]:
    """(unique NY trade dates, summed $ per date), in date order."""

    if curve.n_trades == 0:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float64)
    dates, inverse = np.unique(curve.trade_dates, return_inverse=True)
    sums = np.zeros(len(dates), dtype=np.float64)
    np.add.at(sums, inverse, curve.pnl)
    return dates, sums


@dataclass(frozen=True)
class SessionStats:
    """Desk-review summary of the session's simulated trading."""

    n_trades: int
    wins: int
    win_rate: float
    avg_r: float
    expectancy_usd: float
    profit_factor: float
    total_pnl: float
    end_equity: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    days_traded: int


def session_stats(curve: EquityCurve, r_multiples, settings: AccountSettings) -> SessionStats:
    """Aggregate the curve; ``r_multiples`` aligns with the curve's trade order."""

    n = curve.n_trades
    if n == 0:
        return SessionStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, settings.starting_balance, 0.0, 0.0, 0)
    r_arr = np.asarray(list(r_multiples), dtype=np.float64)
    wins = int((curve.pnl > 0).sum())
    gains = float(curve.pnl[curve.pnl > 0].sum())
    losses = float(-curve.pnl[curve.pnl < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0
    max_dd = float(-curve.drawdown.min()) if n else 0.0
    peak = float(np.maximum.accumulate(curve.equity).max())
    dates, _sums = daily_pnl(curve)
    return SessionStats(
        n_trades=n,
        wins=wins,
        win_rate=wins / n,
        avg_r=float(r_arr.mean()),
        expectancy_usd=float(curve.pnl.mean()),
        profit_factor=profit_factor,
        total_pnl=float(curve.pnl.sum()),
        end_equity=float(curve.equity[-1]),
        max_drawdown_usd=max_dd,
        max_drawdown_pct=(max_dd / peak * 100.0) if peak > 0 else 0.0,
        days_traded=len(dates),
    )
