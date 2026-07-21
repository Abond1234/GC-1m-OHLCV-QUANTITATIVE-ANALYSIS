"""Versioned prop-firm rules engine (PRD FR-10).

The PRD treats the prop firm as a versioned policy layer, never a fixed
assumption inside strategy code: profit target, daily loss, static or trailing
drawdown, consistency, minimum days, maximum contracts, and effective dates
are all configurable policy fields.  This module provides that layer plus a
deterministic evaluation simulator and a bootstrap estimator of evaluation
economics (pass probability, expected days to pass, breach probabilities).

Honest boundaries, declared up front:

- The simulator consumes a chronological per-trade P&L stream.  Daily-loss and
  trailing-drawdown checks run after every trade, so breaches driven purely by
  unrealized intra-trade excursions are not modeled unless the caller supplies
  excursion rows.  This makes the simulator mildly optimistic; internal buffer
  fractions (stricter-than-firm limits, per the PRD risk-buffer requirement)
  exist to absorb exactly this class of gap.
- The bundled example policies are illustrative templates only.  They are not
  the terms of any real firm; the PRD requires the selected firm's current
  published terms to be re-verified before implementation and deployment.
- No strategy claim is made here.  The engine is Phase 4 infrastructure; the
  research program has not approved any strategy to feed it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

DRAWDOWN_MODES = ("static", "trailing_intraday", "trailing_eod")

OUTCOME_PASSED = "PASSED"
OUTCOME_FAILED_DAILY_LOSS = "FAILED_DAILY_LOSS"
OUTCOME_FAILED_MAX_DRAWDOWN = "FAILED_MAX_DRAWDOWN"
OUTCOME_PASS_BLOCKED_CONSISTENCY = "PASS_BLOCKED_CONSISTENCY"
OUTCOME_INCOMPLETE_TARGET = "INCOMPLETE_TARGET_NOT_REACHED"
OUTCOME_INCOMPLETE_MAX_DAYS = "INCOMPLETE_MAX_DAYS_EXCEEDED"


@dataclass(frozen=True)
class PropFirmPolicy:
    """One versioned evaluation-account rule set with an effective date."""

    policy_id: str
    version: str
    effective_date: str
    account_size_usd: float
    profit_target_usd: float
    daily_loss_limit_usd: float
    max_drawdown_usd: float
    drawdown_mode: str
    trailing_caps_at_initial_balance: bool
    max_contracts: int
    minimum_trading_days: int
    maximum_evaluation_days: int | None = None
    consistency_max_daily_share: float | None = None
    internal_daily_loss_buffer_fraction: float = 0.8
    internal_drawdown_buffer_fraction: float = 0.9
    notes: str = ""

    def validate(self) -> None:
        if self.drawdown_mode not in DRAWDOWN_MODES:
            raise ValueError(f"drawdown_mode must be one of {DRAWDOWN_MODES}")
        for name in (
            "account_size_usd",
            "profit_target_usd",
            "daily_loss_limit_usd",
            "max_drawdown_usd",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_contracts < 1:
            raise ValueError("max_contracts must be at least 1")
        if self.minimum_trading_days < 0:
            raise ValueError("minimum_trading_days must be non-negative")
        if self.maximum_evaluation_days is not None and (
            self.maximum_evaluation_days < max(1, self.minimum_trading_days)
        ):
            raise ValueError("maximum_evaluation_days must allow the minimum trading days")
        if self.consistency_max_daily_share is not None and not (
            0.0 < self.consistency_max_daily_share <= 1.0
        ):
            raise ValueError("consistency_max_daily_share must be in (0, 1]")
        for name in (
            "internal_daily_loss_buffer_fraction",
            "internal_drawdown_buffer_fraction",
        ):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> PropFirmPolicy:
        policy = cls(**payload)
        policy.validate()
        return policy


def example_policies() -> tuple[PropFirmPolicy, ...]:
    """Illustrative templates only - not any real firm's terms."""

    static = PropFirmPolicy(
        policy_id="example_static_50k",
        version="1.0",
        effective_date="2026-07-20",
        account_size_usd=50_000.0,
        profit_target_usd=3_000.0,
        daily_loss_limit_usd=1_000.0,
        max_drawdown_usd=2_000.0,
        drawdown_mode="static",
        trailing_caps_at_initial_balance=False,
        max_contracts=5,
        minimum_trading_days=5,
        maximum_evaluation_days=None,
        consistency_max_daily_share=0.5,
        notes="Illustrative static-drawdown template; verify real terms before use.",
    )
    trailing = PropFirmPolicy(
        policy_id="example_trailing_100k",
        version="1.0",
        effective_date="2026-07-20",
        account_size_usd=100_000.0,
        profit_target_usd=6_000.0,
        daily_loss_limit_usd=2_000.0,
        max_drawdown_usd=3_500.0,
        drawdown_mode="trailing_eod",
        trailing_caps_at_initial_balance=True,
        max_contracts=10,
        minimum_trading_days=7,
        maximum_evaluation_days=60,
        consistency_max_daily_share=None,
        notes="Illustrative trailing-drawdown template; verify real terms before use.",
    )
    for policy in (static, trailing):
        policy.validate()
    return static, trailing


@dataclass
class EvaluationResult:
    """Outcome plus full daily ledger and breach detail for one simulation."""

    outcome: str
    trading_days: int
    final_equity_usd: float
    breach_detail: str
    daily_ledger: pd.DataFrame
    internal_daily_buffer_hits: int
    internal_drawdown_buffer_hits: int
    policy: PropFirmPolicy | None = field(repr=False, default=None)


def _drawdown_floor(policy: PropFirmPolicy, watermark: float) -> float:
    if policy.drawdown_mode == "static":
        return policy.account_size_usd - policy.max_drawdown_usd
    trail_from = watermark
    if policy.trailing_caps_at_initial_balance:
        trail_from = min(watermark, policy.account_size_usd + policy.max_drawdown_usd)
    return trail_from - policy.max_drawdown_usd


def simulate_evaluation(trades: pd.DataFrame, policy: PropFirmPolicy) -> EvaluationResult:
    """Walk a chronological per-trade P&L stream through one policy.

    ``trades`` needs ``trade_date`` (sortable day key) and ``pnl_usd``.  Rows
    must be in execution order within each day; days are processed in sorted
    order.  Checks run after every trade: daily loss against the firm limit,
    equity against the drawdown floor, and the stricter internal buffers
    (counted, never a firm breach).  The profit target passes only once the
    minimum trading days are also met; the consistency rule, when configured,
    blocks the pass rather than failing the account.
    """

    policy.validate()
    if not {"trade_date", "pnl_usd"}.issubset(trades.columns):
        raise ValueError("trades needs trade_date and pnl_usd columns")

    equity = policy.account_size_usd
    watermark = equity
    ledger_rows: list[dict] = []
    internal_daily_hits = 0
    internal_drawdown_hits = 0
    outcome: str | None = None
    breach_detail = ""
    daily_profits: list[float] = []
    target_reached = False

    for day_index, (day, day_trades) in enumerate(trades.groupby("trade_date", sort=True), start=1):
        day_start_equity = equity
        day_pnl = 0.0
        day_breach = ""
        day_hit_daily_buffer = False
        day_hit_drawdown_buffer = False
        for pnl in day_trades["pnl_usd"].to_numpy(dtype=np.float64):
            equity += pnl
            day_pnl += pnl
            if policy.drawdown_mode == "trailing_intraday":
                watermark = max(watermark, equity)
            floor = _drawdown_floor(policy, watermark)
            if day_pnl <= -policy.daily_loss_limit_usd * policy.internal_daily_loss_buffer_fraction:
                day_hit_daily_buffer = True
            if equity <= floor + policy.max_drawdown_usd * (
                1.0 - policy.internal_drawdown_buffer_fraction
            ):
                day_hit_drawdown_buffer = True
            if day_pnl <= -policy.daily_loss_limit_usd and not day_breach:
                day_breach = OUTCOME_FAILED_DAILY_LOSS
            if equity <= floor and not day_breach:
                day_breach = OUTCOME_FAILED_MAX_DRAWDOWN
            if day_breach:
                break
        internal_daily_hits += int(day_hit_daily_buffer)
        internal_drawdown_hits += int(day_hit_drawdown_buffer)
        if policy.drawdown_mode == "trailing_eod":
            watermark = max(watermark, equity)
        daily_profits.append(day_pnl)
        ledger_rows.append(
            {
                "trade_date": day,
                "day_index": day_index,
                "day_start_equity_usd": day_start_equity,
                "day_pnl_usd": day_pnl,
                "equity_usd": equity,
                "watermark_usd": watermark,
                "drawdown_floor_usd": _drawdown_floor(policy, watermark),
                "breach": day_breach,
            }
        )
        if day_breach:
            outcome = day_breach
            breach_detail = f"{day_breach} on {day} (day {day_index})"
            break
        cumulative_profit = equity - policy.account_size_usd
        if (
            cumulative_profit >= policy.profit_target_usd
            and day_index >= policy.minimum_trading_days
        ):
            target_reached = True
            if policy.consistency_max_daily_share is not None:
                gross_positive = sum(p for p in daily_profits if p > 0)
                largest = max((p for p in daily_profits if p > 0), default=0.0)
                if gross_positive > 0 and largest / gross_positive > (
                    policy.consistency_max_daily_share + 1e-12
                ):
                    outcome = OUTCOME_PASS_BLOCKED_CONSISTENCY
                    breach_detail = (
                        f"largest day {largest / gross_positive:.1%} of gross profit exceeds "
                        f"{policy.consistency_max_daily_share:.0%} cap"
                    )
                    break
            outcome = OUTCOME_PASSED
            break
        if (
            policy.maximum_evaluation_days is not None
            and day_index >= policy.maximum_evaluation_days
        ):
            outcome = OUTCOME_INCOMPLETE_MAX_DAYS
            break

    if outcome is None:
        outcome = OUTCOME_INCOMPLETE_TARGET if not target_reached else OUTCOME_PASSED
    ledger = pd.DataFrame.from_records(ledger_rows)
    return EvaluationResult(
        outcome=outcome,
        trading_days=len(ledger),
        final_equity_usd=float(equity),
        breach_detail=breach_detail,
        daily_ledger=ledger,
        internal_daily_buffer_hits=internal_daily_hits,
        internal_drawdown_buffer_hits=internal_drawdown_hits,
        policy=policy,
    )


def estimate_evaluation_statistics(
    daily_pnl_pool: Iterable[float],
    policy: PropFirmPolicy,
    *,
    replicates: int = 2_000,
    horizon_days: int = 120,
    seed: int = 20260729,
) -> pd.Series:
    """Bootstrap evaluation economics from a pool of daily P&L outcomes.

    Each replicate draws ``horizon_days`` daily P&L values with replacement,
    simulates the evaluation, and records the outcome.  Deterministic under
    ``seed``.  This estimates the PRD's evaluation-economics metrics for
    whatever daily distribution the caller supplies; it makes no claim about
    any particular strategy.
    """

    policy.validate()
    pool = np.asarray(list(daily_pnl_pool), dtype=np.float64)
    if len(pool) < 2:
        raise ValueError("daily_pnl_pool needs at least two observations")
    rng = np.random.default_rng(seed)
    outcomes: list[str] = []
    days_to_pass: list[int] = []
    for _ in range(replicates):
        draws = pool[rng.integers(0, len(pool), size=horizon_days)]
        trades = pd.DataFrame({"trade_date": np.arange(horizon_days), "pnl_usd": draws})
        result = simulate_evaluation(trades, policy)
        outcomes.append(result.outcome)
        if result.outcome == OUTCOME_PASSED:
            days_to_pass.append(result.trading_days)
    counts = pd.Series(outcomes).value_counts()
    return pd.Series(
        {
            "replicates": replicates,
            "horizon_days": horizon_days,
            "pass_probability": counts.get(OUTCOME_PASSED, 0) / replicates,
            "daily_loss_breach_probability": counts.get(OUTCOME_FAILED_DAILY_LOSS, 0) / replicates,
            "max_drawdown_breach_probability": counts.get(OUTCOME_FAILED_MAX_DRAWDOWN, 0)
            / replicates,
            "consistency_block_probability": counts.get(OUTCOME_PASS_BLOCKED_CONSISTENCY, 0)
            / replicates,
            "incomplete_probability": (
                counts.get(OUTCOME_INCOMPLETE_TARGET, 0)
                + counts.get(OUTCOME_INCOMPLETE_MAX_DAYS, 0)
            )
            / replicates,
            "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else np.nan,
        },
        name="value",
    )
