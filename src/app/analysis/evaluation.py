"""Prop-firm style evaluation rules over the simulated session.

Hard, deterministic checks over realized simulated trades - the discipline
layer of evaluation practice: a daily-loss limit, a static maximum-drawdown
floor, a profit target, and a minimum number of distinct trading days. Nothing
here predicts anything; it only judges what the blotter already did, so it is
entirely outside the research governance's frozen-verdict scope.

Defaults follow the common evaluation shape (5% daily loss, 10% total drawdown
from the starting balance, 8% profit target, 5 trading days), all configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .account import EquityCurve

RULE_DAILY_LOSS = "daily_loss"
RULE_MAX_DRAWDOWN = "max_drawdown"
RULE_PROFIT_TARGET = "profit_target"
RULE_MIN_DAYS = "min_days"


@dataclass(frozen=True)
class EvaluationRules:
    """One evaluation preset. Percentages are of the starting balance."""

    name: str = "Prop 100k"
    daily_loss_limit_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    profit_target_pct: float = 8.0
    min_trading_days: int = 5


PRACTICE = None  # sentinel meaning "no rules armed"

PRESETS: dict[str, EvaluationRules | None] = {
    "Practice - no rules": PRACTICE,
    "Prop 100k": EvaluationRules(),
    "Prop strict": EvaluationRules(
        name="Prop strict",
        daily_loss_limit_pct=4.0,
        max_drawdown_pct=8.0,
        profit_target_pct=10.0,
        min_trading_days=10,
    ),
}


@dataclass(frozen=True)
class RuleState:
    """One rule's live meter: how much is used against its limit."""

    rule: str
    label: str
    used: float  # dollars used (or days for min_days)
    limit: float  # dollars allowed (or days required)
    breached: bool
    detail: str


@dataclass(frozen=True)
class EvaluationStatus:
    """Verdict over the session: IN_PROGRESS until a breach or a completed pass."""

    state: str  # "IN_PROGRESS" | "PASSED" | "FAILED"
    failed_rule: str | None
    rules: tuple[RuleState, ...] = field(default_factory=tuple)


def evaluate(
    curve: EquityCurve, rules: EvaluationRules, starting_balance: float
) -> EvaluationStatus:
    """Judge the session chronologically; the first breach decides FAILED.

    The daily-loss limit applies to each NY trade date's cumulative net loss;
    the drawdown floor is static below the starting balance (evaluation style);
    PASSED requires reaching the profit target with at least the minimum number
    of distinct trading days and no breach anywhere in the sequence.
    """

    daily_limit = starting_balance * rules.daily_loss_limit_pct / 100.0
    dd_floor = starting_balance * (1.0 - rules.max_drawdown_pct / 100.0)
    target_equity = starting_balance * (1.0 + rules.profit_target_pct / 100.0)

    worst_daily_loss = 0.0
    breach: str | None = None
    if curve.n_trades:
        day_net = 0.0
        current_day = curve.trade_dates[0]
        for i in range(curve.n_trades):
            if curve.trade_dates[i] != current_day:
                current_day = curve.trade_dates[i]
                day_net = 0.0
            day_net += float(curve.pnl[i])
            worst_daily_loss = min(worst_daily_loss, day_net)
            if breach is None and day_net <= -daily_limit:
                breach = RULE_DAILY_LOSS
            if breach is None and float(curve.equity[i]) <= dd_floor:
                breach = RULE_MAX_DRAWDOWN

    end_equity = float(curve.equity[-1]) if curve.n_trades else starting_balance
    lowest_equity = float(curve.equity.min()) if curve.n_trades else starting_balance
    days = len(np.unique(curve.trade_dates)) if curve.n_trades else 0

    states = (
        RuleState(
            RULE_DAILY_LOSS,
            "Daily loss",
            used=-worst_daily_loss,
            limit=daily_limit,
            breached=breach == RULE_DAILY_LOSS,
            detail=f"worst day -{-worst_daily_loss:,.0f} of {daily_limit:,.0f} allowed",
        ),
        RuleState(
            RULE_MAX_DRAWDOWN,
            "Max drawdown",
            used=max(0.0, starting_balance - lowest_equity),
            limit=starting_balance - dd_floor,
            breached=breach == RULE_MAX_DRAWDOWN,
            detail=f"lowest equity {lowest_equity:,.0f}; floor {dd_floor:,.0f}",
        ),
        RuleState(
            RULE_PROFIT_TARGET,
            "Profit target",
            used=max(0.0, end_equity - starting_balance),
            limit=target_equity - starting_balance,
            breached=False,
            detail=f"equity {end_equity:,.0f} of {target_equity:,.0f} target",
        ),
        RuleState(
            RULE_MIN_DAYS,
            "Trading days",
            used=float(days),
            limit=float(rules.min_trading_days),
            breached=False,
            detail=f"{days} of {rules.min_trading_days} required days",
        ),
    )

    if breach is not None:
        return EvaluationStatus("FAILED", breach, states)
    if end_equity >= target_equity and days >= rules.min_trading_days:
        return EvaluationStatus("PASSED", None, states)
    return EvaluationStatus("IN_PROGRESS", None, states)
