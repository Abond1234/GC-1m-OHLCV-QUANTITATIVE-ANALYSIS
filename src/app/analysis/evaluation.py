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

from .account import EquityCurve, daily_pnl

RULE_DAILY_LOSS = "daily_loss"
RULE_MAX_DRAWDOWN = "max_drawdown"
RULE_PROFIT_TARGET = "profit_target"
RULE_MIN_DAYS = "min_days"
RULE_CONSISTENCY = "consistency"
RULE_MAX_CONTRACTS = "max_contracts"
RULE_MAX_DAYS = "max_days"


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


# --- Full prop-firm policy engine ------------------------------------------


@dataclass(frozen=True)
class PropFirmPolicy:
    """A configurable evaluation account. Money amounts are % of the balance."""

    name: str = "Prop 100k"
    starting_balance: float = 100_000.0
    profit_target_pct: float = 8.0
    daily_loss_limit_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    trailing_drawdown: bool = False  # True = peak-relative; False = static floor
    min_trading_days: int = 5
    max_evaluation_days: int = 0  # 0 = no cap
    consistency_max_daily_share: float = 0.0  # 0 = off; else max share of profit in one day
    max_contracts: int = 0  # 0 = no cap
    payout_split_pct: float = 80.0  # the trader's share of profit at a payout


POLICY_PRESETS: dict[str, PropFirmPolicy] = {
    "Prop 100k": PropFirmPolicy(),
    "Prop 100k trailing": PropFirmPolicy(
        name="Prop 100k trailing",
        trailing_drawdown=True,
        max_drawdown_pct=4.0,
        consistency_max_daily_share=0.4,
        max_evaluation_days=30,
    ),
    "Prop strict": PropFirmPolicy(
        name="Prop strict",
        daily_loss_limit_pct=4.0,
        max_drawdown_pct=8.0,
        profit_target_pct=10.0,
        min_trading_days=10,
        consistency_max_daily_share=0.3,
    ),
}


@dataclass
class EvaluationReport:
    """Rich verdict: state, first breach, progress, payout, and per-trade curves."""

    state: str  # "IN_PROGRESS" | "PASSED" | "FAILED"
    failed_rule: str | None
    breach_detail: str
    breach_date: str  # NY date of the first breach, or ""
    trading_days: int
    trades_taken: int
    days_to_pass: int  # -1 until passed
    trades_to_pass: int  # -1 until passed
    payout_eligible: bool
    simulated_payout: float
    post_payout_balance: float
    rules: tuple[RuleState, ...]
    equity: np.ndarray  # running account equity
    drawdown: np.ndarray  # equity minus running peak (<= 0)
    utilization: np.ndarray  # worst drawdown utilisation per trade (fraction of the cap)


def _date_str(day) -> str:
    try:
        return str(np.datetime_as_string(np.datetime64(day), unit="D"))
    except (ValueError, TypeError):
        return str(day)


def evaluate_policy(curve: EquityCurve, policy: PropFirmPolicy) -> EvaluationReport:
    """Judge the session against a full prop-firm policy; first breach decides."""

    sb = policy.starting_balance
    daily_limit = sb * policy.daily_loss_limit_pct / 100.0
    target_equity = sb * (1.0 + policy.profit_target_pct / 100.0)
    dd_amount = sb * policy.max_drawdown_pct / 100.0
    n = curve.n_trades

    if n:
        peak = np.maximum.accumulate(np.concatenate([[sb], curve.equity]))[1:]
        floor = peak - dd_amount if policy.trailing_drawdown else np.full(n, sb - dd_amount)
        drawdown = curve.equity - peak
        utilization = (
            np.clip((peak - curve.equity) / dd_amount, 0.0, None) if dd_amount else (np.zeros(n))
        )
    else:
        peak = np.array([sb])
        floor = np.array([sb - dd_amount])
        drawdown = np.array([0.0])
        utilization = np.array([0.0])

    breach: str | None = None
    breach_detail = ""
    breach_date = ""
    worst_daily = 0.0
    day_net = 0.0
    current_day = None
    days_seen: list = []
    pass_idx = -1
    for i in range(n):
        d = curve.trade_dates[i]
        if d != current_day:
            current_day = d
            day_net = 0.0
            days_seen.append(d)
        day_net += float(curve.pnl[i])
        worst_daily = min(worst_daily, day_net)
        if (
            breach is None
            and policy.max_contracts
            and int(curve.contracts[i]) > policy.max_contracts
        ):
            breach, breach_date = RULE_MAX_CONTRACTS, _date_str(d)
            breach_detail = (
                f"{int(curve.contracts[i])} contracts exceeds the {policy.max_contracts} cap"
            )
        if breach is None and day_net <= -daily_limit:
            breach, breach_date = RULE_DAILY_LOSS, _date_str(d)
            breach_detail = f"lost {-day_net:,.0f} on {breach_date}, limit {daily_limit:,.0f}"
        if breach is None and float(curve.equity[i]) <= floor[i]:
            breach, breach_date = RULE_MAX_DRAWDOWN, _date_str(d)
            breach_detail = (
                f"equity {float(curve.equity[i]):,.0f} hit the floor {float(floor[i]):,.0f}"
            )
        if (
            breach is None
            and policy.max_evaluation_days
            and len(days_seen) > policy.max_evaluation_days
        ):
            breach, breach_date = RULE_MAX_DAYS, _date_str(d)
            breach_detail = f"passed {policy.max_evaluation_days} evaluation days without passing"
        if (
            pass_idx < 0
            and float(curve.equity[i]) >= target_equity
            and len(np.unique(days_seen)) >= policy.min_trading_days
        ):
            pass_idx = i
        if breach is not None:
            break

    trading_days = int(len(np.unique(curve.trade_dates))) if n else 0
    end_equity = float(curve.equity[-1]) if n else sb
    profit = end_equity - sb

    consistency_ok = True
    consistency_detail = "not evaluated"
    biggest_share = 0.0
    if policy.consistency_max_daily_share and n:
        dates, day_pnls = daily_pnl(curve)
        gross = float(day_pnls[day_pnls > 0].sum())
        if gross > 0:
            biggest_share = float(day_pnls.max()) / gross
            consistency_ok = biggest_share <= policy.consistency_max_daily_share
        consistency_detail = (
            f"biggest day {biggest_share * 100:.0f}% of profit "
            f"(max {policy.consistency_max_daily_share * 100:.0f}%)"
        )

    if breach is not None:
        state = "FAILED"
    elif pass_idx >= 0 and consistency_ok:
        state = "PASSED"
    else:
        state = "IN_PROGRESS"

    days_to_pass = trades_to_pass = -1
    if state == "PASSED":
        trades_to_pass = pass_idx + 1
        days_to_pass = int(len(np.unique(curve.trade_dates[: pass_idx + 1])))

    payout_eligible = state == "PASSED"
    simulated_payout = (
        max(0.0, profit) * policy.payout_split_pct / 100.0 if payout_eligible else 0.0
    )
    post_payout_balance = end_equity - simulated_payout

    lowest = float(curve.equity.min()) if n else sb
    rules = (
        RuleState(
            RULE_DAILY_LOSS,
            "Daily loss",
            -worst_daily,
            daily_limit,
            breach == RULE_DAILY_LOSS,
            f"worst day -{-worst_daily:,.0f} of {daily_limit:,.0f} allowed",
        ),
        RuleState(
            RULE_MAX_DRAWDOWN,
            "Trailing drawdown" if policy.trailing_drawdown else "Max drawdown",
            max(0.0, float(peak.max()) - lowest) if n else 0.0,
            dd_amount,
            breach == RULE_MAX_DRAWDOWN,
            f"lowest equity {lowest:,.0f}; cap {dd_amount:,.0f}",
        ),
        RuleState(
            RULE_PROFIT_TARGET,
            "Profit target",
            max(0.0, profit),
            target_equity - sb,
            False,
            f"equity {end_equity:,.0f} of {target_equity:,.0f} target",
        ),
        RuleState(
            RULE_MIN_DAYS,
            "Trading days",
            float(trading_days),
            float(policy.min_trading_days),
            False,
            f"{trading_days} of {policy.min_trading_days} required",
        ),
        RuleState(
            RULE_CONSISTENCY,
            "Consistency",
            biggest_share,
            policy.consistency_max_daily_share,
            not consistency_ok,
            consistency_detail,
        ),
    )

    return EvaluationReport(
        state=state,
        failed_rule=breach,
        breach_detail=breach_detail,
        breach_date=breach_date,
        trading_days=trading_days,
        trades_taken=n,
        days_to_pass=days_to_pass,
        trades_to_pass=trades_to_pass,
        payout_eligible=payout_eligible,
        simulated_payout=simulated_payout,
        post_payout_balance=post_payout_balance,
        rules=rules,
        equity=curve.equity if n else np.array([sb]),
        drawdown=drawdown,
        utilization=utilization,
    )
