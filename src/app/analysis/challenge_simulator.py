"""Serial multi-account prop-firm challenge simulation over strategy trades.

The strategy replay engine supplies a chronological log of real Development +
Validation trade outcomes.  This module applies user-defined account rules to
that log without changing the underlying strategy or exit calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class PhaseRules:
    """Profit and loss limits for one challenge phase."""

    profit_target_pct: float
    daily_loss_limit_pct: float
    max_loss_limit_pct: float
    consistency_pct: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("profit_target_pct", self.profit_target_pct),
            ("daily_loss_limit_pct", self.daily_loss_limit_pct),
            ("max_loss_limit_pct", self.max_loss_limit_pct),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.consistency_pct is not None and not 0 < self.consistency_pct <= 100:
            raise ValueError("consistency_pct must be between 0% and 100%")


@dataclass(frozen=True)
class ChallengeRules:
    """Account pool and challenge progression selected in Prop firm."""

    account_count: int = 10
    starting_balance: float = 100_000.0
    phases: tuple[PhaseRules, ...] = (PhaseRules(8.0, 5.0, 10.0),)
    payout_target_pct: float = 5.0
    funded_consistency_pct: float | None = None

    def __post_init__(self) -> None:
        if self.account_count < 1:
            raise ValueError("account_count must be at least one")
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if len(self.phases) not in (1, 2):
            raise ValueError("a challenge must have one or two phases")
        if not 3.0 <= self.payout_target_pct <= 5.0:
            raise ValueError("payout_target_pct must be between 3% and 5%")
        if self.funded_consistency_pct is not None and not 0 < self.funded_consistency_pct <= 100:
            raise ValueError("funded_consistency_pct must be between 0% and 100%")


@dataclass(frozen=True)
class RiskRules:
    """Trader risk plan; daily stops rotate to another eligible account."""

    risk_per_trade_pct: float = 1.0
    daily_profit_target_pct: float = 2.0
    daily_loss_limit_pct: float = 2.0
    max_trades_per_day: int = 0

    def __post_init__(self) -> None:
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be positive")
        if self.daily_profit_target_pct < 0 or self.daily_loss_limit_pct < 0:
            raise ValueError("daily targets cannot be negative")
        if self.max_trades_per_day < 0:
            raise ValueError("max_trades_per_day cannot be negative")


@dataclass(frozen=True)
class AccountSummary:
    account_id: int
    status: str
    stage: str
    equity: float
    trades: int
    payout: float
    breach_reason: str


@dataclass(frozen=True)
class ChallengeSimulationResult:
    """Challenge outcomes plus the subset of strategy trades actually taken."""

    trades: pd.DataFrame
    accounts: tuple[AccountSummary, ...]
    accounts_requested: int
    phase1_passes: int
    phase2_passes: int
    challenges_passed: int
    payouts_received: int
    accounts_breached: int
    total_payout: float
    total_pnl: float
    skipped_trades: int
    three_breach_stop_days: int

    @property
    def trades_taken(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return int((self.trades["gross_r"] > 0).sum()) if len(self.trades) else 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades_taken if self.trades_taken else 0.0

    @property
    def mean_r(self) -> float:
        return float(self.trades["gross_r"].mean()) if self.trades_taken else 0.0


@dataclass
class _Account:
    account_id: int
    equity: float
    phase_index: int = 0
    status: str = "CHALLENGE"
    day_pnl: float = 0.0
    day_trades: int = 0
    total_trades: int = 0
    paused_day: object | None = None
    payout: float = 0.0
    breach_reason: str = ""
    stage_day_pnl: dict[object, float] = field(default_factory=dict)


_ADDED_COLUMNS = {
    "account_id": "int64",
    "account_stage": "object",
    "simulated_pnl": "float64",
    "simulated_equity": "float64",
    "account_event": "object",
    "account_status": "object",
}


def _empty_log(source: pd.DataFrame) -> pd.DataFrame:
    out = source.iloc[0:0].copy()
    for name, dtype in _ADDED_COLUMNS.items():
        out[name] = pd.Series(dtype=dtype)
    return out


def _stage_label(account: _Account, rules: ChallengeRules) -> str:
    if account.status == "FUNDED":
        return "Funded"
    if account.status == "PAYOUT":
        return "Payout"
    if account.status == "BREACHED":
        return "Breached"
    return f"Phase {account.phase_index + 1}"


def _phase_for(account: _Account, rules: ChallengeRules) -> PhaseRules:
    if account.status == "FUNDED":
        base = rules.phases[-1]
        return PhaseRules(
            rules.payout_target_pct,
            base.daily_loss_limit_pct,
            base.max_loss_limit_pct,
            rules.funded_consistency_pct,
        )
    return rules.phases[account.phase_index]


def _consistency_ratio(account: _Account, challenge: ChallengeRules) -> float:
    """Largest profitable day as a percentage of the current stage's profit."""

    stage_profit = account.equity - challenge.starting_balance
    if stage_profit <= 0:
        return float("inf")
    largest_day = max((max(0.0, pnl) for pnl in account.stage_day_pnl.values()), default=0.0)
    return largest_day / stage_profit * 100.0


def _consistency_met(
    account: _Account,
    challenge: ChallengeRules,
    phase: PhaseRules,
) -> tuple[bool, float | None]:
    if phase.consistency_pct is None:
        return True, None
    ratio = _consistency_ratio(account, challenge)
    return ratio <= phase.consistency_pct + 1e-9, ratio


def _next_account(accounts: list[_Account], cursor: int, day) -> tuple[int, _Account] | None:
    for offset in range(len(accounts)):
        index = (cursor + offset) % len(accounts)
        account = accounts[index]
        if account.status in {"CHALLENGE", "FUNDED"} and account.paused_day != day:
            return index, account
    return None


def simulate_challenges(
    strategy_log: pd.DataFrame,
    challenge: ChallengeRules,
    risk: RiskRules,
) -> ChallengeSimulationResult:
    """Apply serial account rules to a chronological strategy replay.

    Exactly one account receives each eligible setup. Daily trader stops pause
    that account and rotate to the next one. Firm daily/overall loss breaches
    permanently fail it. Three consecutive account breaches on one NY trade
    date stop all remaining trading for that date.
    """

    required = {"trade_date_ny", "gross_r", "entry_position", "exit_position"}
    if len(strategy_log) and not required.issubset(strategy_log.columns):
        missing = ", ".join(sorted(required - set(strategy_log.columns)))
        raise ValueError(f"strategy log is missing: {missing}")

    ordered = (
        strategy_log.sort_values(["entry_position", "exit_position"], kind="stable")
        .reset_index(drop=True)
        .copy()
        if len(strategy_log)
        else strategy_log.copy()
    )
    accounts = [
        _Account(account_id=index + 1, equity=challenge.starting_balance)
        for index in range(challenge.account_count)
    ]
    rows: list[dict] = []
    cursor = 0
    current_day = None
    occupied_until = -1
    consecutive_breaches = 0
    stopped_days: set[object] = set()
    phase1_passes = 0
    phase2_passes = 0
    challenges_passed = 0
    payouts_received = 0

    for _, source_row in ordered.iterrows():
        day = pd.Timestamp(source_row["trade_date_ny"]).date()
        if day != current_day:
            current_day = day
            consecutive_breaches = 0
            for account in accounts:
                account.day_pnl = 0.0
                account.day_trades = 0
                account.paused_day = None
        if day in stopped_days:
            continue

        entry_position = int(source_row["entry_position"])
        if entry_position <= occupied_until:
            continue
        selected = _next_account(accounts, cursor, day)
        if selected is None:
            continue
        index, account = selected
        stage_before = _stage_label(account, challenge)
        phase = _phase_for(account, challenge)
        risk_dollars = max(0.0, account.equity) * risk.risk_per_trade_pct / 100.0
        pnl = float(source_row["gross_r"]) * risk_dollars
        account.equity += pnl
        account.day_pnl += pnl
        account.stage_day_pnl[day] = account.day_pnl
        account.day_trades += 1
        account.total_trades += 1
        equity_after_trade = account.equity
        occupied_until = int(source_row["exit_position"])

        event = "Continue"
        daily_firm_floor = -challenge.starting_balance * phase.daily_loss_limit_pct / 100.0
        overall_floor = challenge.starting_balance * (1.0 - phase.max_loss_limit_pct / 100.0)
        target = challenge.starting_balance * (1.0 + phase.profit_target_pct / 100.0)
        breached = False
        if account.day_pnl <= daily_firm_floor:
            account.breach_reason = "Firm daily loss limit"
            breached = True
        elif account.equity <= overall_floor:
            account.breach_reason = "Firm overall loss limit"
            breached = True

        if breached:
            account.status = "BREACHED"
            event = account.breach_reason
            consecutive_breaches += 1
            cursor = (index + 1) % len(accounts)
            if consecutive_breaches >= 3:
                stopped_days.add(day)
        else:
            target_reached = account.equity >= target
            consistency_ok, consistency_ratio = _consistency_met(account, challenge, phase)

        if not breached and target_reached and consistency_ok:
            consecutive_breaches = 0
            if account.status == "FUNDED":
                account.status = "PAYOUT"
                account.payout = max(0.0, account.equity - challenge.starting_balance)
                payouts_received += 1
                event = "Payout received"
            else:
                if account.phase_index == 0:
                    phase1_passes += 1
                if account.phase_index + 1 < len(challenge.phases):
                    account.phase_index += 1
                    phase2_label = account.phase_index + 1
                    account.equity = challenge.starting_balance
                    account.day_pnl = 0.0
                    account.day_trades = 0
                    account.stage_day_pnl.clear()
                    account.paused_day = day
                    event = f"Advanced to Phase {phase2_label}"
                else:
                    if len(challenge.phases) == 2:
                        phase2_passes += 1
                    challenges_passed += 1
                    account.status = "FUNDED"
                    account.equity = challenge.starting_balance
                    account.day_pnl = 0.0
                    account.day_trades = 0
                    account.stage_day_pnl.clear()
                    account.paused_day = day
                    event = "Challenge passed - funded"
            cursor = (index + 1) % len(accounts)
        elif not breached:
            if target_reached and consistency_ratio is not None:
                event = f"Consistency {consistency_ratio:.1f}% > {phase.consistency_pct:.1f}%"
            daily_profit = challenge.starting_balance * risk.daily_profit_target_pct / 100.0
            daily_loss = challenge.starting_balance * risk.daily_loss_limit_pct / 100.0
            if daily_profit and account.day_pnl >= daily_profit:
                account.paused_day = day
                event = "Daily profit target"
            elif daily_loss and account.day_pnl <= -daily_loss:
                account.paused_day = day
                event = "Daily risk loss stop"
            elif risk.max_trades_per_day and account.day_trades >= risk.max_trades_per_day:
                account.paused_day = day
                event = "Daily trade cap"
            if account.paused_day == day:
                consecutive_breaches = 0
                cursor = (index + 1) % len(accounts)

        result_row = source_row.to_dict()
        result_row.update(
            {
                "account_id": account.account_id,
                "account_stage": stage_before,
                "simulated_pnl": pnl,
                "simulated_equity": equity_after_trade,
                "account_event": event,
                "account_status": account.status,
            }
        )
        rows.append(result_row)

    trades = pd.DataFrame(rows) if rows else _empty_log(ordered)
    if rows:
        trades = trades.reindex(columns=[*ordered.columns, *_ADDED_COLUMNS])
    summaries = tuple(
        AccountSummary(
            account_id=account.account_id,
            status=account.status,
            stage=_stage_label(account, challenge),
            equity=account.equity,
            trades=account.total_trades,
            payout=account.payout,
            breach_reason=account.breach_reason,
        )
        for account in accounts
    )
    return ChallengeSimulationResult(
        trades=trades,
        accounts=summaries,
        accounts_requested=challenge.account_count,
        phase1_passes=phase1_passes,
        phase2_passes=phase2_passes,
        challenges_passed=challenges_passed,
        payouts_received=payouts_received,
        accounts_breached=sum(account.status == "BREACHED" for account in accounts),
        total_payout=sum(account.payout for account in accounts),
        total_pnl=float(trades["simulated_pnl"].sum()) if len(trades) else 0.0,
        skipped_trades=max(0, len(ordered) - len(trades)),
        three_breach_stop_days=len(stopped_days),
    )
