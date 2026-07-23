"""GARCH(1,1) conditional volatility features for the statistical research branch.

The research question, definitions and acceptance criteria are pre-declared in
``project_docs/garch_volatility_research_contract.md``; this module implements
the feature side of that contract only.  It produces two columns:

``feat_garch_cond_vol``
    Per-bar conditional volatility in log-return units.
``feat_garch_high_vol_regime``
    ``True`` where conditional volatility exceeds the development-partition
    quantile named in the contract.

Leakage control
---------------
Parameters are estimated once on development bars and frozen.  The recursion is
then run forward over every bar using only information available before that
bar: ``sigma2[t]`` depends on ``eps[t-1]`` and ``sigma2[t-1]``.  The recursion
seed is the *variance of the development returns*, which is a development-only
quantity, so no validation or final-test observation influences any value.  The
regime threshold is likewise a development-only quantile.

Determinism
-----------
Fitting and the recursion run on the CPU path.  The recursion is sequential by
definition - ``sigma2[t]`` requires ``sigma2[t-1]`` - so it cannot be vectorised
across time or split across devices, and no attempt is made to do so.  Products
are independent of one another and are the unit of parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GarchConfig:
    """Frozen configuration for the GARCH(1,1) volatility feature."""

    #: Products fitted independently. Each gets its own parameter set.
    products: tuple[str, ...] = ("GC", "MGC")
    #: Development-partition quantile defining the high-volatility regime.
    high_vol_quantile: float = 0.75
    #: Returns are scaled before fitting purely for optimiser conditioning and
    #: unscaled afterwards; the scale cancels out of the reported feature.
    return_scale: float = 10_000.0
    #: Contract H3 sanity check: persistence above this is "high persistence".
    persistence_threshold: float = 0.90
    #: Persistence at or above this is treated as a non-stationary boundary
    #: solution and reported as such rather than as a passing sanity check.
    non_stationary_persistence: float = 0.999
    partition_column: str = "research_partition"
    development_partition: str = "development"
    price_column: str = "close"
    product_column: str = "product"
    cond_vol_column: str = "feat_garch_cond_vol"
    regime_column: str = "feat_garch_high_vol_regime"


@dataclass(frozen=True)
class GarchParameters:
    """Fitted GARCH(1,1) parameters for one product, on the scaled series."""

    product: str
    omega: float
    alpha: float
    beta: float
    mu: float
    seed_variance: float
    development_bars: int
    converged: bool
    optimiser_message: str

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def is_stationary(self, limit: float) -> bool:
        return self.persistence < limit

    def to_dict(self) -> dict[str, object]:
        return {
            "product": self.product,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "persistence": self.persistence,
            "seed_variance": self.seed_variance,
            "development_bars": self.development_bars,
            "converged": self.converged,
            "optimiser_message": self.optimiser_message,
        }


@dataclass
class GarchResult:
    """Fitted parameters plus the feature columns they generated."""

    config: GarchConfig
    parameters: dict[str, GarchParameters]
    conditional_volatility: np.ndarray
    high_vol_regime: np.ndarray
    threshold: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    def parameter_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.parameters.values()])


def log_returns_by_product(bars: pd.DataFrame, config: GarchConfig | None = None) -> np.ndarray:
    """One-minute log returns, reset at each product boundary.

    The first bar of every product has no predecessor and is returned as NaN
    rather than as a spurious jump across the product change.
    """

    config = config or GarchConfig()
    prices = pd.to_numeric(bars[config.price_column], errors="coerce").to_numpy("float64")
    products = bars[config.product_column].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[np.nan], returns])
    boundary = np.ones(len(prices), dtype=bool)
    boundary[1:] = products[1:] != products[:-1]
    returns[boundary] = np.nan
    returns[~np.isfinite(returns)] = np.nan
    return returns


def fit_development_garch(
    returns: np.ndarray,
    development_mask: np.ndarray,
    *,
    product: str,
    config: GarchConfig | None = None,
) -> GarchParameters:
    """Fit GARCH(1,1) on development bars only and freeze the parameters."""

    from arch import arch_model

    config = config or GarchConfig()
    development = np.asarray(returns, dtype="float64")[np.asarray(development_mask, dtype=bool)]
    development = development[np.isfinite(development)]
    if development.size < 100:
        raise ValueError(
            f"{product}: {development.size} finite development returns is too few to fit GARCH(1,1)"
        )
    scaled = development * config.return_scale
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="Normal")
    fitted = model.fit(disp="off")
    # arch exposes the SciPy optimiser result; code 0 is a genuine convergence.
    optimiser = getattr(fitted, "optimization_result", None)
    converged = bool(getattr(optimiser, "success", True))
    message = str(getattr(optimiser, "message", "")).strip() or "not reported"
    return GarchParameters(
        product=product,
        omega=float(fitted.params["omega"]),
        alpha=float(fitted.params["alpha[1]"]),
        beta=float(fitted.params["beta[1]"]),
        mu=float(fitted.params["mu"]),
        seed_variance=float(np.var(scaled)),
        development_bars=int(development.size),
        converged=converged,
        optimiser_message=message,
    )


def forward_conditional_volatility(
    returns: np.ndarray,
    parameters: GarchParameters,
    *,
    config: GarchConfig | None = None,
) -> np.ndarray:
    """Apply the frozen recursion forward over every bar, using only the past.

    The seed is the development return variance.  Seeding with the fitted
    long-run variance ``omega / (1 - alpha - beta)`` is not usable here because
    a boundary fit with persistence at one makes that quantity undefined.
    """

    config = config or GarchConfig()
    scaled = np.asarray(returns, dtype="float64") * config.return_scale
    innovations = scaled - parameters.mu
    # A missing return contributes no shock; the recursion decays instead of
    # propagating a NaN through the remainder of the series.
    innovations = np.where(np.isfinite(innovations), innovations, 0.0)
    n = innovations.size
    sigma2 = np.empty(n, dtype="float64")
    if n == 0:
        return sigma2
    sigma2[0] = parameters.seed_variance
    omega, alpha, beta = parameters.omega, parameters.alpha, parameters.beta
    for t in range(1, n):
        sigma2[t] = omega + alpha * innovations[t - 1] ** 2 + beta * sigma2[t - 1]
    conditional = np.sqrt(sigma2) / config.return_scale
    conditional[~np.isfinite(np.asarray(returns, dtype="float64"))] = np.nan
    return conditional


def build_garch_features(bars: pd.DataFrame, config: GarchConfig | None = None) -> GarchResult:
    """Fit per product, apply forward, and derive the development-set regime flag."""

    config = config or GarchConfig()
    for column in (config.price_column, config.product_column, config.partition_column):
        if column not in bars.columns:
            raise KeyError(f"bars is missing required column {column!r}")

    returns = log_returns_by_product(bars, config)
    products = bars[config.product_column].to_numpy()
    is_development = (
        bars[config.partition_column].astype("string").to_numpy() == config.development_partition
    )

    conditional = np.full(len(bars), np.nan, dtype="float64")
    parameters: dict[str, GarchParameters] = {}
    for product in config.products:
        selector = products == product
        if not selector.any():
            continue
        product_returns = returns[selector]
        product_dev = is_development[selector]
        if not product_dev.any():
            raise ValueError(f"{product}: no development bars available to fit GARCH")
        fitted = fit_development_garch(product_returns, product_dev, product=product, config=config)
        parameters[product] = fitted
        conditional[selector] = forward_conditional_volatility(
            product_returns, fitted, config=config
        )

    development_values = conditional[is_development]
    development_values = development_values[np.isfinite(development_values)]
    if development_values.size == 0:
        raise ValueError("no finite development conditional volatility to derive a threshold")
    threshold = float(np.quantile(development_values, config.high_vol_quantile))
    regime = np.where(np.isfinite(conditional), conditional > threshold, False)

    diagnostics = {
        "threshold": threshold,
        "development_bars": int(is_development.sum()),
        "total_bars": int(len(bars)),
        "regime_rate_development": float(regime[is_development].mean()),
        "non_stationary_products": sorted(
            product
            for product, fitted in parameters.items()
            if not fitted.is_stationary(config.non_stationary_persistence)
        ),
        "non_converged_products": sorted(
            product for product, fitted in parameters.items() if not fitted.converged
        ),
    }
    return GarchResult(
        config=config,
        parameters=parameters,
        conditional_volatility=conditional,
        high_vol_regime=regime.astype(bool),
        threshold=threshold,
        diagnostics=diagnostics,
    )


def attach_garch_features(
    bars: pd.DataFrame, config: GarchConfig | None = None
) -> tuple[pd.DataFrame, GarchResult]:
    """Return a copy of ``bars`` with the two contract feature columns attached."""

    config = config or GarchConfig()
    result = build_garch_features(bars, config)
    enriched = bars.copy()
    enriched[config.cond_vol_column] = result.conditional_volatility
    enriched[config.regime_column] = result.high_vol_regime
    return enriched, result
