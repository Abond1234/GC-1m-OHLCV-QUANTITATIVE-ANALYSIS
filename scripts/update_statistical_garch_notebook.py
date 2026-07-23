"""Rewrite the GARCH volatility section of the statistical notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "exploration" / "statistical_feature_research.ipynb"
SECTION_HEADING = "# GARCH Volatility Modelling"


def md(source: str):
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
section_start = next(
    (
        i
        for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith(SECTION_HEADING)
    ),
    len(notebook.cells),
)
notebook.cells = notebook.cells[:section_start]

section_cells = [
    md(
        """
# GARCH Volatility Modelling

GARCH(1,1) models **volatility clustering**: large moves tend to follow large moves and calm
periods cluster together. The research question, frozen definitions and acceptance criteria are
pre-declared in `project_docs/garch_volatility_research_contract.md`.

**Model:** sigma^2_t = omega + alpha * eps^2_(t-1) + beta * sigma^2_(t-1)

The implementation lives in `src/statistical_research/garch_volatility.py` and is covered by
`tests/test_garch_volatility.py`, including a direct leakage test.

**Leakage control.** Parameters are estimated on development bars only and frozen. The recursion
is then applied forward over every bar using only prior information, seeded with the development
return variance - a development-only quantity. The regime threshold is the development-partition
75th percentile. No validation or final-test observation influences any feature value.

**Scope of this section.** It produces the two contract feature columns and reports the fit
diagnostics. It does **not** evaluate contract hypotheses H1 or H2; see the status note in
`reports/statistical_research/summaries/garch_volatility_status.md` for what remains outstanding
and why. Consistent with the contract, the final-test partition is not summarised here.
"""
    ),
    code(
        """
from src.statistical_research.garch_volatility import GarchConfig, attach_garch_features

# Reload the trusted bars if an earlier memory-release cell dropped them.
if "research_bars" not in globals() or research_bars is None:
    print("Loading trusted research_bars dataset from DATA_PATH...")
    research_bars = pd.read_parquet(DATA_PATH, engine="pyarrow", dtype_backend="pyarrow")

if "research_partition" not in research_bars.columns:
    _trade_dates = pd.to_datetime(research_bars["trade_date_ny"])
    research_bars["research_partition"] = np.select(
        [
            _trade_dates.le(pd.Timestamp("2023-12-31")),
            _trade_dates.between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
            _trade_dates.ge(pd.Timestamp("2025-01-01")),
        ],
        ["development", "validation", "final_test"],
        default="unassigned",
    )

GARCH_CONFIG = GarchConfig()
research_bars, garch_result = attach_garch_features(research_bars, GARCH_CONFIG)

print("Fitted GARCH(1,1) parameters (development partition only):")
display(garch_result.parameter_frame().round(6))
print(f"High-volatility threshold (development p{GARCH_CONFIG.high_vol_quantile:.0%}): "
      f"{garch_result.threshold:.8f}")
"""
    ),
    code(
        """
# ---------------------------------------------------------------------------
# Fit quality. Persistence at the stationarity boundary is a specification
# signal, not a passing sanity check, so it is surfaced explicitly.
# ---------------------------------------------------------------------------
for product, params in garch_result.parameters.items():
    flags = []
    if not params.converged:
        flags.append(f"optimiser did not converge ({params.optimiser_message})")
    if not params.is_stationary(GARCH_CONFIG.non_stationary_persistence):
        flags.append("persistence at the stationarity boundary (integrated GARCH)")
    status = "; ".join(flags) if flags else "converged, stationary"
    print(f"{product}: alpha={params.alpha:.4f} beta={params.beta:.4f} "
          f"persistence={params.persistence:.4f} -> {status}")

garch_fit_is_clean = not (
    garch_result.diagnostics["non_converged_products"]
    or garch_result.diagnostics["non_stationary_products"]
)
print(f"\\nnon-converged products : {garch_result.diagnostics['non_converged_products']}")
print(f"non-stationary products: {garch_result.diagnostics['non_stationary_products']}")
"""
    ),
    code(
        """
# ---------------------------------------------------------------------------
# Regime summary, development and validation only. The contract reserves the
# final-test partition for a single read after the hypothesis verdicts are
# fixed, and those verdicts are not yet fixed.
# ---------------------------------------------------------------------------
EVALUATED_PARTITIONS = ["development", "validation"]
garch_summary = (
    research_bars.loc[research_bars["research_partition"].isin(EVALUATED_PARTITIONS)]
    .groupby(["product", "research_partition"], observed=True)
    .agg(
        bars=("feat_garch_cond_vol", "size"),
        mean_cond_vol=("feat_garch_cond_vol", "mean"),
        high_vol_rate=("feat_garch_high_vol_regime", "mean"),
    )
    .round(8)
)
display(garch_summary)

assert "final_test" not in garch_summary.index.get_level_values("research_partition"), (
    "final-test partition must not be summarised before the hypothesis verdicts are fixed"
)
"""
    ),
    code(
        """
# ---------------------------------------------------------------------------
# Conditional volatility and regime clustering, GC, development + validation.
# ---------------------------------------------------------------------------
garch_plot_frame = (
    research_bars.loc[
        research_bars["product"].eq("GC")
        & research_bars["tradable_research_flag"]
        & research_bars["research_partition"].isin(EVALUATED_PARTITIONS)
    ]
    .sort_values("ts_event_utc")
    .set_index("ts_event_utc")
    .resample("15min")
    .last()
    .dropna(subset=["feat_garch_cond_vol"])
)
regime_mask = garch_plot_frame["feat_garch_high_vol_regime"].to_numpy(dtype=bool)

figure, axis = plt.subplots(figsize=(14, 4))
axis.plot(
    garch_plot_frame.index,
    garch_plot_frame["feat_garch_cond_vol"],
    color="darkorange",
    linewidth=0.8,
    label="GARCH conditional volatility",
)
axis.axhline(
    garch_result.threshold,
    color="crimson",
    linewidth=1.2,
    linestyle="--",
    label=f"Development p75 threshold ({garch_result.threshold:.5f})",
)
axis.fill_between(
    garch_plot_frame.index,
    0,
    garch_plot_frame["feat_garch_cond_vol"],
    where=regime_mask,
    color="firebrick",
    alpha=0.15,
)
axis.set_title("GC — GARCH(1,1) conditional volatility (development + validation)")
axis.set_ylabel("Conditional volatility (log-return units)")
axis.set_xlabel("UTC timestamp")
axis.legend(loc="upper left", fontsize=9)
figure.tight_layout()
plt.show()
"""
    ),
    code(
        """
# ---------------------------------------------------------------------------
# Gate: the feature is built and leakage-safe. This asserts integrity only and
# deliberately does not assert any particular research outcome.
# ---------------------------------------------------------------------------
garch_checks = {
    "both_products_fitted": set(garch_result.parameters) == set(GARCH_CONFIG.products),
    "cond_vol_column_present": GARCH_CONFIG.cond_vol_column in research_bars.columns,
    "regime_column_present": GARCH_CONFIG.regime_column in research_bars.columns,
    "threshold_positive": garch_result.threshold > 0.0,
    "development_regime_rate_matches_quantile": abs(
        garch_result.diagnostics["regime_rate_development"]
        - (1.0 - GARCH_CONFIG.high_vol_quantile)
    ) < 0.01,
    "final_test_not_summarised": "final_test" not in set(
        garch_summary.index.get_level_values("research_partition")
    ),
}
for name, passed in garch_checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")

garch_section_ready = all(garch_checks.values())
assert garch_section_ready, "GARCH integrity checks failed"
print(
    "\\nHypotheses H1 and H2 remain unevaluated; see "
    "reports/statistical_research/summaries/garch_volatility_status.md"
)
print("\\nGARCH SECTION STATUS: READY")
"""
    ),
]

notebook.cells.extend(section_cells)
nbformat.write(notebook, NOTEBOOK_PATH)
print(f"GARCH section rewritten: {len(section_cells)} cells, {len(notebook.cells)} total")
