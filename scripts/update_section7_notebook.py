"""Insert completed True POI context research before the legacy appendix."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import nbformat

NOTEBOOK = Path("notebooks/exploration/exp1.ipynb")
BACKUP_DIR = Path("logs/notebook_backups")
APPENDIX_ID = "7e076934"


def markdown(cell_id: str, source: str):
    cell = nbformat.v4.new_markdown_cell(source.strip())
    cell.id = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = nbformat.v4.new_code_cell(source.strip())
    cell.id = cell_id
    return cell


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb.cells = [cell for cell in nb.cells if not cell.id.startswith("7ctx")]
    appendix_index = next(i for i, cell in enumerate(nb.cells) if cell.id == APPENDIX_ID)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"exp1_pre_section7_context_{datetime.now():%Y%m%d_%H%M%S}.ipynb"
    shutil.copy2(NOTEBOOK, backup)

    # Correct only the stale exit statement in completed Section 5.8.
    target_cell = next(
        cell for cell in nb.cells if cell.source.startswith("## 5.8 Target / Exit Logic")
    )
    target_cell.source = target_cell.source.replace(
        "All trades flat by 12:00pm New York time.\nNo overnight positions.",
        "12:00pm New York time = no new entries and True POI expiry.\n"
        "Positions entered before noon may remain open.\n"
        "3:30pm New York time = mandatory exit for every open position.\n"
        "No overnight positions.",
    )

    # Migrate current reader-facing Section 7 text while preserving legacy
    # internal column names in the historical input files.
    nb.cells[311].source = (
        "# 7.0 POI Context & Signal Research\n\n"
        "## 7.1 Research Contract and Authoritative Inputs\n\n"
        "The refined **True POI** definition is frozen. A True POI is one unique refined "
        "A/B/C market formation after duplicate swing-window and break-mode representations "
        "are removed; the term does not imply profitability or validation. The authoritative "
        "input remains `section6c_refined_signal_frame_gc.parquet`. Legacy `canonical_*` "
        "columns remain compatibility fields only; all new objects use `true_poi_id`, "
        "`true_retest_id`, and `true_trade_opportunity_id`. Section 8 has not started."
    )
    nb.cells[312].source = (
        "## 7.2 Refined Baseline Event Study\n\n"
        "The saved refined event study is the baseline that motivated this context pass. "
        "Its reader-facing population is the True POI-deduplicated candidate population. "
        "Internal `section7r_*` names and legacy identity fields remain unchanged solely for reproducibility."
    )
    nb.cells[
        313
    ].source = """section7r_summary = pd.read_parquet(section6c_output_dir / "section7r_event_study_summary_gc.parquet")
section7r_ranking = pd.read_parquet(section6c_output_dir / "section7r_candidate_signal_ranking_gc.parquet")
section7r_fvg_thresholds = pd.read_parquet(section6c_output_dir / "section7r_fvg_threshold_comparison_gc.parquet")
section7r_true_poi_60m = section7r_summary.query(
    "analysis_population == 'canonical_candidate_variants' and metric_mode == 'capped' and horizon_minutes == 60"
).copy()
section7r_true_poi_60m["analysis_population"] = "true_poi_candidate_variants"
section7r_true_poi_60m.loc[
    section7r_true_poi_60m["group_spec"].isin(["baseline_all_candidates", "poi_touch_direction", "poi_geometry_case", "structural_window", "execution_session"]),
    ["group_spec", "group_key", "valid_count", "mean_r_cap_5", "median_r", "positive_r_rate", "hit_neg_1r_rate"],
]"""
    nb.cells[314].source = (
        "## 7.3 Current Findings and Limitations\n\n"
        "The refined baseline showed mild positive 60-minute central tendency and a broad short-side "
        "asymmetry, but independent MFE/MAE hits could not determine whether a target preceded a stop. "
        "It also expanded each retest by entry and stop variants. Those results therefore remain a "
        "market-event baseline, not an executable policy approval. The completed work below replaces "
        "headline variant counts with one True POI/retest row, creates paired continuation and reversal "
        "labels, and uses ordered first passage for policy decisions."
    )
    nb.cells[315].source = (
        "## 7.4 Feature-Engineering and Conditional-Signal Roadmap\n\n"
        "The roadmap is now implemented in reusable modules. It covers formation geometry, displacement, "
        "five pre-touch approach windows, touch interaction, VWAP, trend, volatility, volume, session, "
        "extension/exhaustion, paired directional labels, first-passage stops and targets, chronological "
        "feature studies, twelve pre-specified interactions, and exact policy ranking."
    )
    nb.cells[316].source = (
        "## 7.5 Section 7 Current Status\n\n"
        "Section 7 is complete as an event-level context and policy-screening study. It is not a sequential "
        "backtest. No policy met the advancement standard; one exact policy remains `RESEARCH_ONLY` and "
        "three frozen policies are rejected. Consequently: **No Section 8 candidate approved.**"
    )

    cells = [
        markdown(
            "7ctx0600",
            """
## 7.6 Research Design, Feature Registry and No-Lookahead Contract

Research question: which information known at formation, immediately before touch, or only after the touch candle closes helps distinguish continuation, reversal, and no trade?

The registry contains all 264 delivered `feat_*` columns: 29 formation/geometry, 24 displacement, 132 approach, 23 touch-interaction, 49 market-context, four retest-state, and three session-context fields. Same-bar boundary entries may use formation and pre-touch features only. Touch-close features are restricted to next-bar confirmation entries. Rolling and time-of-day fields are past-only; development thresholds are locked before validation and final test. Predictor, label, policy, and diagnostic namespaces use `feat_`, `label_`, `policy_`, and `diag_`.
        """,
        ),
        code(
            "7ctx0601",
            """
section7_processed = project_root_path / "data" / "processed"
section7_reports = project_root_path / "reports"
section7_feature_registry = pd.read_csv(section7_reports / "section7_true_poi_feature_registry.csv")
section7_feature_registry.groupby(
    ["feature_family", "availability_timestamp"], observed=True
).size().rename("feature_count").to_frame()
        """,
        ),
        markdown(
            "7ctx0700",
            """
## 7.7 True POI Research Unit and Decision-Time Dataset

The delivered `section7_true_poi_context_frame_gc.parquet` has exactly one row per `true_poi_id × true_retest_id` before directional or stop/target expansion. It contains 7,433 True POIs, 179,036 unique retests, and 882 trading dates. Research variants are retained as counts and 3/5/7-bar plus wick/close flags; they are never independent headline observations.

Chronological coverage is 776 True POIs / 13,643 retests / 352 dates in development, 694 / 11,487 / 194 in validation, and 5,963 / 153,906 / 336 in final test. First and later touches remain in the object; later touches are not silently discarded.
        """,
        ),
        code(
            "7ctx0701",
            """
section7_context = pd.read_parquet(section7_processed / "section7_true_poi_context_frame_gc.parquet")
section7_context.groupby("research_partition", observed=True).agg(
    true_pois=("true_poi_id", "nunique"),
    true_retests=("true_retest_id", "nunique"),
    trading_dates=("trade_date_ny", "nunique"),
)
        """,
        ),
        markdown(
            "7ctx0800",
            """
## 7.8 True POI Formation and Displacement Features

Formation-time features measure corrected Case 1/Case 2 geometry, POI/FVG/opening-gap size in ticks and ATR, body/range and directional close location for B/C, structural-break strength, active research-variant coverage, and 15-bar versus local structure. The displacement interval is the maximal associated interval already active before the retest; it cannot cross date, contract, or continuous-segment boundaries.

The strongest locked development formation bucket was a very small opening-gap-to-POI-width ratio: robust 60-minute continuation mean was 0.439R in development, 0.229R in validation, and 0.290R in final test. It is retained as a context variable, not a rule, because the full relationship was non-monotonic. High displacement relative volume looked strong in validation (0.676R) but weakened to 0.124R in final test; it remains interaction-only.
        """,
        ),
        code(
            "7ctx0801",
            """
section7_feature_study = pd.read_parquet(section7_processed / "section7_true_poi_feature_study_summary_gc.parquet")
section7_feature_study.query(
    "feature_name in ['feat_opening_gap_to_poi_width_ratio', 'feat_displacement_relative_volume', 'feat_displacement_efficiency'] "
    "and hypothesis == 'continuation'"
)[["feature_name", "feature_bin", "research_partition", "true_retest_count", "mean_r", "median_r", "monotonic_direction"]]
        """,
        ),
        markdown(
            "7ctx0900",
            """
## 7.9 Approach-to-POI Features

Approach features are calculated over 3, 5, 10, 15, and 30 completed minutes ending at `touch_bar - 1`. They quantify origin distance, speed, slope, acceleration, net/gross path, efficiency/chop, opposing candles, counter-move, overlap, compression/expansion, volume level/slope/acceleration, and displacement-to-approach ratios.

Low 15-minute candle overlap was the cleanest approach observation: the locked low-overlap bucket produced robust continuation means of 0.158R, 0.231R, and 0.191R across development, validation, and test. The final-test ordering was not monotonic, so overlap is retained for conditional research rather than hard permission. Fast approach and displacement/approach volume interactions did not remain stable enough to become policies.
        """,
        ),
        code(
            "7ctx0901",
            """
section7_feature_study.query(
    "feature_name.str.startswith('feat_approach_15m') and hypothesis == 'continuation'",
    engine="python",
).sort_values(["research_partition", "mean_r"], ascending=[True, False]).head(20)[
    ["feature_name", "feature_bin", "research_partition", "true_retest_count", "mean_r", "q25_r", "monotonic_direction"]
]
        """,
        ),
        markdown(
            "7ctx1000",
            """
## 7.10 POI Interaction and Retest Features

Penetration is direction-neutral: distance from the first-contact edge into the zone divided by POI width. The object records boundary/midpoint/distal penetration, full traversal, wick/close through, close inside/outside, candle body and wicks, rejection-wick/body ratio, close location, relative volume, and recent bars inside.

Touch-close fields are never available to `boundary_touch`. The only frozen touch-confirmation policies entered at the next eligible bar open. Bullish-POI reversal with a touch-candle stop lost −0.153R/−0.154R/−0.152R in development/validation/test for the broad exhaustion policy, while the narrower rejection-and-extension policy lost −0.153R/−0.059R/−0.072R. Touch rejection therefore failed as a standalone directional permission rule.
        """,
        ),
        code(
            "7ctx1001",
            """
section7_candidate_ranking = pd.read_parquet(section7_processed / "section7_true_poi_candidate_ranking_gc.parquet")
section7_candidate_ranking.query(
    "policy_id in ['S7P04_NY_BULL_REV_CONFIRM', 'S7P06_NY_BEAR_REV']"
)[["policy_id", "research_partition", "true_poi_count", "true_retest_count", "mean_realized_r", "median_realized_r", "same_bar_ambiguity_rate"]]
        """,
        ),
        markdown(
            "7ctx1100",
            """
## 7.11 Market-State and Context Features

Past-only VWAP, 5/15/30-minute VWAP slopes, VWAP crosses, 5/15/30/60-minute trend, rolling structure, short/long volatility, development-fitted volatility buckets, raw/relative/time-of-day volume, session state, time-to-cutoff, extension, decay, and compression fields were added. Candle-signed volume is not described as order-flow delta.

The highest development time-of-day-adjusted relative-volume bucket remained favourable for continuation (0.155R development, 0.206R validation, 0.189R final test), but the overall relation was non-monotonic. Volume remains a target/holding/risk context variable, not a buy/sell trigger.
        """,
        ),
        code(
            "7ctx1101",
            """
section7_feature_study.query(
    "feature_name in ['feat_time_of_day_adjusted_relative_volume', 'feat_distance_from_vwap_atr', "
    "'feat_short_to_long_volatility_ratio', 'feat_session_range_before_retest_ticks']"
)[["feature_name", "hypothesis", "feature_bin", "research_partition", "true_retest_count", "mean_r", "q25_r", "monotonic_direction"]]
        """,
        ),
        markdown(
            "7ctx1200",
            """
## 7.12 True POI Location-Quality Study

True POI retests were matched to non-POI bars by chronological partition, session, 30-minute time bin, development-fitted volatility bucket, and recent-move bucket. Median 60-minute expansion was 5.196 versus 4.778 ATR in development, 5.256 versus 5.099 ATR in validation, and 5.944 versus 5.560 ATR in final test.

The location effect is positive but modest relative to the matched state. Both POI and control samples exceeded 0.5 and 1.0 one-minute ATR almost universally, so those thresholds are not discriminative at a 60-minute horizon. Decision: retain True POI as an event location, but do not infer a directional trade from location quality alone.
        """,
        ),
        code(
            "7ctx1201",
            """
section7_location_quality = pd.read_parquet(
    section7_processed / "section7_true_poi_location_quality_summary_gc.parquet"
)
section7_location_quality
        """,
        ),
        markdown(
            "7ctx1300",
            """
## 7.13 Continuation and Reversal Outcome Labels

Each retest has paired continuation and reversal labels at 5/15/30/60/120/180/240 minutes, with forced-exit-capped versions, MFE, MAE, time to 60-minute MFE/MAE, zone hold/failure, immediate exit/rejection, re-entry, and location reaction labels.

Robust ±5R 60-minute continuation-short means were 0.094R, 0.137R, and 0.185R across development, validation, and test. Continuation-long was 0.046R, −0.064R, and 0.124R. Reversal-short appeared positive in development/validation (0.195R/0.124R) but failed final test (−0.105R); reversal-long was 0.116R/−0.185R/−0.060R. The ordered policy engine below confirms that broad reversal is not executable under the tested stops.
        """,
        ),
        code(
            "7ctx1301",
            """
section7_labels = pd.read_parquet(section7_processed / "section7_true_poi_outcome_labels_gc.parquet")
section7_labels.assign(robust_60m_r=section7_labels["label_capped_60m_r"].clip(-5, 5)).groupby(
    ["research_partition", "hypothesis", "trade_side"], observed=True
).agg(
    true_retests=("true_retest_id", "nunique"), mean_robust_r=("robust_60m_r", "mean"),
    median_r=("label_capped_60m_r", "median"), positive_r_rate=("label_capped_60m_r", lambda x: x.gt(0).mean()),
)
        """,
        ),
        markdown(
            "7ctx1400",
            """
## 7.14 First-Passage Stop/Target Engine

The chunked engine evaluates entry fill, target-before-stop, stop-before-target, time/mandatory exit before either, no fill, invalid path, same-bar ambiguity, entry-bar ambiguity, time to each barrier, forced-exit R, and maximum uncapped/adverse R. It rejects entries after noon and paths crossing a New York date, active contract, continuous segment, or unavailable forced-exit bar. Complexity is `O(events × path bars × targets)` with bounded chunk memory.

Conservative stop-first is the headline treatment. At the primary 2R/True-POI-invalidation policy, continuation same-bar ambiguity rose from 4.16% development to 4.89% validation and 17.13% final test. Excluding ambiguous rows changed final-test mean from −0.169R to +0.054R; optimistic target-first changed it to +0.464R. That sensitivity is too large to ignore and directly blocks approval of same-bar variants that depend on favorable bar ordering.
        """,
        ),
        code(
            "7ctx1401",
            """
section7_stop_target = pd.read_parquet(section7_processed / "section7_true_poi_stop_target_summary_gc.parquet")
section7_stop_target.query(
    "analysis_type == 'first_passage_target' and stop_model == 'poi_invalidation_1tick' "
    "and target_r == 2 and holding_policy == '15:30_mandatory'"
)[["research_partition", "hypothesis", "ambiguity_treatment", "opportunity_count", "mean_realized_r", "target_before_stop_rate", "stop_before_target_rate", "same_bar_ambiguity_rate"]]
        """,
        ),
        markdown(
            "7ctx1500",
            """
## 7.15 Stop-Loss Policy Research

Five pre-specified families were tested: one-tick True POI invalidation, recent confirmed micro-swing, volatility hybrid, touch/rejection candle for next-bar entries only, and the legacy conservative-adjacent reference.

For continuation at 2R/240 minutes, volatility hybrid was the only broad family with positive expectancy in every partition: 0.056R development, 0.039R validation, and 0.043R final test; same-bar ambiguity was 0.50%, 0.55%, and 2.17%. Recent micro-swing was also positive (0.020R/0.029R/0.032R) but weaker. The one-tick True POI stop failed out of sample (−0.034R validation, −0.169R test) and became highly ambiguous. Touch-candle stops were negative near −0.10R. Every broad median and 25th percentile remained −1R.

Only 34% of validation volatility-hybrid risks met the legacy 25–100 tick cohort; no row was silently removed. The compatibility range is descriptive here, not an automatic exclusion.
        """,
        ),
        code(
            "7ctx1501",
            """
section7_stop_target.query(
    "analysis_type == 'first_passage_target' and ambiguity_treatment == 'conservative' "
    "and target_r == 2 and holding_policy == '240m'"
)[["research_partition", "hypothesis", "stop_model", "entry_model", "mean_realized_r", "median_realized_r", "q25_realized_r", "same_bar_ambiguity_rate", "risk_25_100_tick_rate", "median_risk_ticks"]]
        """,
        ),
        markdown(
            "7ctx1600",
            """
## 7.16 Target and Holding-Period Research

Fixed targets of 1/1.5/2/3/4/5/7.5/10/15R and time exits of 15/30/60/120/180/240 minutes plus the 3:30pm mandatory exit were evaluated. True profit potential was not clipped in first-passage outcomes; ±5R clipping is used only for robust feature diagnostics.

No universal target was selected. Higher targets occasionally improved mean R through rare runners while median and 25th percentile stayed −1R, so choosing the best target would be tail-driven optimization. Continuation with a volatility-hybrid stop remained positive across 60/120/240-minute exits in all partitions, but win rates were low and downside concentrated at the stop. Section 8, if later authorized, should compare the 2R/240-minute policy with neighbouring 2R/180-minute and 3R/240-minute rules rather than treating one point estimate as optimal.
        """,
        ),
        code(
            "7ctx1601",
            """
section7_stop_target.query(
    "analysis_type == 'time_exit' and hypothesis == 'continuation' and stop_model == 'volatility_hybrid' "
    "and holding_policy in ['60m', '120m', '240m']"
)[["research_partition", "holding_policy", "opportunity_count", "mean_realized_r", "median_realized_r", "q25_realized_r", "positive_r_rate"]]
        """,
        ),
        markdown(
            "7ctx1700",
            """
## 7.17 Conditional Feature Studies

Development quintile edges were applied unchanged to validation and test. Every row reports True POI/retest/date counts, trading-date block-bootstrap confidence intervals, downside quantiles, monotonicity, raw p-values, and development Benjamini–Hochberg q-values.

Retained for further conditional use: opening-gap/POI-width ratio, 15-minute candle overlap, time-of-day-adjusted relative volume, displacement efficiency/relative volume, POI age/touch count, and volatility/session extension measures. They may influence permission, stop, target, or holding time. Rejected as standalone directional rules: first touch, raw volume spike, fast approach, session-range extremes, touch rejection, and any isolated best quintile without monotonic or out-of-sample support.
        """,
        ),
        code(
            "7ctx1701",
            """
section7_feature_study.query(
    "research_partition in ['validation', 'final_test']"
).sort_values("mean_r", ascending=False)[
    ["feature_name", "hypothesis", "feature_bin", "research_partition", "true_poi_count", "true_retest_count", "trading_date_count", "mean_r", "q25_r", "date_cluster_bootstrap_ci_low", "date_cluster_bootstrap_ci_high", "monotonic_direction"]
].head(30)
        """,
        ),
        markdown(
            "7ctx1800",
            """
## 7.18 Pre-Specified Interaction Studies

Twelve hypotheses were written before evaluation. Continuation interactions that helped in both validation and final test were 15-bar structure × displacement quality, volatility × normalized stop width, early-London short context, and standard geometry × normalized FVG. The first three validation advantages versus their complements were +1.396R, +0.505R, and +0.485R in robust 60-minute labels, but the 15-bar and London cohorts had only 33 and 21 validation dates.

Interactions that failed stability included displacement volume × approach volume, displacement quality × approach quality, compression × continuation, and trend alignment. Extension/exhaustion × reversal improved validation and test modestly but not development, so it remains a hypothesis rather than a rule.
        """,
        ),
        code(
            "7ctx1801",
            """
section7_interactions = pd.read_parquet(section7_processed / "section7_true_poi_interaction_summary_gc.parquet")
section7_interactions.query(
    "condition_met == True and research_partition in ['validation', 'final_test']"
)[["interaction_name", "pre_registered_hypothesis", "research_partition", "hypothesis", "true_poi_count", "true_retest_count", "trading_date_count", "mean_r", "median_r", "q25_r"]]
        """,
        ),
        markdown(
            "7ctx1900",
            """
## 7.19 Out-of-Sample Validation and Candidate Ranking

Splits are development 2021-05-24–2023-12-31, validation calendar 2024, and final test 2025-01-01–2026-05-22. Actual available retests begin 2021-05-26. Bins use development only; summaries cluster/bootstrap by trading date and group by True POI. Candidate rules were frozen from development plus validation before their final-test results were read.

Only four pre-specified policies met the minimum freeze coverage. The leading policy—New York bearish True POI continuation short, Case 2, compressed 15-minute approach, boundary entry, volatility-hybrid stop, 3R target, 240-minute maximum hold—returned +0.102R development, +0.122R validation, and +0.037R final test over 323/245/1,862 True POIs and 200/130/308 dates. Yet its median and 25th percentile were −1R in every partition and final-test stop-before-target was 58.31%. It is research-only, not approved.
        """,
        ),
        code(
            "7ctx1901",
            """
section7_candidate_ranking.loc[
    section7_candidate_ranking["frozen_before_final_test"],
    ["policy_id", "research_partition", "true_poi_count", "true_retest_count", "trading_date_count", "mean_realized_r", "median_realized_r", "q25_realized_r", "target_before_stop_rate", "stop_before_target_rate", "same_bar_ambiguity_rate", "positive_year_rate", "worst_year_mean_r"],
].sort_values(["policy_id", "research_partition"])
        """,
        ),
        markdown(
            "7ctx2000",
            """
## 7.20 Section 8 Candidate Registry and Go/No-Go Decision

The registry contains four exact implementable policies:

1. `S7P02_NY_BEAR_CONT` — **RESEARCH_ONLY**. Positive expectancy in all partitions, but median/Q25 = −1R and performance is runner-dependent.
2. `S7P05_NY_BULL_CONT` — **REJECT**. +0.018R validation but −0.077R final test.
3. `S7P04_NY_BULL_REV_CONFIRM` — **REJECT**. −0.153R/−0.059R/−0.072R across the three partitions.
4. `S7P06_NY_BEAR_REV` — **REJECT**. −0.132R/−0.154R/−0.152R across the three partitions.

No policy satisfies positive stable expectancy, reasonable downside, acceptable ambiguity, multi-year stability, and a non-fragile exact rule simultaneously. **No Section 8 candidate approved.** Section 8 must not start from this registry without a new explicit research decision.
        """,
        ),
        code(
            "7ctx2001",
            """
section7_candidate_registry = pd.read_parquet(
    section7_processed / "section7_true_poi_backtest_candidate_registry_gc.parquet"
)
section7_candidate_registry[[
    "policy_id", "hypothesis", "trade_side", "eligible_session", "poi_geometry",
    "required_feature_conditions", "entry_model", "stop_model", "target_model",
    "maximum_holding_minutes", "decision", "decision_reason",
]]
        """,
        ),
        markdown(
            "7ctx2100",
            """
## 7.21 Final Section 7 Summary

Section 7 is complete. The full context pipeline processed 1,080,394 legacy signal rows into 179,036 unique True POI/retest opportunities and 358,072 paired directional labels. The main run took 321.1 seconds; the matched-control location pass added 8.7 seconds. Peak observed RSS was approximately 4.05 GB during first passage. Generated Parquet outputs remain ignored by Git.

What survived: True POIs are modestly better expansion locations than matched controls; continuation-short remains the strongest broad direction; volatility-hybrid and micro-swing stops materially reduce same-bar ambiguity; several formation/approach/context variables merit interaction use; and one exact New York bearish-continuation policy deserves research-only monitoring.

What failed: automatic POI trades, broad reversal, first-touch permission, touch-rejection entries, raw volume spikes, universal structural permission, a universal target, and every frozen policy under the full advancement standard. The correct go/no-go result is **no-go for Section 8 approval**.
        """,
        ),
        code(
            "7ctx2101",
            """
pd.Series({
    "true_pois": section7_context["true_poi_id"].nunique(),
    "true_retests": section7_context["true_retest_id"].nunique(),
    "trading_dates": section7_context["trade_date_ny"].nunique(),
    "paired_directional_rows": len(section7_labels),
    "feature_registry_rows": len(section7_feature_registry),
    "frozen_candidate_policies": len(section7_candidate_registry),
    "advanced_to_section_8": int(section7_candidate_registry["decision"].eq("ADVANCE_TO_SECTION_8").sum()),
})
        """,
        ),
    ]

    nb.cells[appendix_index:appendix_index] = cells
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
    check = nbformat.read(NOTEBOOK, as_version=4)
    appendix_new_index = next(i for i, cell in enumerate(check.cells) if cell.id == APPENDIX_ID)
    if any(cell.id.startswith("7ctx") for cell in check.cells[appendix_new_index + 1 :]):
        raise RuntimeError("Section 7 cell was inserted after Appendix A")
    if not check.cells[appendix_new_index - 1].id == "7ctx2101":
        raise RuntimeError("Section 7 final cell is not immediately before Appendix A")
    print(f"Updated {NOTEBOOK}; backup={backup}; cells={len(check.cells)}")


if __name__ == "__main__":
    main()
