# Decision Memo: Branch A Section 8 Backtest Authorization

**To:** Project Owner (final authority on scope per PRD section 5)
**Date:** 2026-07-20
**Decision requested:** authorize or decline a Section 8 sequential research backtest of the True POI continuation-short family.

## Why this decision is due now

Every prerequisite the roadmap set for Section 8 is complete: the refined True POI definition is frozen, the Section 7 context research is done, the independent statistical branch has finished its full arc (Sections 7-12), and a verified chronological single-position backtester with declared cost scenarios now exists (`src/statistical_research/sequential_backtest.py`, adapted trivially to POI events). The only remaining blocker is the explicit authorization the governance requires.

## The evidence, stated honestly

For the leading policy `S7P02_NY_BEAR_CONT` (New York bearish continuation short, compressed approach, boundary entry, volatility-hybrid stop, 3R target, 240m hold):

- Event-level mean R: +0.102 Development, +0.122 Validation, +0.037 Final test.
- Median and 25th-percentile outcomes are -1R in every partition; the mean is runner-dependent.
- Final-test same-bar ambiguity is 17.13 percent, and the mean swings from -0.169R (conservative) to +0.464R (optimistic) across ambiguity treatments.
- Costs are not yet included; the Section 11 base assumption (2.6 ticks round trip) would reduce the mean further.
- Section 12 additionally established that the statistical opportunity gate does not rescue this family as a filter.

## Options

1. **Authorize a bounded Section 8 research backtest** of S7P02 (and only S7P02) under the existing backtester's conservative ambiguity treatment and declared cost scenarios. Expected outcome given the evidence: rejection, or narrow conditional survival. Either result closes PRD Phase 2 for Branch A with sequential-trade evidence instead of an open question. Estimated effort: one milestone.
2. **Decline for now** and redirect effort to Section 12B (opportunity-conditioned sizing and exit research, contract already frozen in `project_docs/section12b_research_contract.md`), which targets exactly the -1R-median failure mode and could improve the candidate before it is spent on a backtest.
3. **Decline permanently** and archive the family as researched-and-insufficient.

## Recommendation

Option 2 first, then Option 1: run Section 12B, then backtest whichever version of the family (raw or sizing-conditioned) its results justify. Rationale: the final-test event-level mean (+0.037R) is thinner than the base cost load, so a backtest today spends the candidate's one clean sequential test on a configuration the evidence already marks as marginal, while Section 12B directly addresses the known failure mode without touching any new data.

## Decision record

- [ ] Option 1 authorized
- [ ] Option 2 authorized (Section 12B first)
- [ ] Option 3 declined permanently
- Signature/date:

---

## Resolution addendum (2026-07-20)

Both evidence paths this memo weighed have now been executed through the merged record: the Section 8 sequential backtest ran (PR #16) and returned **SEQUENTIAL_REJECTED at base costs**, and the frozen Section 12B contract ran (PR #18) with **NO_ADVANCE on all three hypotheses**. Under the linkage clause in `project_docs/section12b_research_contract.md`, Option 3 therefore applies: **the S7P02 family is archived** with the complete evidence chain (`reports/statistical_research/summaries/section8_poi_sequential_backtest_summary.md`, `reports/statistical_research/summaries/section12b_opportunity_conditioning_summary.md`). Formal sign-off on this record remains with the project owner; the checkboxes above are left for that signature.
