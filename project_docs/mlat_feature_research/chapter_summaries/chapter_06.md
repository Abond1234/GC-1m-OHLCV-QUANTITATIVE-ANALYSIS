# Chapter 6 - The Machine Learning Process

**Assigned source range:** PDF pages 179-202  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; the frozen MLAT batch remains a pre-model feature experiment.

## Main argument

Reliable machine learning is a controlled process rather than a model choice.
The task, target, success metric, data, features, hypothesis space, fitting
procedure, and out-of-sample test must be mutually consistent. Complexity
creates variance and makes noise easier to learn, while overly restrictive
models create bias. Generalization therefore has to be estimated on genuinely
unseen data with a split design that respects the dependence and chronology of
financial observations (PDF pp. 179-202).

For finance, ordinary IID cross-validation is usually invalid. Serial
correlation, heteroskedasticity, overlapping labels, point-in-time availability,
and repeated tuning can leak information or bias error estimates. Walk-forward
splits, purging, embargo, a final untouched sample, and an explicit
multiple-testing policy are central controls (PDF pp. 195-202).

## Concepts and definitions

- **Hypothesis space and inductive bias:** a learner can search only the
  functions its representation permits; no universal algorithm works best for
  every task (PDF pp. 180-181).
- **Supervised learning:** estimate an input/output relationship from features
  \(X\) and observed labels \(y\) for inference or prediction
  (PDF pp. 181-182).
- **Unsupervised learning:** find structure, clusters, or lower-dimensional
  representations without an outcome label (PDF pp. 182-184).
- **Reinforcement learning:** an agent learns sequential actions from delayed
  rewards while balancing exploration and exploitation (PDF p. 184).
- **Regression, classification, and ranking:** alternative problem framings
  for continuous, categorical, and ordered outcomes (PDF pp. 185-186).
- **Inference versus prediction:** parameter/relationship claims and
  out-of-sample forecasting have different objectives; association alone is
  not causality (PDF pp. 186-187).
- **Mutual information:** dependence measure that can capture nonlinear
  relationships; estimator and repeated-screen bias still matter
  (PDF pp. 191-192).
- **Bias-variance trade-off:** underfit models have systematic error; overfit
  models have sample-sensitive error and unstable out-of-sample predictions
  (PDF pp. 192-195).
- **Learning/validation curves:** diagnose whether error reflects model
  capacity, variance, or insufficient data (PDF pp. 193-195, 201-202).
- **Cross-validation:** estimate generalization by fitting and scoring on
  separate samples; ordinary random CV assumes IID observations
  (PDF pp. 195-200).
- **Purging and embargo:** remove training observations whose information
  intervals overlap or immediately follow a validation period
  (PDF p. 200).

## Formulas and notation

Rendered PDF p. 192 defines continuous mutual information as

\[
I(X,Y)=\int_{\mathcal{Y}}\int_{\mathcal{X}}
p(x,y)\log\!\left(\frac{p(x,y)}{p(x)p(y)}\right)\,dx\,dy.
\]

MI is non-negative dependence information, not a signed economic effect. A
large estimate does not reveal direction, causality, stability, or incremental
value after existing predictors.

The chapter also presents regression losses, confusion-matrix metrics, ROC/AUC,
precision/recall, and \(F_1\) (PDF pp. 187-190). The appropriate metric must
match the target and decision cost. Project 1 uses daily Spearman rank IC,
Development-fitted buckets, effect sizes, and tick economics for this
univariate experiment; it does not substitute accuracy or AUC for the declared
continuous feature screens.

Entropy is introduced as the information content underlying MI (PDF p. 192).
The frozen `return_sign_entropy_60` is a Project-original causal adaptation of
that concept; its three-state rolling formula is defined in the MLAT formula
registry and is not attributed verbatim to this chapter.

## Assumptions

- Training and validation inputs are point-in-time and use identical,
  Development-fitted transformations.
- Validation labels and feature values cannot influence fitting, threshold
  choice, imputation, feature selection, or hyperparameters.
- Random CV is defensible only under IID sampling; financial time series
  usually violate this through dependence and changing variance
  (PDF pp. 196, 199).
- Model-selection scores are optimistic after many alternatives are tried;
  an independent sample is required for an unbiased final estimate
  (PDF pp. 196-197).
- Labels have information intervals. Overlap must be removed or accounted for,
  not treated as independent evidence (PDF p. 200).
- Unsupervised structure is not automatically useful; it must improve a
  downstream, out-of-sample objective (PDF pp. 182-184).
- Causal conclusions require stronger experimental or quasi-experimental
  assumptions than predictive association (PDF pp. 186-187).

## Procedures described by the chapter

1. Frame the problem, target, metric, and success criterion.
2. Source, clean, validate, and store the data reproducibly.
3. Explore distributions and relationships; engineer domain-grounded features.
4. Select algorithms whose assumptions and capacity match the task and sample.
5. Train, validate, tune, and diagnose bias/variance without information
   leakage.
6. Evaluate once on a genuinely independent sample and use the result for the
   original decision problem (PDF pp. 184-202).

The Project 1 adaptation freezes hypotheses and evaluation rules before
outcomes, computes features on full causal history, uses chronological
Development and Validation, treats dates rather than minute rows as the main
evidence units, and excludes the already-exposed historical Final test.

## Feature and model examples

- K-nearest-neighbor regression/classification as a workflow illustration
  (PDF pp. 185-190).
- Linear/rank association, mutual information, and domain transformations for
  feature assessment (PDF pp. 191-192).
- Polynomial examples and learning curves illustrating bias and variance
  (PDF pp. 193-195).
- Random train/test, K-fold, leave-one-out, leave-P-out, ShuffleSplit, and
  TimeSeriesSplit schemes (PDF pp. 197-200).
- Purged, embargoed, and combinatorial financial CV (PDF p. 200).
- GridSearchCV/Pipeline and Yellowbrick validation/learning curves
  (PDF pp. 200-202).

These are methodology examples, not authorization for a broad model or
hyperparameter search in MLAT v1.

## Statistical, validation, and backtesting warnings

- A complex model can fit training noise while appearing excellent in sample
  (PDF pp. 181-182, 192-195).
- Association does not establish causality, and omitted drivers can change
  predictive relationships (PDF pp. 186-187).
- Metric choice embeds asymmetric economic costs; a threshold-independent AUC
  does not solve a trading decision (PDF pp. 187-190).
- Feature exploration itself creates researcher degrees of freedom and must be
  governed before outcome inspection.
- Randomized K-fold/ShuffleSplit mixes future and past and violates realistic
  trading chronology (PDF pp. 196-200).
- Overlapping forward returns can leak validation information into training;
  purging/embargo or equivalent interval controls are required
  (PDF p. 200).
- Repeated CV/hyperparameter selection creates multiple-testing bias; the best
  validation result is not an unbiased generalization estimate
  (PDF pp. 196-197).
- Point-in-time mistakes can make hindsight data appear available earlier than
  it really was (PDF p. 200).
- A final holdout that has already been inspected is not restored by starting
  a new notebook.
- Model importance or nonlinear dependence is not proof of economic value.

## Implementation patterns worth preserving

- Encode the experiment contract, random seeds, partitions, target columns,
  feature registry, and artifact fingerprints in machine-readable outputs.
- Fit every learned transform only on Development and apply it unchanged.
- Use chronological split objects and explicit label-information intervals.
- Keep preprocessing inside the fitted pipeline so validation data cannot leak
  into scaling, imputation, or selection.
- Record every alternative tried; correct broad screens for multiplicity.
- Diagnose training/validation divergence and year/session instability before
  escalating model complexity.
- Preserve a simple benchmark and an independent future confirmation period.

## Dated APIs and examples

The chapter's scikit-learn concepts remain recognizable, but exact defaults and
signatures for `KNeighborsRegressor`, scoring strings, `TimeSeriesSplit`,
`GridSearchCV`, and `Pipeline` should be checked against the installed version.
Yellowbrick and the referenced `timeseriescv` package are not project
dependencies. The chapter contains legacy cross-references that place
autoencoders/RL one chapter earlier than this edition's actual Chapter 20/22
locations. Those editorial artifacts do not affect the governance principles.

## Project 1 relevance

This is one of the most directly relevant chapters:

- it supports the frozen, bounded hypothesis batch and model-authorization
  gate;
- it motivates nonlinear-dependence research but does not override the
  predeclared daily rank-IC contract;
- its chronology, leakage, multiple-testing, and point-in-time warnings govern
  every MLAT evaluation;
- it supports the decision to report direction, expansion, volatility, and
  path risk separately;
- it confirms that no nonlinear model should run unless a small,
  non-redundant feature shortlist first survives Validation.

## One-minute GC adaptation

- Never randomize minute observations across Development and Validation.
- Keep all six horizons on controlled complete samples and thin observations
  by horizon as a dependence sensitivity check.
- Use New York dates for block resampling and daily IC aggregation.
- Fit session quantiles on Development and freeze them before Validation.
- Apply Benjamini-Hochberg control within declared screen families.
- Enforce t-close feature availability and t+1-open theoretical entry.
- Exclude MGC and historical Final-test outcomes from discovery.
- Require incremental information beyond `atr_20` and the frozen expansion
  anchors before advancing volatility/risk features.
- Do not equate a large row count with breadth or statistical independence.

## Unsuitable or deferred ideas

- Random K-fold, leave-one-out, and ShuffleSplit are rejected for this
  chronological minute experiment.
- An unrestricted grid search is rejected by governance.
- KNN, tree ensembles, neural networks, and reinforcement learning are deferred
  until a small feature shortlist earns authorization.
- Mutual information is research-only in v1; estimator choice and repeated
  testing would need a separate frozen protocol.
- Classification thresholds and trading rules are out of scope.
- Cross-sectional clustering/risk-factor examples require a different
  instrument universe.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Learning categories, hypothesis spaces, and task matching | 179-184 |
| Reproducible ML workflow and problem framing | 184-186 |
| Inference, causality, regression, and classification metrics | 186-190 |
| Data/feature workflow and mutual information | 191-192 |
| Bias, variance, overfitting, and learning curves | 192-195 |
| Model selection, CV, multiple testing, and split implementations | 195-199 |
| Financial time-series CV, purging, embargo, and tuning | 199-202 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 192, 196, 198, and 200.
- **Confidence: high** for the workflow, metric definitions, bias/variance
  discussion, financial CV warnings, and purging/embargo guidance.
- **Confidence: high** for the MI equation on PDF p. 192, the training/test
  design on PDF p. 196, the K-fold illustration on PDF p. 198, and the
  purging/embargo definitions on PDF p. 200; these pages were visually
  inspected.
- **Confidence: medium** for a contradictory sentence on rendered p. 196 that
  says temporal splits avoid lookahead by “including” future information in
  training. The surrounding warning clearly requires excluding such
  information, so the word is recorded as an apparent editorial error rather
  than implemented.
- The chapter does not specify a complete purging interval formula for this
  project's six overlapping intraday horizons.
- It does not justify treating a high MI estimate or feature-importance score
  as stable GC economic information.
