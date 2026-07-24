# MLAT Concept Registry

The CSV beside this file is authoritative. Recommendations are project
decisions, not claims that the book endorses the exact GC adaptation.

| ID | Concept | Source | Target | Applicability | Leakage | Recommendation |
|---|---|---|---|---|---|---|
| MLAT-C001 | Research-to-execution workflow | Ch. 1, PDF 42-58 | governance | DIRECT | MEDIUM | ADOPT |
| MLAT-C002 | Point-in-time market data | Ch. 2, PDF 59-94 | execution | DIRECT | HIGH | ADOPT |
| MLAT-C003 | Alternative-data quality screen | Ch. 3, PDF 95-114 | later research | LATER | HIGH | DEFER_DIFFERENT_DATA |
| MLAT-C004 | Lagged-return features | Ch. 4, PDF 130-132 | direction | DIRECT | MEDIUM | REJECT_REDUNDANT |
| MLAT-C005 | Bollinger standardized location and bandwidth | Ch. 4 / Appendix, PDF 131-133; 740-742 | direction and expansion | ADAPTED | LOW | IMPLEMENT_AND_TEST_OVERLAP |
| MLAT-C006 | Relative strength index | Ch. 4, PDF 132 | direction / state | ADAPTED | LOW | IMPLEMENT_CUTLER_VARIANT |
| MLAT-C007 | Kalman filtering | Ch. 4, PDF 133-136 | filtered state | ADAPTED | HIGH | DEFER |
| MLAT-C008 | Wavelet denoising | Ch. 4, PDF 137-140 | filtered state | REJECTED | HIGH | REJECT_LEAKAGE_UNLESS_CAUSAL_REDESIGN |
| MLAT-C009 | Information coefficient and factor quantiles | Ch. 4, PDF 141-150 | all feature targets | DIRECT | MEDIUM | ADOPT |
| MLAT-C010 | Risk-adjusted performance metrics | Ch. 5, PDF 169-178 | strategy evaluation | LATER | MEDIUM | DEFER_UNTIL_POLICY |
| MLAT-C011 | Mutual information | Ch. 6, PDF 192 | feature screening | LATER | MEDIUM | RESEARCH_ONLY |
| MLAT-C012 | Bias-variance trade-off | Ch. 6, PDF 192-195 | model selection | DIRECT | HIGH | ADOPT |
| MLAT-C013 | Purging and embargo | Ch. 6, PDF 199-200 | model validation | DIRECT | HIGH | ADOPT_WHEN_MODELLING |
| MLAT-C014 | Regularized linear benchmark | Ch. 7, PDF 214-237 | multivariate benchmark | LATER | HIGH | GATED |
| MLAT-C015 | Event-driven backtesting | Ch. 8, PDF 255-273 | execution | LATER | HIGH | DEFER_UNTIL_SIGNAL |
| MLAT-C016 | Stationarity and differencing | Ch. 9, PDF 280-290 | time-series modelling | DIRECT | MEDIUM | ADOPT |
| MLAT-C017 | AR and variance-ratio state | Ch. 9, PDF 290-296 | direction / regime | ADAPTED | LOW | IMPLEMENT_VARIANCE_RATIO_AND_TEST_OVERLAP |
| MLAT-C018 | ARCH/GARCH conditional variance | Ch. 9, PDF 297-301 | volatility / risk state | ADAPTED | HIGH | AUDIT_SEPARATELY |
| MLAT-C019 | Bayesian state uncertainty | Ch. 10, PDF 320-349 | risk state | LATER | HIGH | DEFER |
| MLAT-C020 | Tree feature importance | Ch. 11, PDF 366-376 | interpretation | LATER | MEDIUM | DO_NOT_USE_AS_EDGE_PROOF |
| MLAT-C021 | Gradient boosting | Ch. 12, PDF 387-428 | nonlinear benchmark | LATER | HIGH | DEFER |
| MLAT-C022 | PCA and feature clustering | Ch. 13, PDF 429-460 | redundancy / regime | ADAPTED | MEDIUM | USE_CLUSTERING_NOT_NEW_PCA_FEATURES |
| MLAT-C023 | Sentiment and topic features | Ch. 14-16, PDF 461-532 | direction / risk state | DIFFERENT_DATA | HIGH | DEFER_DIFFERENT_DATA |
| MLAT-C024 | Deep sequence and representation models | Ch. 17-21, PDF 533-690 | later modelling | LATER | HIGH | DEFER |
| MLAT-C025 | Amihud illiquidity | Ch. 20 / Appendix, PDF 656; 752 | expansion / risk state | ADAPTED | LOW | IMPLEMENT_AND_TEST_OVERLAP |
| MLAT-C026 | Synthetic time-series fidelity | Ch. 21, PDF 663-690 | data augmentation | LATER | HIGH | DEFER |
| MLAT-C027 | Reinforcement-learning trading environment | Ch. 22, PDF 691-723 | execution policy | REJECTED_GOVERNANCE | HIGH | REJECT_FOR_V1 |
| MLAT-C028 | Chaikin money flow adaptation | Ch. Appendix, PDF 752-753 | direction / activity | ADAPTED | LOW | IMPLEMENT_AND_TEST_OVERLAP |
| MLAT-C029 | Average true range | Ch. Appendix, PDF 754-755 | expansion / risk state | DIRECT | LOW | USE_EXISTING_ANCHOR |
| MLAT-C030 | Backtest-overfitting warning | Ch. 23, PDF 724-734 | governance | DIRECT | HIGH | ADOPT |
