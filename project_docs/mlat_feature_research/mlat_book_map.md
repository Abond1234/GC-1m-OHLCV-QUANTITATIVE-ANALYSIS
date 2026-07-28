# MLAT Book Map

## Source

- Stefan Jansen, *Machine Learning for Algorithmic Trading*, second edition.
- Supplied file: `Machine Learning for Algorithmic Trading (2nd Edition).pdf`.
- Physical PDF length: 858 pages.
- Source file size: 27,293,878 bytes; SHA-256
  `4439a7be25210efb1173f30bcb75cb67743acff78de880578dc8a6de351aa39a`.
- Page references below are physical PDF pages, not the book's printed page
  numbers.
- Full text extraction covered PDF pages 1-858. Formula, figure, table, and
  workflow pages were also visually rendered where layout mattered.

## Coverage map

| Unit | Title | PDF pages | Principal topics | GC one-minute relevance | Priority |
|---|---|---:|---|---|---|
| Front matter | Title, publication, contributors, contents | 1-29 | Edition scope, code repository, four-part structure | Provenance and navigation only | BACKGROUND RELEVANCE |
| Preface | Preface | 30-41 | End-to-end ML-for-trading workflow, revised libraries, chapter guide | Frames evidence-first research and reproducibility | HIGH DIRECT RELEVANCE |
| Part 1 | Data, alpha factors, and portfolios (Chapters 1-5) | 42-178 | Market and alternative data, feature research, portfolio and performance evaluation | Feature/data foundations transfer; cross-sectional portfolio methods are later-stage | HIGH DIRECT RELEVANCE |
| 1 | Machine Learning for Trading - From Idea to Execution | 42-58 | ML use cases, alpha discovery, workflow, infrastructure | Direct governance for hypothesis-to-validation sequencing | HIGH DIRECT RELEVANCE |
| 2 | Market and Fundamental Data - Sources and Techniques | 59-94 | Market structure, OHLCV, APIs, point-in-time data, storage | Direct data-quality and timestamp contract; equity fundamentals do not transfer | HIGH DIRECT RELEVANCE |
| 3 | Alternative Data for Finance - Categories and Use Cases | 95-114 | Alternative-data sourcing, evaluation, legal and quality risks | Current OHLCV batch cannot use these inputs; useful for deferral decisions | BACKGROUND RELEVANCE |
| 4 | Financial Feature Engineering - How to Research Alpha Factors | 115-152 | factor rationale, lagging, technical indicators, Kalman and wavelet denoising, evaluation | Primary source for bounded feature hypotheses and overlap controls | HIGH DIRECT RELEVANCE |
| 5 | Portfolio Optimization and Performance Evaluation | 153-178 | covariance, efficient frontier, Black-Litterman, Kelly, performance metrics | Risk/performance governance; portfolio methods are later-stage | MODERATE RELEVANCE |
| Part 2 | ML for trading - Fundamentals (Chapters 6-13) | 179-460 | Supervised and unsupervised learning in an end-to-end research workflow | Validation and simple benchmark methods transfer after GC-specific adaptation | HIGH DIRECT RELEVANCE |
| 6 | The Machine Learning Process | 179-202 | labels, mutual information, bias-variance, CV, purging and embargo | Direct validation and leakage governance | HIGH DIRECT RELEVANCE |
| 7 | Linear Models - From Risk Factors to Return Forecasts | 203-248 | OLS, shrinkage, logistic regression, factor modelling | Suitable only after a univariate shortlist earns the multivariate gate | MODERATE RELEVANCE |
| 8 | The ML4T Workflow - From Model to Strategy Backtesting | 249-279 | backtest design, timing, costs, event-driven systems, Zipline | Direct point-in-time and execution sequencing; strategy work is not authorized here | HIGH DIRECT RELEVANCE |
| 9 | Time-Series Models for Volatility Forecasts and Statistical Arbitrage | 280-319 | stationarity, ARIMA, ARCH/GARCH, diagnostics, cointegration | Primary source for volatility audit and causal forecast requirements | HIGH DIRECT RELEVANCE |
| 10 | Bayesian ML - Dynamic Sharpe Ratios and Pairs Trading | 320-349 | Bayesian updating, MCMC, probabilistic programming, dynamic models | State uncertainty is relevant; pairs/cross-sectional examples are not directly transferable | MODERATE RELEVANCE |
| 11 | Random Forests - A Long-Short Strategy for Japanese Stocks | 350-386 | trees, bagging, random forests, feature importance | Later nonlinear benchmark only; cross-sectional stock strategy is out of scope | MODERATE RELEVANCE |
| 12 | Boosting Your Trading Strategy | 387-428 | AdaBoost, gradient boosting, XGBoost, LightGBM, SHAP | Later model phase only after simpler evidence; no broad tuning in v1 | MODERATE RELEVANCE |
| 13 | Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning | 429-460 | PCA, clustering, manifold learning, risk factors | Regime/redundancy concepts are useful; portfolio allocation is later-stage | MODERATE RELEVANCE |
| Part 3 | Natural language processing (Chapters 14-16) | 461-532 | Sentiment, topic models, and embeddings from financial text | Requires point-in-time text data unavailable to the OHLCV experiment | CURRENTLY OUT OF SCOPE |
| 14 | Text Data for Trading - Sentiment Analysis | 461-485 | NLP pipeline, sentiment, text classification | Requires unavailable text data | CURRENTLY OUT OF SCOPE |
| 15 | Topic Modeling - Summarizing Financial News | 486-505 | LDA, topic quality, financial news | Requires unavailable text/news data | CURRENTLY OUT OF SCOPE |
| 16 | Word Embeddings for Earnings Calls and SEC Filings | 506-532 | word2vec, embeddings, SEC text | Equity-specific text data; not applicable to GC OHLCV v1 | CURRENTLY OUT OF SCOPE |
| Part 4 | Deep and reinforcement learning (Chapters 17-23) | 533-734 | Neural networks, generative models, and sequential agents | Covered for completeness; complexity is unauthorized before simpler GC evidence | BACKGROUND RELEVANCE |
| 17 | Deep Learning for Trading | 533-568 | neural-network foundations, TensorFlow/PyTorch workflows | Complexity benchmark for later work; not authorized for v1 | BACKGROUND RELEVANCE |
| 18 | CNNs for Financial Time Series and Satellite Images | 569-606 | convolution, image transfer learning, time-series grids | Satellite input unavailable; CNN feature images are unjustified in v1 | CURRENTLY OUT OF SCOPE |
| 19 | RNNs for Multivariate Time Series and Sentiment Analysis | 607-638 | RNN/LSTM/GRU sequence models | Later sequence research only after simpler GC signal evidence | BACKGROUND RELEVANCE |
| 20 | Autoencoders for Conditional Risk Factors and Asset Pricing | 639-662 | nonlinear compression, conditional factors, Amihud illiquidity | Liquidity feature adaptation is relevant; equity factor model is not | MODERATE RELEVANCE |
| 21 | Generative Adversarial Networks for Synthetic Time-Series Data | 663-690 | GANs, TimeGAN, fidelity evaluation | Synthetic training data is not authorized and could distort tails | CURRENTLY OUT OF SCOPE |
| 22 | Deep Reinforcement Learning - Building a Trading Agent | 691-723 | MDPs, Q-learning, policy learning, market environment | Requires an approved sequential policy; explicitly premature | CURRENTLY OUT OF SCOPE |
| 23 | Conclusions and Next Steps | 724-734 | data quality, domain expertise, diagnostics, overfitting | Direct handoff and research-governance guidance | HIGH DIRECT RELEVANCE |
| Appendix | Alpha Factor Library | 735-764 | moving averages, Bollinger, momentum, volume, ATR, transforms | Direct formula/hypothesis source, subject to overlap and causal adaptation | HIGH DIRECT RELEVANCE |
| References | References | 765-779 | Bibliographic trail | Traceability only | BACKGROUND RELEVANCE |
| Index | Index | 780-858 | Topic locator | Coverage verification and ambiguity resolution | BACKGROUND RELEVANCE |

## Project interpretation

The highest-value material for this experiment is the combination of Chapter 4
and the Appendix (feature hypotheses), Chapter 6 (time-aware validation),
Chapter 8 (signal/execution sequencing), and Chapter 9 (volatility modelling).
The deep-learning, alternative-data, text, portfolio, pairs, and reinforcement
learning chapters are covered but do not authorize their methods for a
single-instrument OHLCV feature screen.
