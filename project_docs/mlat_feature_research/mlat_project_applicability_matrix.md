# MLAT Project Applicability Matrix

| Idea | Source (chapter:PDF) | Required data | Existing equivalent | Adaptation | Target | Classification | Reason |
|---|---|---|---|---|---|---|---|
| Lagged returns | 4:130-132 | OHLCV | existing return ladder | None | direction | REDUNDANT WITH EXISTING FEATURES | Already represented at frozen horizons. |
| Bollinger location | 4:131-133; App:740-742 | close | range/trend state | Use causal 20-bar z-score; no trading thresholds | direction | APPLICABLE AFTER MODIFICATION | Distinct continuous hypothesis; empirical overlap required. |
| Bollinger bandwidth | App:740-742 | close | compression features | Normalize by trailing mean | expansion | APPLICABLE AFTER MODIFICATION | Direct squeeze concept but likely redundant. |
| RSI | 4:132 | close | momentum/persistence | Declare Cutler rolling convention | direction / state | APPLICABLE AFTER MODIFICATION | Avoid arbitrary 30/70 rule. |
| Kalman filter | 4:133-136 | close/returns | trend filters | Segmented online filter and chronological parameter fit | state | LATER RESEARCH PHASE | Noise parameters and overlap are unresolved. |
| Wavelet denoising | 4:137-140 | returns | none | Would require one-sided online transform | state | REJECTED DUE TO LEAKAGE | Book demonstration uses full-window decomposition/reconstruction. |
| Mutual information | 6:192 | features/labels | rank IC | Chronological resampling and multiple-test control | screening | LATER RESEARCH PHASE | Estimator cost/bias adds little to the first batch. |
| Purging and embargo | 6:199-200 | label spans | date blocks | Use for any trained multivariate model | validation | DIRECTLY APPLICABLE | Mandatory if modelling is authorized. |
| Linear/regularized model | 7:203-248 | frozen features | ridge benchmark | Chronological nested fit | direction/expansion | LATER RESEARCH PHASE | Requires a surviving shortlist. |
| Event-driven backtest | 8:249-279 | signals/orders/costs | sequential backtester | Use only after feature and signal gates | execution | LATER RESEARCH PHASE | No policy is authorized in v1. |
| AR/variance dependence | 9:280-296 | returns | autocorrelation/sign change | Use variance-ratio adaptation | direction/state | APPLICABLE AFTER MODIFICATION | Distinct formula but high overlap risk. |
| GARCH | 9:297-301 | segmented returns | ATR/RV and old exploratory cells | GC-only segmented fit and diagnostics | volatility/risk | LATER RESEARCH PHASE | Audit separately; not a frozen v1 predictor. |
| Cointegration/pairs | 9-10:301-349 | multi-asset prices | none | Would require a defensible instrument basket | portfolio | REQUIRES DIFFERENT DATA | Single GC series cannot supply a pair. |
| Tree/boosting models | 11-12:350-428 | frozen features/labels | none | Chronological tuning after linear gate | model | LATER RESEARCH PHASE | Premature complexity. |
| PCA/clustering | 13:429-460 | feature matrix/universe | hierarchical feature clustering | Use Development-only clustering for redundancy | risk/redundancy | APPLICABLE AFTER MODIFICATION | No new PCA factor in v1. |
| Sentiment/topics/embeddings | 14-16:461-532 | point-in-time text | none | Obtain governed publication data | alternative data | REQUIRES DIFFERENT DATA | Unavailable in OHLCV. |
| Deep/CNN/RNN/autoencoder | 17-20:533-662 | large model-ready corpus | none | Establish simpler evidence first | model | LATER RESEARCH PHASE | Not justified by current results. |
| Amihud illiquidity | 20:656; App:752 | close/volume | liquidity proxies | Within-GC 60-minute notional proxy | risk/expansion | APPLICABLE AFTER MODIFICATION | Distinct formula with high overlap risk. |
| GAN synthetic data | 21:663-690 | training sequences | none | Fidelity/tail validation | data augmentation | REJECTED DUE TO GOVERNANCE | Could manufacture or erase rare risk behavior. |
| Reinforcement learning | 22:691-723 | validated environment/reward | none | Requires approved sequential policy | execution | REJECTED DUE TO GOVERNANCE | No signal or simulator authorization. |
| Chaikin A/D | App:752-753 | OHLCV | CLV/signed-volume features | Use rolling normalized money flow, reset boundaries | direction/state | APPLICABLE AFTER MODIFICATION | Cumulative raw A/D has reset/scale problems. |
| ATR/NATR | App:754-755 | OHLC | atr_20 anchor | None | volatility | REDUNDANT WITH EXISTING FEATURES | Use frozen anchor rather than add another ATR. |
| Quotes/trades/order book/MBO | 2:59-94 | quotes/trades/depth/MBO | none | Acquire different Databento schemas | execution/microstructure | REQUIRES DIFFERENT DATA | Not present in one-minute OHLCV. |
| Macro/fundamental factors | 2-4:59-152 | point-in-time macro/fundamentals | none | Build release/vintage-aware sources | direction/risk | REQUIRES DIFFERENT DATA | Unavailable and often equity-specific. |
| Satellite/imagery | 3/18:95-114;569-606 | images | none | Define economically linked imagery | alternative data | IMPRACTICAL WITH CURRENT OHLCV | No relevant image source. |
