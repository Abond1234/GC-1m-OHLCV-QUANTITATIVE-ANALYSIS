# Frozen MLAT Feature Batch v1

Frozen before loading MLAT outcomes. Hard vetoes are future/entry-bar use,
unavailable data, invalid boundary behavior, MGC dependence, Final-test
dependence, or an exact existing formula.

Each score is 1 (weak) to 5 (strong). For redundancy, leakage, compute, and
overfitting, 5 means lower risk. The selection floor is 44/60 with no
hard veto. Scores prioritize a coherent batch rather than optimizing a
retrospective result.

| Feature | Type | Book | Rationale | Novelty | Causal | Data | Interpret | Target | Redundancy | Leakage | Numeric | Compute | Overfit | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bollinger_zscore_20` | MLAT-ADAPTED | 5 | 4 | 3 | 5 | 5 | 5 | 4 | 3 | 5 | 5 | 5 | 4 | 53 |
| `bollinger_bandwidth_20` | MLAT-DIRECT | 5 | 4 | 3 | 5 | 5 | 5 | 5 | 2 | 5 | 5 | 5 | 4 | 53 |
| `cutler_rsi_14` | MLAT-ADAPTED | 4 | 3 | 3 | 5 | 5 | 5 | 3 | 3 | 5 | 5 | 5 | 4 | 50 |
| `chaikin_money_flow_20` | MLAT-ADAPTED | 5 | 4 | 4 | 5 | 5 | 5 | 4 | 2 | 5 | 4 | 5 | 4 | 52 |
| `amihud_illiquidity_60` | MLAT-ADAPTED | 4 | 4 | 4 | 5 | 5 | 4 | 5 | 2 | 5 | 4 | 5 | 4 | 51 |
| `parkinson_volatility_30` | PROJECT-ORIGINAL-EXTENSION | 3 | 5 | 5 | 5 | 5 | 5 | 5 | 2 | 5 | 4 | 5 | 4 | 53 |
| `rogers_satchell_volatility_30` | PROJECT-ORIGINAL-EXTENSION | 3 | 5 | 5 | 5 | 5 | 4 | 5 | 2 | 5 | 4 | 5 | 4 | 52 |
| `realized_semivariance_balance_60` | PROJECT-ORIGINAL-EXTENSION | 3 | 5 | 5 | 5 | 5 | 5 | 4 | 3 | 5 | 5 | 5 | 4 | 54 |
| `bipower_jump_ratio_60` | PROJECT-ORIGINAL-EXTENSION | 3 | 5 | 5 | 5 | 5 | 4 | 5 | 3 | 5 | 4 | 4 | 4 | 52 |
| `variance_ratio_60_5` | MLAT-ADAPTED | 4 | 5 | 4 | 5 | 5 | 5 | 4 | 2 | 5 | 4 | 5 | 4 | 52 |
| `return_sign_entropy_60` | PROJECT-ORIGINAL-EXTENSION | 3 | 4 | 5 | 5 | 5 | 5 | 4 | 2 | 5 | 5 | 5 | 4 | 52 |
| `volatility_of_volatility_60` | PROJECT-ORIGINAL-EXTENSION | 3 | 5 | 5 | 5 | 5 | 5 | 5 | 3 | 5 | 4 | 5 | 4 | 54 |

## Frozen membership

1. `bollinger_zscore_20`
2. `bollinger_bandwidth_20`
3. `cutler_rsi_14`
4. `chaikin_money_flow_20`
5. `amihud_illiquidity_60`
6. `parkinson_volatility_30`
7. `rogers_satchell_volatility_30`
8. `realized_semivariance_balance_60`
9. `bipower_jump_ratio_60`
10. `variance_ratio_60_5`
11. `return_sign_entropy_60`
12. `volatility_of_volatility_60`

## Explicit exclusions

- GARCH is an audit-only method, not a v1 matrix feature. The prior
  implementation fails GC isolation and boundary governance.
- Kalman/KAMA are deferred because their fitted/recursive state and overlap
  are not justified for the first batch.
- Wavelet denoising is rejected until a genuinely one-sided implementation
  is defined.
- PPO/MACD, NATR, Williams %R, raw AR(1), OBV, and clock features are
  rejected as redundant or reset-ambiguous.
- Cross-sectional, text, quote/trade/order-book, macro, image, deep, GAN,
  and reinforcement-learning methods require a different phase or data.

No membership or parameter may change after Validation results are viewed.
