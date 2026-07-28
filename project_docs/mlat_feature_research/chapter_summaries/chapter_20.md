# Chapter 20 - Autoencoders for Conditional Risk Factors and Asset Pricing

**Assigned source range:** PDF pages 639-662  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

An autoencoder learns a constrained representation by reconstructing its own input. A bottleneck, sparsity penalty, corrupted input, probabilistic latent distribution, or sequence architecture prevents trivial copying and can yield a lower-dimensional code useful for another task. Nonlinear autoencoders generalize PCA when their activations and architecture allow nonlinear mappings (PDF pp. 639-644).

The chapter's trading application combines a characteristic-conditioned network for time-varying factor loadings with an autoencoder-like factor branch. This flexible architecture learns latent risk premia and loadings jointly, but it is fundamentally a large cross-sectional equity design and carries important data-quality and survivorship limitations (PDF pp. 652-662).

## Concepts and definitions

- **Autoencoder:** an encoder maps input \(x\) to hidden representation \(h\); a decoder reconstructs \(x\) from \(h\) (PDF pp. 639-640).
- **Self-supervised learning:** the observed input supplies both predictor and reconstruction target (PDF p. 640).
- **Undercomplete autoencoder:** the code has fewer dimensions than the input, creating a bottleneck and lossy compression (PDF pp. 640-641).
- **Nonlinear PCA generalization:** without nonlinear activations an undercomplete autoencoder learns the same subspace as PCA; nonlinearity permits a wider class of encodings (PDF pp. 640-641).
- **Sparse autoencoder:** L1 or another penalty encourages only a subset of code units to activate (PDF p. 642).
- **Denoising autoencoder:** trains on corrupted input but reconstructs the original input, discouraging the identity function and learning structure robust to noise (PDF p. 642).
- **Seq2seq autoencoder:** recurrent encoder and decoder compress a sequence to a fixed-length representation and reconstruct a sequence (PDF pp. 642-643).
- **Variational autoencoder (VAE):** learns parameters of a latent probability distribution from which new samples can be generated (PDF pp. 643-644).
- **Conditional autoencoder:** conditions factor loadings on observable asset characteristics while learning latent factor premia from returns (PDF pp. 652-659).
- **Instrumented PCA (IPCA):** a linear latent-factor method that conditions factor loadings on asset characteristics; the chapter presents the conditional autoencoder as a nonlinear extension (PDF pp. 652-653).

## Formulas and notation

Only formulas confirmed from the PDF are used.

- Encoder:

  \[
  h=f(x)
  \]

- Decoder and identity solution:

  \[
  x=g(f(x))
  \]

- Constrained reconstruction objective:

  \[
  L\!\left(x,g(f(x))\right)
  \]

  The examples use mean squared reconstruction error (PDF p. 640).

- In the conditional-factor architecture, the output is the dot product of an \(N\times K\) factor-loading matrix and a \(K\times1\) factor-premium vector, producing \(N\times1\) predicted returns:

  \[
  \hat r_t=\beta_t f_t
  \]

  where \(\beta_t\) is a function of \(P\) asset characteristics and \(K\) is the latent-factor count (PDF p. 658).

- Amihud-style illiquidity example:

  \[
  \text{ILLIQ}_t
  =\operatorname{rolling\ mean}_{21}
  \left(\frac{|r_t|}{P_tV_t}\right)
  \]

  This is a faithful mathematical rendering of the code on PDF p. 656.

## Assumptions

- Constraints force the code to preserve salient structure rather than memorize an identity mapping (PDF pp. 640-642).
- Reconstruction quality is related to downstream usefulness; this must be tested rather than assumed.
- Nonlinear latent factors are sufficiently stable to generalize.
- The conditional model assumes a broad cross-section of assets whose common covariance can identify latent risk factors (PDF pp. 652-658).
- Observable asset characteristics legitimately condition time-varying factor loadings.
- Missing-value handling does not become an unintended signal; the chapter's sentinel value `-2` lies outside the normalized feature range and could encode missingness (PDF p. 657).

## Procedures described by the chapter

### Generic autoencoder workflow

1. Define the input representation and reconstruction target.
2. Select an encoder/decoder architecture suited to tabular, image, or sequence topology.
3. Constrain learning with a bottleneck, sparsity, noise corruption, or a probabilistic latent objective.
4. Train by reconstruction loss.
5. Inspect reconstruction and the learned embedding.
6. Freeze or transfer the encoder and measure value on a separate downstream task (PDF pp. 639-652).

### Conditional trading example

1. Source price/metadata for a large equity universe.
2. compute predictive characteristics spanning trend, liquidity, and risk.
3. Rank-normalize characteristics cross-sectionally to `[-1, 1]`; assign a missing sentinel.
4. Use a feedforward branch to map \(P\) characteristics to \(K\) loadings.
5. Use a factor branch to map the cross-section of lagged returns to \(K\) factor premia.
6. Form predicted returns by a dot product.
7. Cross-validate hidden width and factor count over annual folds.
8. Evaluate IC and portfolio spreads (PDF pp. 653-661).

## Feature and model examples

- Top characteristic categories: price trend, liquidity, and risk (PDF p. 655).
- Examples include 11-month momentum skipping the most recent month, Amihud illiquidity, idiosyncratic volatility, and rolling market beta (PDF pp. 655-656).
- The simplified dataset has about 3 million weekly observations, 16 characteristics, and about 3,800 securities (PDF pp. 656-658).
- The example explores \(K=2,\ldots,6\) latent factors and hidden widths 8, 16, or 32 (PDF pp. 658-660).
- Four or six factors with eight hidden units report overall validation IC around 0.02-0.03 (PDF p. 660).
- A selected four-factor/eight-unit model trained for 15 epochs shows roughly 10 bps bottom-top spread at shorter horizons before transaction costs (PDF pp. 660-661).

## Statistical, validation, and backtesting warnings

- An unconstrained high-capacity autoencoder can learn the identity function and produce a useless code (PDF pp. 640-642).
- Good reconstruction does not prove predictive value.
- Free `yfinance` data create adjustment-quality problems, survivorship bias, and restricted coverage (PDF pp. 653-654).
- The chapter uses a simplified subset of the original GKX data, characteristics, and architecture; its results are illustrative rather than a reproduction of the academic study (PDF p. 653).
- Rank normalization and missing-value sentinel handling are fitted transformations that must remain fold-local.
- The chapter notes that the original authors use conservative reporting lags to control data snooping; point-in-time availability is essential (PDF p. 657).
- The displayed portfolio analysis ignores transaction costs (PDF p. 660).
- Architecture, factor count, epoch, and data-frequency searches are related trials and require a declared selection rule.

## Implementation patterns worth preserving

- Separate encoder, decoder/recovery, downstream predictor, and evaluator.
- Persist encoder weights and the exact feature/channel schema.
- Verify reconstruction on held-out chronological data before evaluating downstream signal.
- Compare learned codes with PCA and with original features to quantify incremental value.
- Treat reconstruction error as its own potential state/anomaly feature; do not silently mix it with latent codes.
- Fit normalization, corruption/noise level, bottleneck width, and early stopping inside Development folds only.
- Make missingness explicit with masks instead of relying only on an out-of-range sentinel.

## Dated APIs and examples

- TensorFlow 2/Keras Functional API, Fashion-MNIST loading, `Dense`, `Conv2D`, `MaxPooling2D`, and older optimizer conventions need current-version verification (PDF pp. 644-652, 658-660).
- The example's `yfinance.Tickers(...).info` scraping pattern is unstable and irrelevant to the trusted GC source.
- NASDAQ symbol sourcing through `pandas_datareader.nasdaq_trader` and the 2020 metadata fields should not be treated as a current production interface.
- Alphalens is used for equity quantile spreads and does not encode Project 1's one-minute path, session, or execution constraints.

## Project 1 relevance

- The chapter's **conditional cross-sectional factor model is not directly identifiable from one GC stream**. Its \(N\)-asset covariance structure and characteristic ranks do not map cleanly to a single active futures series.
- Standard and sequence autoencoders are more relevant: a compact causal code or reconstruction error could summarize multivariate one-minute market state.
- Existing Project 1 features already compress volatility, trend, activity, and session state. Any learned embedding must demonstrate incremental Development-to-Validation information beyond the frozen feature set and anchor.
- Because validated evidence currently concerns expansion, a latent-state feature should first target opportunity/volatility expansion, not be interpreted as direction.

## One-minute GC adaptation

Potential bounded hypotheses:

1. A small undercomplete or denoising sequence autoencoder trained only on causal trailing GC channels yields a latent state that improves 60-/180-minute expansion ranking.
2. Reconstruction error identifies unusual market states associated with future range expansion.

Required adaptation:

- Build sequences ending at completed bar \(t\); never reconstruct using a centered window or a decoder target containing bars after \(t\).
- Use only predeclared channels available at \(t\), with fold-local scaling.
- Reset at missing minutes, contract/instrument/segment changes, and invalid tradability/roll boundaries.
- Fit the autoencoder on Development training folds; freeze it before transforming validation rows.
- Compare PCA, a linear autoencoder, nonlinear bottlenecks, and a simple no-embedding baseline.
- Evaluate latent dimensions and reconstruction error individually and jointly with the frozen ridge model using date-block statistics.
- Treat bottleneck width, noise level, sequence length, seed, and downstream model as governed trials.

## Unsuitable or deferred ideas

- Directly porting the \(N\)-equity conditional factor architecture to one GC instrument.
- Cross-sectional rank normalization across one-minute observations from different times as if they were contemporaneous assets.
- Using a decoder or bidirectional sequence encoder that accesses future bars.
- Selecting bottleneck dimensions by downstream Validation/Final-test performance across an open-ended grid.
- Treating low reconstruction error as alpha.
- VAEs or large generative models before a simple PCA/undercomplete benchmark shows stable incremental value.
- The chapter's free-equity data pipeline and gross long-short spreads.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Autoencoder definition and nonlinear feature extraction | 639-641 |
| Sparsity, denoising, seq2seq, and VAE variants | 642-644 |
| TensorFlow/Keras autoencoder examples | 644-652 |
| Conditional factor motivation and data caveats | 652-654 |
| Characteristics and preprocessing | 655-657 |
| Conditional architecture and model implementation | 657-660 |
| Predictive/economic evaluation and next steps | 660-662 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 640 and 658.
- **Confidence: high** for autoencoder definitions, architecture, feature examples, data caveats, and reported validation results.
- **Confidence: high** for \(h=f(x)\), \(x=g(f(x))\), and \(L(x,g(f(x)))\); PDF p. 640 was visually inspected.
- **Confidence: high** for the conditional model's dimensions and dot-product relationship; Figure 20.7 on PDF p. 658 was visually inspected.
- The exact original GKX specification is intentionally simplified by the book; this summary does not substitute for the cited paper.
- The chapter does not establish causal interpretation of the learned latent factors.
- No autoencoder feature is approved for one-minute GC without a separate predeclared experiment.
