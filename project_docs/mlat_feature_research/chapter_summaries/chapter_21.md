# Chapter 21 - Generative Adversarial Networks for Synthetic Time-Series Data

**Assigned source range:** PDF pages 663-690  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature or synthetic-data use is approved by this note.

## Main argument

A generative adversarial network learns to create samples through competition between a generator and a discriminator. The generator maps random input to synthetic observations; the discriminator attempts to distinguish them from real data. For time series, matching the marginal distribution is insufficient: a useful generator must also learn temporal transitions and cross-feature dependence (PDF pp. 663-668, 673-676).

TimeGAN addresses this by combining an autoencoder, an adversarial network, and a supervised stepwise loss in a jointly learned latent space. The chapter demonstrates a TensorFlow 2 implementation and three evaluation ideas: diversity, fidelity, and usefulness (PDF pp. 673-690). The example supports further research, not the claim that synthetic financial paths are valid substitutes for unseen market regimes.

## Concepts and definitions

- **Discriminative model:** estimates outcomes conditional on inputs, \(p(y\mid X)\) (PDF p. 664).
- **Generative model:** learns a joint/data-generating distribution, represented in the chapter as \(p(y,X)\) or an estimate \(p_{\text{model}}\) of \(p_{\text{data}}\) (PDF p. 664).
- **Generator:** converts random noise into synthetic samples (PDF pp. 663-665).
- **Discriminator:** predicts whether a sample is real or generated (PDF pp. 663-665).
- **Adversarial training:** optimizes the competing generator and discriminator objectives (PDF pp. 665-672).
- **Conditional GAN:** supplies labels or auxiliary information so generation can be controlled (PDF p. 666).
- **RGAN/RCGAN:** recurrent generator and discriminator architectures for real-valued multivariate sequences; the conditional variant adds auxiliary information (PDF pp. 667-668).
- **TimeGAN:** combines an embedding/recovery autoencoder with generator/discriminator networks and a supervisor that constrains one-step latent dynamics (PDF pp. 673-676).
- **Diversity:** synthetic and real distributions should cover similar support.
- **Fidelity:** a held-out classifier should not reliably distinguish synthetic from real.
- **Usefulness:** training on synthetic data should perform competitively on a real held-out task (PDF pp. 684-689).

## Formulas and notation

The following are faithful mathematical renderings of code and notation verified on the cited pages.

- Let \(G(z)\) be the generated sample and \(D(\cdot)\) the discriminator logit. With binary cross-entropy `BCE`, the chapter implements:

  \[
  L_G=\operatorname{BCE}(1,D(G(z)))
  \]

  \[
  L_D=\operatorname{BCE}(1,D(x))
      +\operatorname{BCE}(0,D(G(z)))
  \]

  (PDF p. 671).

- TimeGAN reconstruction phase:

  \[
  H=E(X),\qquad \tilde X=R(H)
  \]

  with mean-squared reconstruction error and the displayed implementation loss \(10\sqrt{\operatorname{MSE}(X,\tilde X)}\) (PDF pp. 679-680).

- Supervised latent-dynamics loss compares the supervisor output with the next embedded step:

  \[
  L_S=\operatorname{MSE}\!\left(H_{:,1:,:},\hat H_{:,1:,:}\right)
  \]

  (PDF p. 680).

- The moment loss is implemented as:

  \[
  L_M=
  \operatorname{mean}\!\left|\mu_{\text{real}}-\mu_{\text{synthetic}}\right|
  +
  \operatorname{mean}\!\left|
  \sqrt{\sigma_{\text{real}}^2+10^{-6}}
  -
  \sqrt{\sigma_{\text{synthetic}}^2+10^{-6}}
  \right|
  \]

  (PDF p. 681).

- The displayed composite generator loss is:

  \[
  L_{\text{generator}}
  =L_U+L_{U,E}+100\sqrt{L_S}+100L_M
  \]

  where \(L_U\) and \(L_{U,E}\) are adversarial losses in the supervised and embedded paths (PDF p. 682).

## Assumptions

- The observed training sample adequately represents the distribution worth generating.
- The historical distribution and dynamics remain relevant outside the training period.
- The generator can capture both contemporaneous cross-feature dependence and longitudinal transitions (PDF pp. 673-675).
- The lower-dimensional latent space preserves the drivers of temporal dynamics (PDF pp. 674-675).
- Overlapping fixed-length windows are suitable training samples, despite strong dependence among adjacent windows.
- A weak discriminator on a held-out set indicates fidelity only when the classifier, labels, split, and metric orientation are themselves valid.
- Predictive utility on one narrow next-step task generalizes neither to trading profitability nor to tail/stress realism.

## Procedures described by the chapter

### Basic GAN

1. Build generator and discriminator networks.
2. Define binary-cross-entropy objectives.
3. For each minibatch, generate noise, produce fake samples, score real and fake samples, calculate both losses, take gradients, and update both networks.
4. Monitor training and inspect generated samples (PDF pp. 668-673).

### TimeGAN

1. Scale real multivariate sequences and construct overlapping windows.
2. Create random sequences with matching dimensions.
3. Build embedder, recovery, generator, discriminator, and supervisor networks.
4. Train the autoencoder on reconstruction loss.
5. Train the supervisor on one-step latent dynamics.
6. Jointly train all components with adversarial, supervised, reconstruction, and moment losses.
7. Generate synthetic sequences and reverse the scaling.
8. Evaluate diversity, fidelity, and usefulness on held-out real data (PDF pp. 673-689).

## Feature and model examples

- The TimeGAN example uses over 4,000 daily observations for six equity price series, scaled to `[0, 1]`, with overlapping sequences of 24 steps (PDF pp. 676-678).
- Embedder, recovery, generator, and discriminator use three GRU layers with 24 units; the supervisor uses two (PDF pp. 678-679).
- The reconstructed `SyntheticData` model has 28,854 trainable parameters (PDF p. 681).
- Joint training runs 10,000 iterations; the generator is updated twice as often as the discriminator (PDF p. 682).
- It generates 4,480 synthetic sequences of shape `24 x 6` (PDF p. 683).
- Diversity is inspected with PCA and t-SNE; fidelity with a real-versus-synthetic GRU classifier; usefulness with a next-step GRU trained separately on real or synthetic sequences (PDF pp. 684-689).

## Statistical, validation, and backtesting warnings

- Synthetic data do not create new independent economic regimes; they resample patterns learned from the historical training distribution.
- Adjacent rolling windows share 23 of 24 observations in the example. A simple chronological 80:20 split can still leave near-duplicate boundary windows unless an explicit gap is imposed (PDF pp. 677, 686).
- PCA/t-SNE overlap is qualitative and highly sensitive to preprocessing, sampling, and t-SNE hyperparameters (PDF pp. 684-686).
- The fidelity classifier reports accuracy about 0.4403 and AUC about 0.1596 on a balanced test set (PDF p. 687). An AUC far below 0.5 can indicate systematically reversed discrimination rather than indistinguishability; label orientation, implementation, and repeated-seed behavior must be audited before accepting the chapter's fidelity interpretation.
- The usefulness test predicts the next daily price step for only six tickers. Equal or slightly better MAE does not establish realistic returns, tails, volatility clustering, gaps, volume behavior, or trading performance (PDF pp. 688-689).
- GAN training seeks an equilibrium between two unstable optimizations and can suffer collapse or selective mode coverage (PDF p. 666).
- Generating prices rather than returns can make level/trend replication look realistic while failing to reproduce stationary dynamics.
- Synthetic observations must never be mixed into Validation or Final test or used to tune evaluation thresholds.

## Implementation patterns worth preserving

- Keep real-data preparation, random-data generation, each model component, each loss, each training phase, generation, and evaluation modular.
- Persist the source date range, scaler, sequence schema, architecture, random seeds, loss weights, checkpoints, and generated-data hash.
- Evaluate with multiple independent diagnostics: marginal moments, autocorrelation, cross-correlation, tails, drawdowns, regime coverage, discriminator tests, and downstream real-test tasks.
- Use repeated seeds and report dispersion; one GAN run is not a result.
- Keep synthetic augmentation as an experimental training input with an unchanged real-only validation contract.
- Record provenance at the synthetic-row level so generated and observed data cannot be confused.

## Dated APIs and examples

- The chapter ports original TensorFlow 1 TimeGAN code to TensorFlow 2 and uses `tf.function`, `GradientTape`, `tf.data.Dataset.from_generator`, and older Keras conventions (PDF pp. 671-683). Current signatures and serialization require verification.
- Optimizer and loss defaults are version-sensitive; they must be explicit.
- The example's Quandl Wiki and Yahoo-derived equity inputs are historical and unrelated to the trusted Project 1 data.
- PCA and t-SNE defaults have changed across scikit-learn versions; parameters and random state must be pinned.

## Project 1 relevance

- Synthetic data are **not a first-line feature** for the MLAT notebook. Existing Project 1 has millions of one-minute bars and a mature leakage-controlled research design; the immediate bottleneck is credible signal, not raw row count.
- The most relevant concepts are latent temporal embeddings, regime-conditioned augmentation, and real-only evaluation.
- A TimeGAN experiment might eventually support stress testing or model robustness, but only after the real-data feature hypothesis and baseline model are frozen.
- Existing evidence shows expansion structure and weak direction. Synthetic sequences must preserve this asymmetry instead of manufacturing balanced directional examples.

## One-minute GC adaptation

Deferred, bounded hypothesis:

> A session- and regime-conditioned sequence generator trained only on Development can augment training for an already frozen expansion model without degrading performance on untouched real Validation dates.

Necessary safeguards:

- Generate normalized returns/ranges/activity channels rather than unconstrained raw price levels.
- Condition on London/New York session, time remaining, volatility state, and contract/roll context where appropriate.
- Construct sequences that never cross nonconsecutive minutes, contract/instrument/segment boundaries, invalid roll/tradability periods, or forbidden date/session boundaries.
- Separate Development and Validation by trading-date blocks plus a gap at least as long as sequence and label overlap.
- Fit the generator only on Development; assess all claims on real Validation.
- Require preservation of marginal and joint moments, autocorrelation, volatility clustering, tails, excursion/range distributions, session seasonality, and downstream incremental performance.
- Predeclare augmentation ratio and prohibit tuning it on Final test.

## Unsuitable or deferred ideas

- Treating synthetic rows as additional independent history for p-values or confidence intervals.
- Training a generator on Development plus Validation and then evaluating on Validation.
- Using a GAN to balance profitable/unprofitable or long/short labels, which can alter the economic base rate.
- Judging quality from plots alone.
- Accepting a below-0.5 discriminator AUC as proof of indistinguishability without checking label orientation.
- Backtesting entirely on generated paths as evidence of live profitability.
- Implementing TimeGAN in the first controlled MLAT feature batch before simpler causal features and models are exhausted.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| GAN motivation and generative/discriminative models | 663-665 |
| Adversarial training and GAN variants | 665-668 |
| TensorFlow DCGAN implementation | 668-673 |
| TimeGAN motivation and architecture | 673-676 |
| Inputs, components, and three training phases | 676-683 |
| Synthetic output and diversity/fidelity/usefulness tests | 683-689 |
| Limitations and summary | 689-690 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 671 and 676.
- **Confidence: high** for architecture, training phases, code-level losses, data shapes, and reported diagnostics.
- **Confidence: high** for the DCGAN losses on PDF p. 671 and the TimeGAN three-phase design on PDF p. 676; both pages were visually inspected.
- **Confidence: medium** for the fidelity conclusion because the printed AUC/accuracy require an orientation and implementation audit.
- The book does not provide repeated-seed uncertainty or a formal two-sample test for synthetic fidelity.
- The example does not test one-minute futures, contract rolls, session boundaries, volume realism, or executable strategy performance.
- No synthetic-data use is authorized for Project 1 by this summary.
