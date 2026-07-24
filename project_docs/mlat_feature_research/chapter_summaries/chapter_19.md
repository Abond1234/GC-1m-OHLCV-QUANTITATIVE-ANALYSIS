# Chapter 19 - RNNs for Multivariate Time Series and Sentiment Analysis

**Assigned source range:** PDF pages 607-638  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Recurrent neural networks are designed for ordered data whose current interpretation depends on earlier sequence elements. Reusing the same transformation through time gives an RNN an internal state, but training over long sequences is expensive and vulnerable to vanishing or exploding gradients. LSTM and GRU gates address this problem by controlling what the recurrent state retains, updates, and exposes (PDF pp. 607-615).

The chapter implements univariate, stacked, and multivariate RNNs, then applies recurrent models to sentiment and SEC filings. The useful lesson for Project 1 is not that an RNN automatically finds price direction; it is that a causal sequence model can summarize multiscale temporal state if its information boundary, sequence construction, and validation are correct (PDF pp. 615-638).

## Concepts and definitions

- **Sequence mappings:** one-to-many, many-to-one, and many-to-many architectures allow variable-length input/output sequences (PDF pp. 608-609).
- **Recurrent state:** the current output depends on the current input and a state produced from prior elements; weights are shared across time (PDF pp. 609-610).
- **Backpropagation through time (BPTT):** the graph is unrolled over sequence steps, followed by a forward pass and a reverse gradient pass. The sequential dependency limits parallelism (PDF p. 610).
- **Teacher forcing:** prior observed output values are fed into later recurrent steps during training. It only helps if outcomes carry useful past information (PDF p. 611).
- **Bidirectional RNN:** combines forward and reverse scans, making an output depend on both earlier and later sequence elements (PDF p. 611). This is suitable only when both directions are available at inference.
- **Encoder-decoder/seq2seq:** an encoder maps the input sequence to latent state; a decoder maps that state to an output sequence (PDF pp. 611-612).
- **Attention:** a learned weighted combination of input representations replaces reliance on a single fixed context vector (PDF p. 612).
- **Transformer:** replaces recurrence and convolution with attention, allowing greater parallelism (PDF p. 612).
- **LSTM:** a recurrent unit with forget, input/update, and output gates plus a persistent cell state (PDF pp. 613-614).
- **GRU:** a simpler gated unit without a separate output gate or separate memory cell; the chapter notes competitive performance on smaller datasets (PDF pp. 614-615).

## Formulas and notation

Only formulas directly recoverable from the displayed PDF text are included.
The recurrent transformation on PDF p. 609 and LSTM gates on PDF p. 614 were
additionally checked against rendered pages; later input-shape notation is
reported from displayed code/text with its own page citations.

- The chapter's simple recurrent transformation is:

  \[
  y_t=g\!\left(W_{hh}h_{t-1}+W_{hx}x_t\right)
  \]

  where \(x_t\) is the current input/context, \(h_{t-1}\) the prior hidden state, and the weight matrices are shared through time (PDF p. 609).

- The LSTM equations shown in Figure 19.4 are:

  \[
  f_t=\sigma(W_f x_t+U_f h_{t-1}+b_f)
  \]

  \[
  i_t=\sigma(W_i x_t+U_i h_{t-1}+b_i)
  \]

  \[
  o_t=\sigma(W_o x_t+U_o h_{t-1}+b_o)
  \]

  \[
  u_t=\tanh(W_c x_t+U_c h_{t-1}+b_c)
  \]

  \[
  c_t=f_t\odot c_{t-1}+i_t\odot u_t,
  \qquad
  h_t=o_t\odot\tanh(c_t)
  \]

  where \(\odot\) denotes element-wise multiplication (PDF p. 614).

- TensorFlow recurrent layers expect a three-dimensional input:

  \[
  \text{samples}\times\text{time steps}\times\text{features}
  \]

  (PDF pp. 616, 627).

## Assumptions

- Earlier sequence elements contain information useful for the present prediction (PDF pp. 607-611).
- Shared recurrent parameters are appropriate across all positions in the sequence.
- A finite hidden/cell state can preserve the relevant history.
- Input transformations produce a sufficiently stationary and comparable scale for training (PDF pp. 615-620, 626-628).
- For teacher forcing, prior outputs contain useful information and the inference-time inputs match the training setup (PDF p. 611).
- Bidirectional recurrence assumes future sequence elements are legitimately available. This assumption fails for live bar-\(t\) trading features.

## Procedures described by the chapter

### Time-series RNN workflow

1. Transform and scale the source series.
2. Convert each observation into a trailing sequence with shape `sample x time x feature`.
3. Split in chronological order.
4. Choose a recurrent layer, hidden width, depth, output layer, loss, and optimizer.
5. Preserve temporal ordering during fitting where required (`shuffle=False` in the macro example).
6. stop using a valid holdout or fold and restore the selected checkpoint.
7. Reverse scaling before interpreting predictions (PDF pp. 615-628).

### Examples

- Univariate LSTM: 63 prior S&P 500 observations predict one step ahead (PDF pp. 615-620).
- Stacked LSTM: 52 weekly returns plus learned ticker embeddings and month indicators predict direction or return (PDF pp. 620-625).
- Multivariate LSTM: 18 months of transformed industrial production and sentiment predict both next values; the example compares with a VAR benchmark (PDF pp. 626-628).
- Text RNN: tokenize, pad, embed, use recurrent layers, and predict sentiment or forward equity returns (PDF pp. 629-638).

## Feature and model examples

- Sequential market inputs: lagged scaled index values, rolling weekly returns, and multivariate macro series (PDF pp. 615-628).
- Static/context inputs can join a recurrent branch, illustrated with ticker embeddings and one-hot months (PDF pp. 620-624).
- A stacked-LSTM classification example reports test IC 0.32; the regression variant reports an average weekly IC stated as 3.32 and an overall-period value of 6.68, with a top-bottom predicted-return spread slightly above 20 bps (PDF p. 625). These are book-specific equity results.
- A small multivariate LSTM reports test MAE 0.034 versus 0.043 for a VAR comparison, but the chapter says the forecast setups are not fully comparable (PDF p. 628).
- A bidirectional GRU with learned embeddings predicts five-day returns from SEC filings and reports test Spearman IC 6.02 (PDF pp. 634-638).

## Statistical, validation, and backtesting warnings

- RNNs struggle with long-range dependencies; repeated Jacobian products cause vanishing or exploding gradients even for relatively short sequences (PDF pp. 612-614).
- Bidirectional recurrence leaks future market observations if used to construct a live trading feature.
- The multivariate RNN/VAR comparison is not apples-to-apples because the RNN makes one-step forecasts while the VAR recursively consumes its own predictions (PDF p. 628).
- The SEC-filing label assumes filings occur after market hours and explicitly acknowledges this point-in-time assumption may be wrong (PDF p. 635).
- The filing experiment uses imperfect free prices and parsed text, an arbitrary five-day horizon, length/vocabulary choices, and a 90:10 split; the chapter advises caution (PDF pp. 634-638).
- A recurrent model's high capacity does not make nonstationary intraday price direction learnable.
- Overlapping rolling sequences and forward labels require purging/embargo; the chapter examples do not provide a Project 1-grade overlap audit.

## Implementation patterns worth preserving

- Use one reusable sequence builder that returns explicit arrays plus source-row identifiers and timestamps.
- Assert that each sequence is strictly causal and consecutive.
- Fit scalers on the training fold only and persist them with the model.
- Keep the sequential branch separate from static/session context, then combine them in an explicit model graph.
- Make `shuffle`, state reset, padding, masking, truncation, and sequence direction deliberate configuration fields.
- Save fold boundaries, sequence length, channel order, checkpoint, random seeds, and library versions.
- Compare with autoregressive, linear, and frozen Project 1 baselines on the same observations.

## Dated APIs and examples

- TensorFlow 2/Keras examples use `Sequential`, `LSTM`, `GRU`, `Bidirectional`, `Embedding(input_length=...)`, and legacy metric naming. Current Keras serialization and argument support must be verified.
- The examples use FRED through `pandas_datareader`, the discontinued Quandl Wiki equity dataset, and `yfinance`; none is required for the Project 1 GC notebook.
- The random `train_test_split` in the filing example is not a template for time-series research.
- Tokenization and padding APIs have evolved, and modern text work may use transformer tokenizers rather than the book's Keras sequence pipeline.

## Project 1 relevance

- GC is a single sequential market stream, making causal many-to-one recurrence conceptually more relevant than the book's cross-sectional ticker-embedding setup.
- Existing Project 1 evidence favors opportunity/expansion modeling over direction. An RNN should therefore be a later challenger for expansion or path-risk summaries, not an unrestricted directional strategy.
- Project 1 already carries session, rollover, tradability, contract, instrument, and continuous-segment boundaries. These must reset recurrent input windows.
- The current 85-feature matrix contains engineered summaries. An RNN experiment should test whether a small raw/near-raw trailing sequence adds information beyond those summaries, not merely re-encode the same feature family at much higher capacity.

## One-minute GC adaptation

Potential bounded hypothesis:

> A small causal GRU/LSTM over a predeclared trailing GC sequence captures volatility-state transitions not fully represented by the frozen scalar features and improves 60- or 180-minute expansion ranking.

Required adaptation:

- Use a **forward-only**, many-to-one network. Never use bidirectional recurrence on market bars.
- End the input at completed decision bar \(t\); entry and all outcomes begin at \(t+1\).
- Build channels from causal, scale-stable observations such as lagged returns, true range relative to trailing ATR, close-in-range, log-volume surprise, and explicit session time.
- Create windows from the trusted full GC history, then map them to the existing eligible observations.
- Reset on missing/nonconsecutive minutes, active-contract changes, instrument changes, segment changes, roll/tradability boundaries, and any session boundary required by the feature definition.
- Use Development-only expanding/walk-forward folds with embargo at least as strict as the overlapping outcome contract.
- Benchmark a small GRU against lagged linear/VAR-like models, `atr_20`, and the frozen ridge expansion model.
- Predeclare sequence lengths, hidden units, dropout, seed count, loss, horizon, and selection metric.

## Unsuitable or deferred ideas

- Bidirectional LSTM/GRU features for live GC decisions.
- Random observation splits or overlapping train/validation windows without purge/embargo.
- Text sentiment, ticker embeddings, or SEC-filing architectures in an OHLCV-only notebook.
- Stateful training that silently carries hidden state across contract, segment, date, or session resets.
- Very long sequences before short causal windows show incremental evidence.
- Treating the chapter's equity IC values as expected GC performance.
- Selecting sequence length or architecture after inspecting Final-test behavior.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| RNN motivation and sequence mappings | 607-609 |
| Recurrent graph, BPTT, alternative architectures | 609-612 |
| Long-range dependency problem, LSTM, GRU | 612-615 |
| Univariate S&P 500 LSTM | 615-620 |
| Stacked LSTM with contextual inputs | 620-625 |
| Multivariate macro RNN | 626-628 |
| Sentiment and pretrained/custom embeddings | 629-634 |
| SEC filing return model and caveats | 634-638 |

## Unresolved ambiguities and extraction confidence

- **Visual inspection pages:** PDF pp. 609 and 614.
- **Confidence: high** for architecture definitions, procedures, examples, warnings, and reported metrics.
- **Confidence: high** for the recurrent and LSTM equations; PDF pp. 609 and 614 were visually inspected.
- The simple recurrence labels its transformed value \(y_t\) while the surrounding diagram emphasizes hidden state \(h_t\); this note preserves the book's displayed notation rather than silently changing it.
- The stated weekly IC values on PDF p. 625 appear to be percentage-form reporting in context; this note preserves the printed numbers without reinterpretation.
- The chapter does not quantify multiple-testing costs across its recurrent architectures and preprocessing choices.
- No result in this chapter establishes incremental value for one-minute GC.
