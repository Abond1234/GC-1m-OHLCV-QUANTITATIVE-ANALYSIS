# Chapter 16 - Word Embeddings for Earnings Calls and SEC Filings

**Assigned source range:** PDF pages 506-532  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Word embeddings replace sparse token-count vectors with dense learned representations whose relative positions reflect how words are used in context. The chapter develops word2vec's continuous-bag-of-words (CBOW) and skip-gram objectives, pretrained GloVe vectors, domain-specific financial-news and SEC-filing training, doc2vec document representations, and the transition to contextual transformer models (PDF pp. 506-531).

Embedding quality depends on the corpus, context definition, vocabulary, objective, sampling scheme, and downstream task. Word analogies and nearest neighbors can diagnose representation quality, but they do not validate a trading signal. The chapter outlines how SEC returns could label filings yet does not complete or report a costed SEC-filing trading strategy (PDF pp. 510-531).

## Concepts and definitions

- **Embedding:** a dense, lower-dimensional vector learned for a semantic unit such as a token, phrase, or document (PDF pp. 506-507).
- **Distributional hypothesis:** words used in similar contexts tend to have related meanings, so context prediction can produce semantically useful vectors (PDF p. 507).
- **Context window:** the nearby tokens used to predict a target or predicted from a target (PDF pp. 507-509).
- **Continuous bag of words (CBOW):** predicts a target token from combined context vectors; it is faster and tends to favor frequent words in the chapter's comparison (PDF p. 508).
- **Skip-gram:** predicts surrounding context tokens from a target token; it can represent rare words better and work with smaller corpora (PDF p. 508).
- **Hierarchical softmax:** arranges the vocabulary in a binary tree so a word probability is computed along its path rather than against every vocabulary item (PDF p. 509).
- **Noise-contrastive estimation (NCE):** replaces full-vocabulary prediction with discrimination between observed and sampled noise pairs (PDF p. 509).
- **Negative sampling:** a simplified sampling objective that directly trains useful target-context similarity rather than a fully normalized language model (PDF p. 509).
- **Phrase detection:** merges frequently co-occurring tokens into a single vocabulary item, allowing one embedding for a multi-token expression (PDF pp. 509, 514-516, 521-522).
- **Semantic arithmetic:** vector offsets can encode regular relationships, evaluated through analogy tasks such as capital-country or grammatical transformations (PDF pp. 510-511).
- **Out-of-vocabulary (OOV) token:** a token absent from the fitted vocabulary and therefore lacking a learned static vector (PDF p. 514).
- **GloVe:** learns embeddings from aggregate global word-word co-occurrence statistics rather than the local prediction objective used by word2vec (PDF pp. 511-514).
- **doc2vec:** extends the representation to paragraphs or documents by learning document identifiers jointly with a token-prediction task (PDF pp. 524-527).
- **Static embedding:** assigns one vector to a token regardless of sentence context; it cannot distinguish different senses of a polysemous word (PDF pp. 528-530).
- **Self-attention:** learns how strongly every token contributes to another token's contextual representation (PDF p. 529).
- **Transformer:** applies multi-head self-attention and feedforward layers without recurrent sequential processing, enabling parallel computation and long-range context (PDF p. 529).
- **BERT:** a bidirectional transformer pretrained with masked-token and next-sentence objectives, then fine-tuned for downstream tasks (PDF pp. 529-530).

## Formulas and notation

The softmax and phrase-scoring equations were checked against rendered PDF p. 509. The SEC return-label construction was checked against rendered PDF p. 522.

- For target word \(w\), context \(c\), hidden/context representation \(h\), output vector \(v'_w\), and vocabulary \(V\), the chapter's full softmax is:

  \[
  p(w\mid c)
  =
  \frac{\exp(h^\mathsf{T}v'_w)}
       {\sum_{w_i\in V}\exp(h^\mathsf{T}v'_{w_i})}
  \]

  (PDF p. 509). Its denominator requires a dot product for every vocabulary item, motivating hierarchical softmax and sampling approximations.

- The displayed phrase score for candidate bigram \((w_i,w_j)\), with discount \(\delta\), is:

  \[
  \operatorname{score}(w_i,w_j)
  =
  \frac{\operatorname{count}(w_i,w_j)-\delta}
       {\operatorname{count}(w_i)\operatorname{count}(w_j)}
  \]

  (PDF p. 509).

- The displayed normalized pointwise mutual information is:

  \[
  \operatorname{NPMI}(w_i,w_j)
  =
  \frac{
    \ln\!\left(
      P(w_i,w_j)/(P(w_i)P(w_j))
    \right)}
    {-\ln P(w_i,w_j)}
  \]

  and ranges from \(-1\) to \(+1\) (PDF p. 509).

- For normalized vectors \(a\) and \(b\), their dot product is cosine similarity; semantic analogy tests seek a vector near:

  \[
  d\approx c+(b-a)
  \]

  (PDF pp. 508, 510-511).

- The SEC example labels filing \(i\), filed on date \(d_i\), with:

  \[
  r_i
  =
  \frac{P_{i,\mathrm{last}(d_i,d_i+1\text{ month})}}
       {P_{i,\mathrm{first}(d_i,d_i+1\text{ month})}}
  -1
  \]

  (PDF p. 522). This is a code-level calendar slice, not a fully specified executable event-study return.

## Assumptions

- Similar linguistic context implies useful semantic similarity for the downstream financial task (PDF pp. 506-508).
- The training corpus is large, representative, legally usable, and point-in-time appropriate.
- A fixed context window captures enough meaning for word2vec, while its order-insensitive treatment within that window is acceptable.
- Static word2vec/GloVe vectors adequately represent each token despite polysemy and evolving domain usage (PDF pp. 514, 528-530).
- Negative samples and token subsampling approximate the desired objective without distorting rare or finance-specific terms (PDF pp. 509, 516-519).
- Analogy accuracy and nearest-neighbor plausibility correlate with downstream feature quality (PDF pp. 510-514, 519-523).
- A generic pretrained corpus transfers to finance. The chapter also explains why corporate and industry language may require domain-specific training (PDF pp. 511-515).
- A document vector can summarize the sentiment-relevant content of a long, multi-entity financial document (PDF pp. 524-528).
- Transformer pretraining provides transferable language knowledge, and fine-tuning data is sufficient to adapt it without overfitting (PDF pp. 528-531).

## Procedures described by the chapter

### Word2vec training workflow

1. Define and clean a corpus, segment it into sentences, and detect reusable phrases.
2. Fit a vocabulary with a declared minimum frequency and OOV policy.
3. Choose CBOW or skip-gram, context-window size, embedding dimension, hierarchical-softmax or negative-sampling objective, and sample count.
4. Generate positive target-context pairs and sampled negative pairs.
5. Train shared embedding weights with a dot-product/sigmoid objective or a specialized word2vec implementation.
6. Save the vocabulary, vectors, model, and training configuration.
7. Evaluate analogy coverage/accuracy, nearest neighbors, stability, and performance on the actual downstream task (PDF pp. 507-523).

### Pretrained and financial-news examples

1. Convert and load GloVe vectors with Gensim and score standard word2vec analogy files.
2. For custom financial embeddings, clean more than 125,000 articles into about 2.43 million sentences.
3. Use repeated Gensim phrase detection to form bigrams and longer n-grams.
4. Generate roughly 120.4 million balanced target-context pairs for a transparent TensorFlow/Keras skip-gram illustration.
5. Train a 200-dimensional shared embedding layer, then use TensorBoard projections for qualitative inspection.
6. Train a faster 300-dimensional Gensim skip-gram model and continue epochs while monitoring analogy results (PDF pp. 511-520).

### SEC-filing example

1. Start from more than 22,000 10-Ks filed in 2013-2016 by over 6,500 companies.
2. Retain Items 1/1A, 7, and 7A; tokenize without lemmatization and detect phrases.
3. Match about 11,000 filings for roughly 3,000 companies to stock prices.
4. Illustrate a one-month post-filing return label.
5. Train and compare word2vec configurations using analogy performance, including architecture, negative sampling, minimum frequency, context window, sample count, and vector dimension (PDF pp. 521-523).

### doc2vec and transformer workflows

1. Sample 100,000 Yelp reviews per star class, clean them, discard very short reviews, and create tagged documents.
2. Fit 300-dimensional Gensim document vectors.
3. Train random forest, LightGBM, and multinomial-logistic classifiers on the document vectors.
4. Compare the result with the prior count-feature baseline.
5. For transformers, begin with a pretrained checkpoint, attach a task-specific head, and fine-tune on labeled in-domain data; this part is conceptual rather than a complete financial experiment (PDF pp. 524-531).

## Feature and model examples

- GloVe sources include Common Crawl, Wikipedia/Gigaword, and Twitter. The displayed analogy results vary materially by source, coverage, and category (PDF pp. 511-514).
- The Wikipedia-based GloVe example reports 75.44 percent overall analogy accuracy; Common Crawl is shown near 78 percent on covered analogies, while Twitter has much lower coverage and 56.4 percent accuracy (PDF pp. 512-513).
- The financial-news Keras example uses a vocabulary near 59,617 and 200 dimensions, yielding almost 12 million embedding parameters (PDF pp. 516-518).
- The faster financial-news Gensim model uses skip-gram, a five-token window, 300 dimensions, negative sampling, and multiple epochs; the best reported analogy accuracy is 41.75 percent (PDF pp. 519-520).
- The SEC experiment reports that skip-gram and negative sampling beat CBOW and hierarchical softmax in that setup; a 600-dimensional model reaches the best reported analogy accuracy of 38.5 percent (PDF pp. 522-523).
- The Yelp doc2vec example leaves 485,825 documents and compares 300-dimensional document vectors across three classifiers. LightGBM reports 62.24 percent accuracy, versus 41.50 percent for random forest and 39.81 percent for logistic regression (PDF pp. 524-527).
- The chapter notes that the earlier count/structured-feature Yelp LightGBM result was 73.6 percent, so the embedding example does not establish superiority (PDF p. 527).

## Statistical, validation, and backtesting warnings

- Training an embedding on the full historical corpus leaks future vocabulary, semantic usage, entities, and events into earlier predictions.
- A pretrained checkpoint can also leak future language relative to a historical backtest if its pretraining cutoff postdates the simulated decision period.
- Analogy benchmarks measure selected linguistic regularities, not finance-specific meaning, return prediction, calibration, or net trading value (PDF pp. 510-514, 519-523).
- Vocabulary coverage makes analogy accuracy conditional: models evaluated on different covered subsets are not directly comparable.
- Nearest-neighbor lists and two-dimensional projections invite subjective cherry-picking.
- Phrase detection, minimum frequency, context window, dimension, epochs, negative samples, seed, and corpus choice create a large multiple-testing surface.
- Documents from the same issuer, filing, story, or review author can cross random splits and inflate generalization.
- The SEC code uses `date_filed` and the first adjusted close in a calendar slice. It does not resolve SEC acceptance timestamp, after-hours filing, weekends/holidays, actionable next open, corporate actions, or a benchmark-adjusted return (PDF p. 522).
- Overlapping one-month filing labels require chronological purging/embargo and issuer-aware grouping.
- The SEC section evaluates embeddings through analogies but does not report a supervised post-filing return model, transaction costs, or portfolio backtest (PDF pp. 521-523).
- The Yelp doc2vec split is a product-review classification demonstration, not a chronological financial validation (PDF pp. 524-528).
- Financial documents can discuss multiple entities and mixed positive/negative aspects; one document-level sentiment score may erase the decision-relevant target and span (PDF pp. 527-531).
- Large pretrained transformers reduce training cost but increase capacity, hidden pretraining-data dependence, reproducibility burden, and sensitivity to fine-tuning choices.

## Implementation patterns worth preserving

- Record immutable raw-document hashes, source and first-seen timestamps, revisions, language, entity links, corpus cutoff, and license.
- Fit text cleaning, phrases, vocabulary, embedding weights, and downstream model inside Development training windows.
- Save token-to-ID order, OOV policy, frequency counts, context-window definition, negative-sampling distribution, seed, optimizer, epochs, checkpoint, and vector matrix.
- Identify pretrained checkpoints by immutable revision/hash and document their training-data cutoff when known.
- Separate intrinsic embedding diagnostics from downstream prediction and economic evaluation.
- Use issuer/story grouping plus chronological folds to prevent related documents crossing boundaries.
- For event labels, preserve acceptance timestamp, market calendar, first executable price, benchmark, horizon, and overlap metadata.
- Compare frozen count/TF-IDF, static embedding, and contextual embedding baselines on identical documents and rows.
- Monitor OOV rate, vocabulary drift, embedding-neighbor stability, and representation shift through time.
- Cache expensive immutable artifacts, but never reuse a model fitted beyond the current training boundary.

## Dated APIs and examples

This chapter is especially time-sensitive: both embedding APIs and the transformer ecosystem have changed substantially since the book's examples.

- Gensim `Word2Vec(size=..., iter=...)`, `wv.accuracy`, `model.docvecs`, and some `most_similar` access patterns have newer names or locations; current releases use version-specific vocabulary-building and training semantics (PDF pp. 512, 519-526).
- `glove2word2vec` conversion and `KeyedVectors` loading should be checked against the exact Gensim version and source format (PDF p. 512).
- spaCy `create_pipe("sentencizer")`, `add_pipe` with a component object, and multiprocessing examples reflect an older pipeline API (PDF pp. 514-515).
- TensorFlow/Keras `skipgrams`, sampling tables, functional `Input`, `Embedding(input_length=...)`, TensorBoard projector metadata, and serialization formats require current checks (PDF pp. 516-519).
- LightGBM's `early_stopping_rounds` and verbosity arguments, and scikit-learn multiclass defaults, have changed across releases (PDF pp. 526-527).
- The stated counts of Hugging Face models/languages and the 2019 library/company descriptions on PDF pp. 530-531 are historical, not current inventory.
- The chapter predates modern finance-specific transformer checkpoints and current model-governance expectations; any candidate requires independent validation rather than selection by name.

## Project 1 relevance

- Project 1 currently has no approved point-in-time text corpus, so word2vec, doc2vec, and transformer features are outside the new statistical-feature notebook's implementation scope.
- The chapter's SEC-filing application is cross-sectional US equity research and does not transfer directly to one GC futures stream.
- The transferable lesson is reproducibility for learned representations: corpus cutoff, preprocessing, vocabulary, checkpoint, seed, and downstream join must be frozen artifacts.
- If text is introduced later, a simple TF-IDF/linear baseline should precede custom embeddings or transformer fine-tuning.
- Any text representation must add stable Validation value beyond the existing market-only anchor/ridge benchmarks on identical eligible decision rows.

## One-minute GC adaptation

Potential deferred hypothesis:

> Contextual embeddings from an approved, timestamp-audited macro/commodities news corpus may improve entity-aware event classification, whose frozen outputs could add incremental information to one-minute GC forecasts.

Prerequisites and adaptation requirements:

- Do not implement without a licensed historical source carrying source, first-seen, revision, and vendor-arrival timestamps.
- Prefer event/category classification over an opaque single sentiment score, and define the GC relevance rule before examining returns.
- Use a pretrained checkpoint only if its identity, revision, tokenizer, and known corpus cutoff are recorded; disclose unresolved vintage leakage.
- Fine-tune only inside Development windows and make the text score available after realistic ingestion/inference latency.
- Deduplicate story families and group related documents across folds.
- Aggregate document scores to decision minutes with fixed age, decay, source, missingness, and confidence rules.
- Keep London and New York evaluation separate and account explicitly for scheduled macro releases.
- Compare count/TF-IDF, static embeddings, contextual embeddings, and market-only baselines on identical chronological folds, labels, and costs.
- Treat every checkpoint, pooling strategy, seed, learning rate, epoch, maximum length, and layer choice as a logged trial.

## Unsuitable or deferred ideas

- Training embeddings on the full 2010-present corpus before simulating early dates.
- Using current pretrained models in historical tests without documenting corpus-cutoff uncertainty.
- Copying SEC filing-date return labels to one-minute GC.
- Treating analogy accuracy, nearest neighbors, or embedding plots as evidence of alpha.
- Averaging all token vectors from a multi-entity financial document without target-aware segmentation.
- Fine-tuning a large transformer before a sparse linear baseline earns escalation.
- Selecting checkpoint, layer, pooling, or prompt using Validation or Final-test outcomes.
- Adding large text dependencies and model downloads to the reproducible statistical-feature notebook without an approved text workstream.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Motivation, dense contextual representations, and distributional semantics | 506-508 |
| CBOW/skip-gram, softmax approximations, negative sampling, and phrase detection | 508-510 |
| Semantic arithmetic, analogy evaluation, and pretrained-versus-domain vectors | 510-511 |
| GloVe sources, loading, analogy results, and visualization | 511-514 |
| Financial-news preprocessing, sentence segmentation, and phrase construction | 514-516 |
| TensorFlow/Keras skip-gram sampling, layers, training, and visualization | 516-519 |
| Faster Gensim training and financial-news evaluation | 519-521 |
| SEC filing corpus, return-label illustration, model training, and evaluation | 521-523 |
| doc2vec Yelp preparation and training | 524-526 |
| Downstream doc2vec classifiers, results, and finance-domain limitations | 526-528 |
| Attention, transformers, and BERT innovations | 528-530 |
| Pretrained libraries, financial-text outlook, and summary | 530-532 |

## Unresolved ambiguities and extraction confidence

- **Confidence: high** for the word2vec objectives, corpus workflows, reported corpus sizes, parameter comparisons, and doc2vec classifier results; sequential extraction was clear across PDF pp. 506-532.
- **Confidence: high** for the formulas and architecture details transcribed above. The actual visually inspected PDF pages were **508, 509, 517, 522, 527, and 530**.
- **Visual inspection pages:** PDF pp. 508, 509, 517, 522, 527, 530.
- PDF p. 508 was inspected to verify the CBOW-versus-skip-gram processing diagram.
- PDF p. 509 was inspected to verify the softmax, phrase-score, and NPMI equations.
- PDF p. 517 was inspected to verify the positive/negative pair generation and shared embedding-layer design.
- PDF p. 522 was inspected to verify the exact filing-return code and its unresolved execution-time issue.
- PDF p. 527 was inspected to verify the three confusion matrices and reported Yelp classifier accuracies.
- PDF p. 530 was inspected to verify the stated BERT depth/head counts and pretraining tasks.
- **Source ambiguity:** the DBOW and distributed-memory descriptions on PDF p. 524 appear reversed relative to conventional doc2vec naming. This summary does not rely on that mapping; verify against the implementation's pinned documentation.
- **Source/code ambiguity:** PDF p. 515 says retained sentences are between 6 and 99 tokens, but the displayed condition uses `or` rather than `and`; copying it would not enforce the prose range.
- **Source ambiguity:** PDF p. 516 first describes a 31,300-token vocabulary and shortly afterward reports close to 60,000 tokens including n-grams; the exact intermediate pipeline state is not fully clear from the chapter text.
- The chapter does not report a completed supervised SEC-return model, a point-in-time filing event study, or a costed trading strategy.
- Project 1 applicability is deferred pending an approved text corpus. This summary authorizes no embedding model, transformer, or Final-test exposure.
