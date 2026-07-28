# Chapter 15 - Topic Modeling - Summarizing Financial News

**Assigned source range:** PDF pages 486-505  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Topic modeling compresses a high-dimensional document-term matrix (DTM) into latent themes that can summarize, organize, retrieve, and annotate large text collections. The chapter traces three approaches: latent semantic indexing (LSI) applies linear dimensionality reduction; probabilistic latent semantic analysis (pLSA) adds a probabilistic model of document-word co-occurrence; and latent Dirichlet allocation (LDA) adds priors that generate document-topic and topic-word distributions (PDF pp. 486-496).

Latent topics are not observed facts. Their meaning depends on corpus construction, tokenization, vocabulary thresholds, topic count, priors, inference, random initialization, and human interpretation. Perplexity and coherence can assist comparison, but neither guarantees economic meaning or a useful trading feature (PDF pp. 496-505).

## Concepts and definitions

- **Topic model:** an unsupervised model that represents documents through latent themes inferred from patterns of token usage (PDF pp. 486-488).
- **Latent semantic indexing (LSI), or latent semantic analysis (LSA):** applies truncated singular value decomposition (SVD) to the DTM to create a lower-rank document-topic space (PDF pp. 488-492).
- **Singular value:** scales a latent direction in the SVD; retaining the largest values yields a lower-rank approximation (PDF pp. 488-489).
- **Probabilistic latent semantic analysis (pLSA):** models document-word co-occurrences through hidden topics and conditionally independent multinomial distributions (PDF pp. 492-494).
- **Non-negative matrix factorization (NMF):** with a Kullback-Leibler objective, the implementation described is equivalent to pLSA and produces non-negative, more directly interpretable weights (PDF p. 493).
- **Latent Dirichlet allocation (LDA):** a hierarchical Bayesian model in which documents are distributions over topics and topics are distributions over words (PDF pp. 494-496).
- **Dirichlet distribution:** generates positive probability vectors that sum to one. A smaller concentration parameter tends to place most mass on fewer entries (PDF pp. 494-495).
- **Document-topic distribution \(\theta_m\):** topic proportions for document \(m\) (PDF pp. 495-496).
- **Topic-word distribution \(\phi_k\):** token probabilities for topic \(k\) (PDF pp. 495-496).
- **Bayesian inference:** reverses the assumed LDA document-generation process to infer latent topic assignments and distributions from observed words (PDF p. 496).
- **Variational Bayes:** the approximate posterior method used in the original LDA paper; Gibbs sampling and expectation propagation are alternatives mentioned by the chapter (PDF p. 496).
- **Perplexity:** a predictive-fit measure evaluated on unseen text; lower values indicate better probability assigned to that sample under a consistent definition (PDF pp. 496-498).
- **Topic coherence:** measures whether highly weighted topic words co-occur in a way that tends to appear semantically meaningful to people (PDF p. 497).
- **UMass coherence:** an intrinsic metric based on document co-occurrences in the training corpus (PDF p. 497).
- **UCI coherence:** an external metric based on pointwise mutual information from sliding-window co-occurrences in a reference corpus (PDF p. 497).
- **pyLDAvis relevance:** ranks a term for a selected topic by trading off within-topic probability against lift relative to corpus-wide frequency (PDF pp. 498-499).

## Formulas and notation

The LSI decomposition, pLSA mixture equation, LDA plate notation, and evaluation equations were checked against rendered PDF pages 489, 492, 496, and 497.

- For an \(M\times N\) document-term matrix \(X\), LSI applies:

  \[
  X=U\Sigma V^\mathsf{T}
  \]

  and retains \(T<N\) singular directions:

  \[
  X\approx U_T\Sigma_TV_T^\mathsf{T}
  \]

  The rows of \(U_T\Sigma_T\) locate documents in the latent topic space (PDF pp. 488-489).

- The pLSA symmetric and asymmetric factorizations shown by the chapter are:

  \[
  P(w,d)
  =
  \sum_t P(d\mid t)P(w\mid t)
  =
  P(d)\sum_tP(t\mid d)P(w\mid t)
  \]

  (PDF p. 492).

- The LDA plate diagram uses:

  \[
  \theta_m\sim\operatorname{Dirichlet}(\alpha),
  \qquad
  \phi_k\sim\operatorname{Dirichlet}(\beta)
  \]

  followed conceptually by a topic assignment \(z_{mn}\) drawn from \(\theta_m\) and observed word \(w_{mn}\) drawn from the corresponding \(\phi_{z_{mn}}\) (PDF pp. 495-496).

- The chapter displays perplexity through entropy \(H(p)\) of word distribution \(p\):

  \[
  2^{H(p)}
  =
  2^{-\sum_w p(w)\log_2p(w)}
  \]

  (PDF p. 497). This display is not the full corpus-normalized held-out likelihood formula commonly implemented by LDA libraries, so metric definitions must be recorded rather than inferred from the label alone.

- With top-word set \(W\) and smoothing \(\epsilon\), the displayed UCI coherence is:

  \[
  \operatorname{coherence}_{\mathrm{UCI}}
  =
  \sum_{(w_i,w_j)\in W}
  \log
  \frac{p(w_i,w_j)+\epsilon}
       {p(w_i)p(w_j)}
  \]

  (PDF p. 497).

- The displayed UMass coherence is:

  \[
  \operatorname{coherence}_{\mathrm{UMass}}
  =
  \sum_{(w_i,w_j)\in W}
  \log
  \frac{D(w_i,w_j)+\epsilon}
       {D(w_j)}
  \]

  where \(D(\cdot)\) denotes training-corpus document counts (PDF p. 497).

## Assumptions

- Repeated word co-occurrence reflects stable latent themes rather than source templates, boilerplate, duplicated stories, or period-specific entities.
- A bag-of-words representation is sufficient even though it omits word order and much linguistic context (PDF pp. 486-488).
- LSI assumes a linear low-rank representation is meaningful; its orthogonal factors may have both positive and negative weights that complicate interpretation (PDF pp. 488-492).
- pLSA assumes observed document-word co-occurrences can be generated by a mixture of conditionally independent topic-word distributions (PDF pp. 492-494).
- LDA assumes exchangeable words within a document and exchangeable documents within a corpus, conditional on latent distributions.
- Sparse Dirichlet priors appropriately encode that a document emphasizes few topics and a topic emphasizes few words (PDF pp. 494-496).
- A fixed number of topics can be chosen before training. Topic count is a hyperparameter, not a quantity identified unambiguously by the data (PDF pp. 492-505).
- Human-readable topic words correspond to a coherent semantic construct and remain stable out of sample (PDF pp. 496-505).
- An external coherence corpus used for UCI-style evaluation is appropriate for the language and domain (PDF p. 497).

## Procedures described by the chapter

### Generic topic-modeling workflow

1. Define documents and construct a time-appropriate corpus.
2. Clean and tokenize text; remove generic and domain-specific stop words.
3. Fit the vocabulary and DTM on training documents.
4. Choose LSI, pLSA/NMF, or LDA and predeclare a candidate topic-count range.
5. Fit the model and transform held-out documents without rebuilding the vocabulary.
6. Inspect topic-word lists and document-topic mixtures.
7. Compare held-out perplexity, one or more explicitly defined coherence metrics, and human review.
8. Test topic stability across seeds, samples, time blocks, and preprocessing choices.
9. Freeze the model before using document-topic weights as predictive features (PDF pp. 486-505).

### LSI and pLSA examples

1. Split the 2,225 BBC articles into training data and a stratified 50-document test set.
2. Fit a TF-IDF vocabulary of roughly 2,900 words using training documents only.
3. Fit five-component `TruncatedSVD`, transform train/test documents, and compare topic weights with five known BBC categories.
4. Fit five-component NMF with multiplicative updates and Kullback-Leibler loss as the pLSA implementation.
5. Compare the non-negative pLSA topic weights and top words with LSI's signed factors (PDF pp. 489-494).

### LDA examples

1. Fit five-topic scikit-learn LDA to the BBC DTM and monitor in-sample perplexity.
2. Inspect inter-topic distances and relevance-ranked words with pyLDAvis.
3. Convert the sparse matrix and vocabulary to Gensim structures, fit `LdaModel`, and evaluate UMass coherence.
4. Apply the trained model to 50 test articles; the chapter reports four category-assignment errors (PDF pp. 498-501).
5. For earnings calls, split about 700 transcripts into speaker statements, exclude operators and very short text, lemmatize, remove domain stop words, and fit 15 topics.
6. Run several hundred earnings-call experiments over vocabulary thresholds, count type, topic count, and training passes.
7. Apply a 15-topic model to a selected subset of 120,000 financial-news articles from the first five months of 2018 (PDF pp. 501-505).

## Feature and model examples

- Five LSI components explain only about 5.4 percent of the BBC DTM variance; positive and negative topic weights make interpretation difficult (PDF pp. 489-492).
- NMF/pLSA produces non-negative topic weights and a reported reconstruction error, yielding more distinct BBC category associations (PDF pp. 493-494).
- The LDA BBC example uses five topics, batch learning, up to 500 iterations, and periodic evaluation (PDF p. 498).
- The Gensim BBC example reports UMass coherence and 92 percent category assignment on a 50-document test set, but this is an illustrative topic/category comparison rather than a trading result (PDF pp. 499-501).
- The earnings-call corpus yields 32,047 speaker statements before additional cleaning and 22,582 after removing domain stop words and short statements. Its DTM contains 1,529 terms, and the example fits 15 topics with 25 passes (PDF pp. 501-503).
- The earnings-call experiment varies minimum/maximum document frequency, binary versus absolute counts, 3-50 topics, and 1 versus 25 passes. The displayed results deteriorate beyond roughly 25-30 topics in that setup (PDF pp. 503-504).
- The financial-news example starts from more than 306,000 articles, selects 120,000, and fits a 15-topic model with a 3,570-token vocabulary (PDF pp. 504-505).

## Statistical, validation, and backtesting warnings

- Topic modeling is unsupervised but still learns from the corpus. Fitting vocabulary, stop words, IDF, topics, or priors using future documents leaks future distribution and event information.
- Topic IDs are permutation-invariant and can split, merge, or drift across seeds and time windows. “Topic 7” is not a stable feature name without an explicit alignment procedure.
- Coherent top words do not imply a topic predicts returns. Human interpretation after seeing outcomes can create a narrative-selection bias.
- Perplexity and coherence can disagree with each other and with human judgment (PDF pp. 496-504).
- The chapter's metric-direction wording is internally inconsistent: PDF p. 497 says coherence nearer zero is better, while PDF pp. 503-504 call higher coherence better. Implementations and coherence variants use different ranges and conventions, so compare only identically defined metrics.
- The displayed entropy perplexity on PDF p. 497 has a lower bound of 1, despite the nearby phrase “closer to zero.” Lower is better under a fixed definition; zero should not be treated as the literal target.
- Topic-count, vocabulary, preprocessing, passes, priors, seeds, and model family create a large multiple-testing surface. The earnings-call example explicitly explores hundreds of configurations (PDF pp. 503-504).
- The 50-document BBC test is too small to support precise performance claims, and category labels are not an objective measure of topic quality (PDF pp. 489-501).
- Statement-level earnings-call documents are not independent: many share the same company, call, speaker, quarter, and market event.
- Financial-news articles may be duplicated, syndicated, revised, or published after the timestamp assumed by a market join.
- Assigning subsequent returns as labels after inspecting discovered topics risks both data snooping and ambiguous event-window attribution (PDF p. 503).

## Implementation patterns worth preserving

- Store raw-document identity, immutable content hash, source/first-seen/ingestion timestamps, revision lineage, language, and entity mapping.
- Define the document unit explicitly: full article, paragraph, speaker statement, headline, or time bucket.
- Fit tokenizer, domain stop words, vocabulary, DTM weighting, and topic model within each Development training window.
- Save vocabulary order, token counts, model priors, random seed, inference settings, topic-word matrix, document-topic matrix, and package versions.
- Keep held-out perplexity, each named coherence variant, and human-review notes as separate diagnostics.
- Align topics across folds using a frozen similarity-and-assignment rule; preserve unmatched, split, or merged-topic warnings.
- Report topic prevalence and stability through time, not only top-word lists from the full sample.
- Maintain sparse matrices throughout preprocessing.
- Treat topic probabilities as derived model outputs and join them to market rows only through audited availability timestamps.
- Preserve every tried configuration in the research ledger, including visually rejected models.

## Dated APIs and examples

The examples reflect the book's software environment. Reuse requires current API verification and exact version pins.

- scikit-learn `TruncatedSVD`, `NMF`, and `LatentDirichletAllocation` parameters/defaults differ across releases; the displayed estimator representation still includes legacy `n_topics=None` (PDF pp. 489, 493, 498).
- `get_feature_names()` has been replaced in newer scikit-learn APIs by a different feature-name method (PDF p. 499).
- Gensim's `Sparse2Corpus`, `LdaModel`, `LdaMulticore`, `top_topics`, coherence implementations, defaults, and model serialization require version-specific checks (PDF pp. 499-505).
- pyLDAvis integration adapters and notebook rendering have changed; record both pyLDAvis and underlying model versions (PDF pp. 498-503).
- spaCy model names and pipeline APIs used for the earnings-call/news cleaning reflect an older environment (PDF pp. 501-504).
- Seeking Alpha scraping and the 2018 Kaggle news collection are historical examples, not approved or guaranteed reproducible Project 1 data sources.

## Project 1 relevance

- Project 1 currently studies reproducible causal features from one-minute GC market data and has no approved text corpus. Topic modeling is therefore outside the new notebook's implementation scope.
- If point-in-time text is later introduced, topic mixtures could provide lower-dimensional event/theme descriptors, but their vocabulary, model, topic alignment, and availability join would require fold-local governance.
- Topic coherence is not a sufficient escalation criterion. A proposed topic feature must add stable Validation value beyond the market-only anchor/ridge benchmarks on identical eligible rows.
- The chapter reinforces a general notebook principle: learned representations and their preprocessing artifacts must be serialized, versioned, and reproducible rather than recomputed silently.

## One-minute GC adaptation

Potential deferred hypothesis:

> Fold-fitted topic proportions from an approved, point-in-time macro and commodities news corpus may identify event themes whose interaction with causal market state improves short-horizon GC expansion or direction forecasts.

Prerequisites and adaptation requirements:

- Do not proceed without a legally usable historical corpus with first-seen timestamps, revisions, vendor latency, source identity, and deduplication metadata.
- Predeclare the document unit, GC relevance rule, language policy, and handling of scheduled versus unscheduled news.
- Fit all text cleaning, vocabulary thresholds, topic count, priors, and topic alignment using Development data only.
- Make a document-topic vector available only after realistic arrival and processing latency.
- Aggregate topic probabilities to one-minute decision rows using a frozen rule that records age, count, missingness, and source concentration.
- Prevent documents from the same story cluster or event from crossing folds where this would compromise independence.
- Compare topic-plus-market and market-only models on the same dates, sessions, horizons, purging/embargo, costs, and labels.
- Require topic stability across seeds and chronological windows before interpreting a topic as a named recurring event class.

## Unsuitable or deferred ideas

- Adding earnings-call or general-equity-news topics to a GC statistical-feature notebook without an approved source and mechanism.
- Fitting one global topic model over the complete history before walk-forward evaluation.
- Naming topics after viewing subsequent GC returns.
- Treating pyLDAvis separation or top-word plausibility as predictive validation.
- Selecting topic count or vocabulary thresholds on Validation or Final test.
- Using raw topic IDs without cross-fold permutation alignment.
- Copying the BBC five-topic, earnings-call 15-topic, or financial-news 15-topic settings.
- Treating the chapter's BBC category assignment as evidence that the same topics will generalize through market regimes.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Motivation, uses, topic-model evolution, and limitations of bag of words | 486-488 |
| LSI/LSA, truncated SVD, BBC implementation, and topic interpretation | 488-492 |
| pLSA generative model, NMF implementation, strengths, and limitations | 492-494 |
| LDA, Dirichlet priors, and the document-generation process | 494-496 |
| LDA inference, perplexity, UCI/UMass coherence, and sklearn | 496-498 |
| pyLDAvis relevance and Gensim implementation | 498-500 |
| BBC topic evaluation and earnings-call corpus preparation | 500-502 |
| Earnings-call model, visualization, and hyperparameter experiments | 502-504 |
| Large financial-news example and summary | 504-505 |

## Unresolved ambiguities and extraction confidence

- **Confidence: high** for the model progression, examples, corpus sizes, procedures, and reported metrics; sequential extraction was clear across PDF pp. 486-505.
- **Confidence: high** for the formulas and model dependencies transcribed above. The actual visually inspected PDF pages were **489, 492, 496, 497, 503, and 504**.
- **Visual inspection pages:** PDF pp. 489, 492, 496, 497, 503, 504.
- PDF p. 489 was inspected to verify the LSI/SVD matrix roles and \(U_T\Sigma_T\) document-topic projection.
- PDF p. 492 was inspected to verify the symmetric/asymmetric pLSA equation and plate diagram.
- PDF p. 496 was inspected to verify LDA parameter meanings and plate dependencies.
- PDF p. 497 was inspected to verify the displayed perplexity and UCI/UMass equations.
- PDF pp. 503-504 were inspected to verify the pyLDAvis earnings-call example and the plotted hyperparameter sensitivity.
- **Source ambiguity:** the chapter gives inconsistent shorthand about the desired direction/bound of perplexity and coherence on PDF pp. 497, 503-504. Any implementation must name the exact library metric and verify its direction empirically.
- **Confidence: medium** for exact current output of scikit-learn, Gensim, spaCy, and pyLDAvis because the book does not provide a current lockfile or model-artifact hashes.
- The chapter does not specify a point-in-time trading join, a robust topic-alignment method, a multiple-testing adjustment, or an executable costed strategy.
- Project 1 applicability is deferred pending an approved text corpus. This summary authorizes no topic feature or Final-test exposure.
