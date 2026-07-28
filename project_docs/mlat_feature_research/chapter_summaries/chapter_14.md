# Chapter 14 - Text Data for Trading - Sentiment Analysis

**Assigned source range:** PDF pages 461-485  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature is approved by this note.

## Main argument

Text contains potentially valuable information but must be converted from unstructured language into numerical features without discarding the meaning relevant to the prediction task. The chapter presents a classical natural-language-processing (NLP) pipeline: parse and tokenize documents, add linguistic and semantic annotations, form a document representation, attach labels and other data, and train a predictive model (PDF pp. 461-465).

Its practical baseline is the bag-of-words document-term matrix, optionally weighted by term frequency-inverse document frequency (TF-IDF), followed by a simple classifier such as multinomial naive Bayes or regularized logistic regression. The examples show that basic sparse models can be competitive, but they do not establish a tradable text signal or solve release-time, entity, label, and domain-transfer problems (PDF pp. 471-485).

## Concepts and definitions

- **Document:** one text sample, such as a filing, article, headline, analyst note, or social-media post (PDF p. 462).
- **Corpus:** a collection of documents used to define a vocabulary or fit a model (PDF p. 462).
- **Token:** a character sequence treated as a semantic unit; the vocabulary is the subset of tokens retained for modeling (PDF pp. 461, 463-464).
- **n-gram:** a contiguous sequence of \(n\) tokens. Bigrams and trigrams preserve some short-range ordering but expand the feature space (PDF pp. 463-464, 469).
- **Stemming:** removes common endings using relatively simple rules; it can reduce vocabulary size while producing linguistically imperfect roots (PDF pp. 464, 471).
- **Lemmatization:** maps an inflected token to a canonical form using richer linguistic rules (PDF p. 464).
- **Part-of-speech (POS) tagging:** assigns a grammatical role, such as noun or verb, to each token (PDF pp. 463-464, 466-470).
- **Dependency parsing:** identifies syntactic relationships among tokens in a sentence (PDF pp. 463-464, 466-467).
- **Named-entity recognition (NER):** identifies real-world objects such as organizations, people, places, and dates (PDF pp. 463-464, 468).
- **Bag-of-words model:** represents a document by token presence, counts, or weights while discarding most word order and grammar (PDF pp. 471-473).
- **Document-term matrix (DTM):** a sparse matrix with documents as rows, vocabulary tokens as columns, and token values as entries (PDF p. 472).
- **Cosine similarity:** compares document vectors by their angle, reducing the direct effect of document magnitude (PDF pp. 472-473).
- **Term frequency (TF):** the count or transformed count of token \(t\) in document \(d\) (PDF pp. 473-477).
- **Inverse document frequency (IDF):** downweights tokens that occur in many documents and therefore tend to be less discriminating (PDF pp. 473, 476-477).
- **Sentiment polarity:** an estimate on a negative-to-positive scale; the TextBlob example also reports subjectivity on an objective-to-subjective scale (PDF p. 471).
- **Naive Bayes classifier:** applies Bayes' theorem under conditional independence of features given a class, enabling fast closed-form estimates in high-dimensional sparse data (PDF pp. 477-480).
- **Domain-specific sentiment model:** learns labels relevant to a particular corpus or task rather than relying on a generic sentiment lexicon (PDF pp. 480-481).

## Formulas and notation

The displayed cosine-similarity, TF-IDF, and naive Bayes equations were checked against rendered PDF pages 472, 473, and 479.

- For document vector \(d\) and query or comparison vector \(q\), cosine similarity is:

  \[
  \operatorname{similarity}(d,q)
  =
  \cos(\alpha)
  =
  \frac{d^\mathsf{T}q}{\lVert d\rVert\,\lVert q\rVert}
  \]

  (PDF pp. 472-473).

- The chapter's default scikit-learn-style weighting is:

  \[
  \operatorname{tfidf}(d,t)
  =
  \operatorname{tf}(d,t)\operatorname{idf}(t)
  \]

  with smoothed inverse document frequency:

  \[
  \operatorname{idf}(t)
  =
  \log\left(
  \frac{1+n_d}{1+\operatorname{df}(d,t)}
  \right)+1
  \]

  where \(n_d\) is the number of documents and \(\operatorname{df}(d,t)\) is the number of documents containing token \(t\) (PDF p. 473).

- Bayes' theorem for class \(c\) and observed feature vector \(x\) is:

  \[
  P(c\mid x)
  =
  \frac{P(x\mid c)P(c)}{P(x)}
  \]

  (PDF pp. 478-479).

- Under the naive conditional-independence assumption for tokens \(x_1,\ldots,x_m\):

  \[
  P(c\mid x_1,\ldots,x_m)
  \propto
  P(c)\prod_{j=1}^{m}P(x_j\mid c)
  \]

  The rendered “send money now” example shows this factorization explicitly; the denominator is common across candidate classes when only their ranking is required (PDF p. 479).

## Assumptions

- Tokenization, normalization, vocabulary filtering, and n-gram choices preserve task-relevant meaning rather than removing signal (PDF pp. 462-465, 477).
- The retained corpus is representative of the language, entities, sources, and time period on which the model will operate.
- Bag-of-words counts are sufficient even though they omit most word order, syntax, negation scope, and wider context (PDF pp. 471-475).
- Cosine proximity in the chosen token-weight space corresponds to the intended notion of semantic similarity (PDF pp. 472-475).
- TF-IDF's corpus-level rarity is a useful proxy for token relevance (PDF pp. 473, 476-477).
- Naive Bayes assumes token features are conditionally independent given the class; correlated phrases or interactions violate this simplifying assumption (PDF pp. 478-479).
- Generic lexicon sentiment transfers to the target domain. The chapter's Twitter comparison illustrates that a task-trained model can outperform a generic TextBlob score (PDF pp. 480-481).
- Classification labels accurately encode the desired outcome. For trading, a positive-language label is not equivalent to a positive, executable subsequent return (PDF pp. 464-465, 480).

## Procedures described by the chapter

### NLP feature pipeline

1. Define the source document and the unit of analysis.
2. Parse and tokenize the text.
3. Normalize case and optionally apply stemming or lemmatization.
4. Add linguistic annotations such as sentence boundaries, POS tags, and dependencies.
5. Add semantic annotations such as named entities.
6. Choose a vocabulary, stop-word policy, n-gram range, and document-frequency limits.
7. Construct a sparse DTM using binary presence, counts, or TF-IDF weights.
8. Add labels and structured enrichment data.
9. Fit a classifier and evaluate it on untouched documents (PDF pp. 462-477).

### News-classification example

1. Load 2,225 BBC articles from five categories.
2. Split documents into stratified training and test subsets.
3. Fit `CountVectorizer` only on training text.
4. Transform the test documents with the frozen training vocabulary.
5. Fit multinomial naive Bayes and score test accuracy; the reported result is about 97.7 percent (PDF pp. 467-480).

### Sentiment examples

1. Fit a sparse multinomial naive Bayes model to approximately 1.57 million labeled tweets using a 934-token vocabulary.
2. Compare its test predictions with TextBlob sentiment. The custom model reports about 77.7 percent accuracy and AUC 0.848 versus TextBlob AUC 0.825 (PDF pp. 480-481).
3. For Yelp, use reviews through 2017 for training and 2018 reviews for testing.
4. Combine sparse text vectors with encoded structured features without densifying the matrix.
5. Compare a majority-class benchmark, naive Bayes, multinomial logistic regression, and LightGBM (PDF pp. 482-484).

## Feature and model examples

- spaCy supplies tokens, lemmas, POS tags, detailed tags, dependencies, shape, alphabetic/stop-word flags, sentences, and entities (PDF pp. 465-470).
- Textacy adds convenient entity and n-gram access on top of spaCy; TextBlob supplies stemming integration and lexicon polarity/subjectivity (PDF pp. 465, 468-471).
- `CountVectorizer` supports binary or count-valued sparse matrices; vocabulary controls include stop words, n-gram range, case normalization, document-frequency thresholds, and a maximum feature count (PDF pp. 473-476).
- `TfidfVectorizer` combines token counting, smoothed IDF, optional sublinear TF, and vector normalization (PDF pp. 476-477).
- The BBC binary DTM contains 2,225 rows, 29,275 columns, and less than 0.7 percent nonzero entries, illustrating why sparse storage matters (PDF p. 474).
- Yelp's majority-class accuracy is about 0.520; text-only naive Bayes reaches about 0.647 and about 0.671 after structured features are added. Regularized multinomial logistic regression exceeds 0.74, while the reported default LightGBM model reaches 0.736 (PDF pp. 482-484).

## Statistical, validation, and backtesting warnings

- Vocabulary learning is model fitting. Building the vocabulary, IDF, frequency thresholds, n-grams, or normalization rules on all documents leaks held-out corpus statistics.
- Near-duplicate, syndicated, quoted, or revised documents can cross folds and inflate apparent generalization.
- Random document splits are not suitable evidence for a chronological trading claim. The BBC and Twitter examples teach classification mechanics, not market-valid evaluation (PDF pp. 479-481).
- Generic topic or sentiment labels may be weak proxies for returns. Market reaction depends on expectations, novelty, entity, horizon, release timing, and what was already priced.
- Lexicon scores can mishandle negation, sarcasm, evolving language, finance-specific meanings, and entity-specific context (PDF pp. 462, 471, 480-481).
- Accuracy can conceal class imbalance, confidence miscalibration, or asymmetric economic errors. The Yelp example's 52 percent majority benchmark demonstrates the need for a declared baseline (PDF pp. 482-484).
- The Yelp 2018 test is temporally cleaner than a random split, but reviews and star labels do not validate a financial signal.
- Comparing many tokenization choices, frequency thresholds, n-grams, classifiers, regularization values, and label definitions is multiple testing.
- A classifier score says nothing about tradability until linked point-in-time to an asset, decision timestamp, executable entry, holding horizon, cost model, and position rule.

## Implementation patterns worth preserving

- Preserve the immutable raw document, canonical source identifier, source timestamp, first-seen timestamp, ingestion timestamp, revision identifier, language, and content hash.
- Keep document acquisition, normalization, annotation, vectorization, labeling, model fitting, and market alignment as separate stages.
- Deduplicate before splitting and record the exact duplicate/near-duplicate policy.
- Fit the vocabulary and all corpus statistics inside each Development training fold; transform later documents without refitting.
- Serialize the tokenizer configuration, stop-word list, vocabulary order, n-gram range, document-frequency thresholds, IDF vector, model, and software versions.
- Use sparse matrices end to end and guard against accidental densification.
- Record unknown-token rates and vocabulary drift through time.
- Evaluate a simple count/TF-IDF linear baseline before adding context-heavy models.
- Store model outputs separately from price data and join through an auditable point-in-time availability rule.

## Dated APIs and examples

The examples reflect the book's software environment and should be treated as conceptual patterns until current APIs are pinned and verified.

- `spacy.load('en')`, model linking, `Doc.is_parsed`, `nlp.pipe(..., n_threads=...)`, and several attribute assumptions on PDF pp. 465-470 predate current spaCy packaging and pipeline APIs.
- Textacy entity/n-gram helpers and TextBlob/Pattern resources require current model and lexicon version checks (PDF pp. 468-471).
- scikit-learn's `get_feature_names()`, vectorizer defaults, logistic-regression multiclass options, and solver behavior have changed across releases (PDF pp. 473-484).
- The LightGBM `early_stopping_rounds` call shown on PDF p. 484 may require a callback or different placement in a current release.
- `np.uint`, pandas display/output conventions, and joblib persistence examples should not be copied without a pinned environment.
- The 2009 Twitter, 2010-2018 Yelp, and BBC corpora are pedagogical datasets, not current market-data feeds.

## Project 1 relevance

- Project 1's current reproducible feature research is based on one-minute GC market data; no governed point-in-time text corpus is part of the approved input set. Consequently, this chapter does not justify adding sentiment columns to the new notebook.
- The most relevant contribution is data lineage. A future text feature needs stronger availability metadata than an OHLCV bar because publication, revision, vendor arrival, and entity-resolution times can differ.
- Classical count/TF-IDF plus regularized linear models would be the appropriate first text benchmark if an approved corpus is later added.
- Text-model fitting must remain inside Development folds, and any predicted sentiment must be frozen before it is joined to decision bars.
- A future text hypothesis must demonstrate incremental value over the existing causal statistical features and anchor/ridge benchmarks on identical eligible observations.

## One-minute GC adaptation

Potential deferred hypothesis:

> Point-in-time sentiment or event-category scores from an approved, deduplicated, timestamp-audited macro/commodities news feed may add incremental information about short-horizon GC expansion or direction beyond market-only features.

Prerequisites and adaptation requirements:

- Do not implement the hypothesis until a legally usable source, historical coverage, source timezone, first-seen timestamp, correction policy, and retention policy are documented.
- Map each document to GC through a predeclared entity/topic rule; do not use subsequent market reaction to decide relevance.
- Make a score available only after the recorded vendor-arrival time plus a realistic processing latency.
- Deduplicate syndicated and revised documents before chronological splitting.
- Fit tokenizer, vocabulary, IDF, labels, and model only in Development walk-forward training windows.
- Aggregate multiple documents within a minute using a predeclared rule and retain age, count, and missingness indicators.
- Keep London and New York diagnostics separate because news intensity and macro-release structure differ by session.
- Compare text-plus-market features against the same market-only baseline using identical rows, horizons, purging/embargo, costs, and decision timing.

## Unsuitable or deferred ideas

- Adding TextBlob scores without an approved, point-in-time corpus and domain validation.
- Treating article publication dates as precise tradable availability timestamps.
- Training on the full corpus before splitting because vectorization is “only preprocessing.”
- Randomly splitting overlapping or near-duplicate news documents.
- Using tweet/review accuracy from the chapter as a prior for GC profitability.
- Tuning vocabulary, n-grams, labels, and classifiers against Validation or Final test.
- Densifying a production-scale DTM or embedding raw copyrighted text in reproducibility artifacts.
- Mixing text acquisition into the statistical-feature notebook before the market-only baseline is frozen and reproducible.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Chapter scope, text-data challenges, and NLP workflow | 461-463 |
| Tokenization, vocabulary, linguistic/semantic annotation, labels, and trading applications | 463-465 |
| spaCy/textacy pipeline, token attributes, dependencies, batching, entities, and n-grams | 465-469 |
| Multilingual processing and TextBlob | 469-471 |
| Bag of words, document-term matrix, and cosine similarity | 471-473 |
| CountVectorizer, sparse vocabulary diagnostics, and document similarity | 473-475 |
| TF-IDF computation, smoothing, summarization, and preprocessing choices | 476-477 |
| Naive Bayes derivation and conditional-independence assumption | 477-479 |
| BBC news classification and introduction to sentiment analysis | 479-480 |
| Twitter sentiment and comparison with TextBlob | 480-481 |
| Yelp temporal split, sparse structured features, and benchmarks | 482-483 |
| Logistic regression, LightGBM, model comparison, and summary | 483-485 |

## Unresolved ambiguities and extraction confidence

- **Confidence: high** for the workflow, library examples, DTM/TF-IDF definitions, classifier procedures, and reported metrics; text extraction was clear over PDF pp. 461-485.
- **Confidence: high** for the formulas transcribed above. The actual visually inspected PDF pages were **462, 472, 473, 479, and 484**.
- **Visual inspection pages:** PDF pp. 462, 472, 473, 479, 484.
- PDF p. 462 was inspected to verify the seven-stage NLP workflow diagram from parsing/tokenization through predictive modeling.
- PDF pp. 472-473 were inspected to verify the DTM geometry, cosine-similarity equation, and displayed smoothed-IDF equation.
- PDF p. 479 was inspected to verify the naive Bayes factorization and conditional-independence example.
- PDF p. 484 was inspected to verify that the plotted comparison places regularized logistic regression above the reported default LightGBM, naive Bayes, and majority benchmark.
- **Confidence: medium** for exact current behavior of the named software APIs; the book does not pin a modern environment, language-model artifact, or lexicon checksum.
- The chapter does not specify a complete point-in-time market join, document-revision policy, trading cost model, or return-label construction for its trading applications.
- Project 1 applicability is deferred pending an approved text source. This summary authorizes no text ingestion, sentiment feature, or Final-test exposure.
