# Chapter 3 - Alternative Data for Finance: Categories and Use Cases

**Assigned source range:** PDF pages 95-114  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature or data acquisition is approved by this note.

## Main argument

Alternative data can create an informational advantage when it reveals economically relevant activity that conventional market or fundamental data do not contain, or reveals it sooner. Its potential comes from digitized individual behavior, business-process exhaust, and networked sensors, but novelty and scale make reliability, legality, availability time, representativeness, processing cost, and signal decay harder—not easier—to establish (PDF pp. 95-103).

The chapter therefore treats data acquisition as part of quantitative research. It proposes evaluating a dataset's incremental signal, target asset and horizon, cost, capacity, history, frequency, latency, reliability, exclusivity, format, and legal or reputational exposure before investing in a modeling pipeline. Web-scraping examples illustrate how raw HTML or dynamically rendered pages can become structured observations, while also showing why an externally controlled website is a fragile research dependency (PDF pp. 100-114).

## Concepts and definitions

- **Alternative data:** information outside the conventional market-price and reported-fundamental sources, including raw, aggregated, and vendor-processed data. What counts as alternative changes as sources become mainstream (PDF pp. 95-97).
- **Five Vs:** **volume** (scale), **velocity** (generation and delivery speed), **variety** (structured, semi-structured, and unstructured formats), **veracity** (reliability), and **value** (useful information relative to acquisition and processing cost) (PDF p. 96; visually inspected).
- **Informational advantage:** access to useful information not available from traditional sources, or access sufficiently earlier to support a tradable decision (PDF pp. 96-97).
- **Individual-generated data:** social-media posts, searches, reviews, e-commerce activity, app use, and other digital behavior (PDF pp. 97-98).
- **Business-process data:** transaction records and operational exhaust such as card payments, scanner data, supply-chain orders, credit information, and detailed market-flow records (PDF pp. 98-99).
- **Sensor data:** imagery, geolocation, weather, pollution, camera, industrial, and other measurements collected by connected physical devices (PDF pp. 99-100).
- **Company exhaust:** information created as a byproduct of normal operations rather than specifically for investment research (PDF pp. 98-99).
- **Signal incrementality / orthogonality:** the extent to which a dataset contributes information not already contained in existing features or conventional risk premia (PDF pp. 101-102).
- **Signal decay:** loss of predictive value as competitors acquire, process, and trade on the same information (PDF pp. 100, 102-103).
- **Exclusivity:** restricted or difficult access that may slow signal crowding. Exclusivity can be contractual, operational, or a consequence of demanding processing (PDF pp. 100, 102-103).
- **Capacity:** capital that can use a signal before trading impact or other constraints undermine its economics. A low-capacity signal may not repay a costly dataset (PDF p. 102).
- **Latency:** delay caused by collection, aggregation, processing, transmission, or legal constraints between the underlying event and usable delivery (PDF p. 103).
- **Material non-public information (MNPI):** information whose acquisition or use may trigger insider-trading restrictions (PDF p. 102).
- **Personally identifiable information (PII):** data linked to an individual and subject to privacy and data-protection obligations, including the GDPR context cited by the chapter (PDF p. 102).
- **Selection bias:** systematic differences between people or events observed in a dataset and the target population. Voluntary reviews, app panels, and users continuously active throughout a sample are not automatically representative (PDF pp. 98-100, 107).
- **Web scraping:** automated retrieval and parsing of web content into structured data, subject to site behavior, permissions, terms, and law (PDF pp. 107-114).
- **Static versus dynamically rendered content:** a direct HTTP response may contain static HTML, while JavaScript can request important fields after initial load; a browser-capable process may be needed to observe the latter (PDF pp. 107-110).

## Formulas and notation

The assigned chapter does not display a material mathematical formula. It refers to alpha, Sharpe ratio, correlation with conventional risk premia, frequency, latency, and signal half-life conceptually, but does not define them with an equation in this range (PDF pp. 100-103).

Accordingly, no Sharpe-ratio, correlation, or decay equation is reconstructed from memory. Any later Project 1 evaluation must use the project's own explicitly defined metrics and sampling conventions.

## Assumptions

- The measured digital activity maps to a real economic mechanism relevant to the target asset and horizon.
- The source population and its coverage remain sufficiently representative through time.
- Collection starts and stops, panel turnover, device or platform adoption, and vendor methodology changes are observable.
- The data's availability timestamp is known separately from the event or reference timestamp.
- The researcher may lawfully acquire, store, process, and trade on the information.
- The provider has the rights it claims and does not introduce conflicts by trading the same data under undisclosed conditions.
- A historical archive reconstructs what was actually delivered at each past decision time rather than a later revised dataset.
- The dataset's incremental information survives fees, engineering cost, trading costs, latency, and capacity limits.
- The target horizon is consistent with the source's generation frequency and delivery lag.
- Any vendor-derived score is auditable enough to distinguish source information from methodology changes.
- Website structure, selectors, and access behavior are versioned when scraping is used.

## Procedures described by the chapter

### Evaluate an alternative dataset

1. State the economic mechanism, target asset class, investment style, prediction horizon, and expected availability advantage.
2. Determine whether the candidate measures an established risk premium, supplies incremental alpha, or duplicates existing signals.
3. quantify acquisition, compliance, infrastructure, engineering, and ongoing maintenance costs.
4. assess strategy capacity and whether likely economic value can recover those costs.
5. audit history length, observation frequency, population coverage, sampling changes, accuracy, and source continuity.
6. distinguish the underlying event time from reporting, processing, and delivery latency.
7. review MNPI, PII, privacy, licensing, conflict-of-interest, and reputational risks with qualified compliance reviewers.
8. inspect raw and processed methodology, including revisions, aggregation, identifiers, missingness, and vendor model changes.
9. compare the candidate with existing features on strictly out-of-sample data and under realistic availability timestamps.
10. monitor for signal decay, crowding, coverage drift, and source discontinuation after adoption (PDF pp. 100-103).

### Acquire static web content

1. Confirm that collection is permitted by applicable law, contractual terms, licensing, robots/access policies, and project governance.
2. issue an HTTP request with an identified, rate-limited client.
3. store the raw response, retrieval timestamp, URL, headers needed for provenance, and content hash.
4. parse the HTML with a structured parser such as Beautiful Soup.
5. locate elements using stable semantics where possible and extract the target fields.
6. validate row counts, types, missingness, duplicates, and representative examples before using the output (PDF pp. 107-108).

### Acquire dynamically rendered content

1. Inspect whether target fields are absent from the initial response and loaded through JavaScript.
2. prefer an authorized structured API or underlying permitted endpoint when one exists.
3. if browser automation is permitted, use a controlled headless browser and explicit wait conditions.
4. capture the rendered page source and parse it with tested selectors.
5. paginate with termination, retry, throttling, and deduplication rules.
6. close browser resources and persist raw snapshots or sufficient audit evidence.
7. monitor selector and navigation drift because an unchanged script can silently return an incomplete dataset (PDF pp. 108-111).

### Scale a crawler with Scrapy and a renderer

The chapter combines a Scrapy spider with Splash to request rendered pages, apply CSS selectors, yield structured items, and store a crawl log. The transferable design is a crawler with explicit start URLs, permitted-domain rules, deterministic parsing, structured outputs, throttling, retry behavior, logs, and raw-content provenance. The exact OpenTable selectors and Splash setup are historical examples, not current instructions (PDF pp. 111-112; p. 111 was visually inspected).

### Parse earnings-call transcripts

1. enumerate transcript listing pages and identify matching links.
2. retrieve each transcript page and record its source URL.
3. parse company, symbol, call date, fiscal quarter, participant roles, speaker names, prepared remarks, and question-and-answer sections.
4. retain speaker and section boundaries instead of flattening the document into one text field.
5. write structured metadata, participants, and content tables for later natural-language processing.
6. validate parsed dates, ticker mappings, missing speakers, duplicated calls, corrections, and publication time before modeling (PDF pp. 112-114).

## Feature and model examples

- Online product prices as a higher-frequency proxy for inflation (PDF p. 96).
- Store visits, payment transactions, email receipts, app activity, or web traffic as sales and economic-activity proxies (PDF pp. 96-100, 106-107).
- Satellite or drone imagery for crop yield, mine activity, metal storage, construction, shipping, and industrial incidents (PDF pp. 96, 99, 106).
- Smartphone or on-site sensor activity as a measure of visits, dwell time, and repeat traffic (PDF pp. 99-100, 106-107).
- Social, search, news, and review text transformed into sentiment, attention, quality, or momentum-related indicators (PDF pp. 97-98, 105-107).
- Restaurant count, bookings, price class, ratings, reviews, cuisine, and location as inputs to geographic activity or company-level models (PDF pp. 107-112).
- Earnings-call speaker, topic, sentiment, emphasis, and communication-style features (PDF pp. 112-114).
- Weather, pollution, shipping, and commodity-production observations as macro or commodity-relevant inputs (PDF pp. 99-101).

The provider and case descriptions establish plausible uses; they do not provide Project 1 evidence of prediction or economic value.

## Statistical, validation, and backtesting warnings

- **Short histories:** many sources cited have only several years of coverage, making regime diversity and independent test size small (PDF pp. 98-100, 103).
- **Selection bias:** reviewers, social posters, consenting app users, and continuously active panel members differ from the target population (PDF pp. 98-100, 107).
- **Coverage and adoption drift:** smartphone, platform, satellite, and vendor coverage can grow for reasons unrelated to the economic target.
- **Survivorship:** analyzing only sources, locations, companies, users, or identifiers still present at the end of the sample overstates historical continuity.
- **Availability leakage:** event date, scrape date, vendor processing date, and first tradable delivery time are distinct. Backtests must use the last of the required availability steps.
- **Revision leakage:** a current vendor history may contain cleaned, geocoded, reclassified, or backfilled observations unavailable in real time.
- **Vendor-model leakage:** a historical sentiment or signal score may have been recomputed with a later model unless archived vintages are supplied.
- **Multiple testing:** the number of possible sources, cohorts, transformations, geographies, lags, and assets creates a large research search space.
- **Incrementality:** a candidate can correlate with returns but add nothing after existing price, volume, seasonality, or macro features (PDF pp. 101-102).
- **Signal decay and crowding:** broad vendor access and easy processing can shorten the useful life of a public alternative-data effect (PDF pp. 100, 102-103).
- **Frequency mismatch:** high collection frequency is not low delivery latency, and neither guarantees a signal suitable for one-minute trading.
- **Capacity and cost:** weak alpha can be real but uneconomic after data fees, turnover, slippage, and capital limits (PDF pp. 100-103).
- **Image and sensor confounding:** cloud cover, seasonality, holidays, irregular revisit schedules, and sensor changes can masquerade as economic variation (PDF p. 99).
- **Text and social manipulation:** bots, coordinated promotion, duplicated articles, and platform moderation can change measured sentiment independently of fundamentals.
- **Scraper breakage:** layout or class-name changes can yield missing or misclassified data without an obvious program failure (PDF p. 111).
- **Legal and reputational risk:** MNPI, PII, consent, licensing, terms of service, and provider conflicts must be resolved before experimentation, not after a promising backtest (PDF p. 102).

## Implementation patterns worth preserving

- Create a data contract covering owner, license, collection authority, target population, units, identifiers, event time, first-availability time, revisions, and retention.
- Preserve immutable raw payloads or legally permissible audit snapshots with retrieval metadata and hashes.
- Version parser code, selectors, vendor schemas, taxonomies, and any vendor/model methodology.
- Use fixture-based parser tests from approved archived pages rather than testing only against a live site.
- Log requests, responses, pagination state, retries, throttling, parser version, row counts, and rejected records.
- Implement explicit completeness checks so an empty selector result cannot silently become a valid zero.
- Maintain vintage datasets when records can be revised or backfilled.
- Separate observed fields, provider-derived scores, and in-house transformations.
- Track coverage by time, geography, entity, device, and source panel; expose breaks as diagnostics.
- Align external identifiers to Project 1 instruments through a documented mapping with effective dates.
- Pin browser, driver, parser, and crawler versions when browser automation is genuinely required.
- Cache all permitted input needed for a partner to rerun the notebook without contacting an unstable live website.

## Dated APIs and examples

- Market-size, spending, provider-count, adoption, valuation, and coverage statements are snapshots from approximately 2017-2020 and must not be treated as current market facts (PDF pp. 95-106).
- Twitter-era names, access arrangements, Gnip, StockTwits, Dataminr, RavenPack, RS Metrics, Advan, Eagle Alpha, AlternativeData.org, and the profiled coverage can all change (PDF pp. 104-107).
- The printed Dataminr paragraph reports an additional \$391 million in funding but then states total funding as “\$569 billion”; that scale is internally inconsistent and should not be silently corrected without an authoritative source (PDF p. 106).
- The OpenTable URL, page layout, CSS classes, pagination, availability fields, and sample results reflect an early-2020 page and are intentionally fragile (PDF pp. 107-111).
- `webdriver.Firefox()` without explicit options/service management and `find_element_by_link_text` reflect an older Selenium API (PDF pp. 109-110).
- `time.sleep(1)` is not a robust readiness condition for dynamic pages.
- The Scrapy/Splash stack and selectors are historical examples; current deployment, browser-rendering, and security guidance must govern any approved crawler (PDF pp. 111-112).
- Seeking Alpha listing URLs, transcript markup, access rules, and class names may have changed, and transcript reuse may require a license (PDF pp. 112-114).

## Project 1 relevance

The chapter is relevant mainly as a **candidate-data governance framework**. Project 1's current reproducible base is trusted one-minute CME GC/MGC market data. Introducing any alternative source would expand the data contract, compliance surface, missingness mechanisms, timing model, and partner-reproduction requirements.

Commodity-relevant categories do exist: mine and refinery activity, metal inventories, shipping flows, weather, inflation nowcasts, macro releases, news, and geopolitical text could have an economic relationship with gold. Most are unlikely to update or become tradable every minute. Their natural use would be a timestamped state or event context, evaluated at a horizon consistent with release frequency and latency, rather than a fabricated minute-by-minute stream.

Before any acquisition:

- establish that GC is the target and specify why the source should affect it;
- obtain a legally usable historical vintage with first-delivery timestamps;
- test whether the source adds information beyond existing returns, volatility, volume, VWAP, time-of-day, and calendar features;
- keep the current trusted source matrix unchanged until the new source passes an isolated reproducibility and leakage audit;
- budget explicit maintenance for vendor and parser drift.

## One-minute GC adaptation

A defensible one-minute adaptation is a **timestamped external-event gating experiment**:

1. Select one legally cleared commodity- or macro-relevant source with an archived first-public or first-delivery timestamp.
2. define a pre-registered economic hypothesis, expected sign or regime interaction, and maximum relevance window.
3. map an event to the first completed GC minute at which every required field was available.
4. create only causal elapsed-time, event-presence, surprise, or state variables; do not assign the value to earlier minutes.
5. keep decision time \(t\) and entry no earlier than \(t+1\).
6. compare the incremental feature against a baseline containing existing GC statistical features and time-of-day controls.
7. use purged temporal folds, a final untouched holdout, multiplicity control, and explicit coverage/latency diagnostics.
8. report results by event type and regime, including minutes with no observation and sources that failed.

This design evaluates whether sparse external information changes the conditional behavior of already trusted one-minute features. It does not assume that low-frequency alternative data can independently forecast every next-minute return.

## Unsuitable or deferred ideas

- Scraping current OpenTable restaurant bookings as a direct next-minute GC predictor.
- Seeking Alpha equity-call transcripts without a specific, timestamped gold-market mechanism.
- US retail foot traffic, stock-specific app usage, or corporate card-spend panels as generic GC features.
- Treating a present-day scraped archive as if its pages and values existed historically.
- Inferring minute-resolution signal timing from daily, weekly, irregular, or vendor-batched observations.
- Using PII, MNPI, ambiguous-consent location data, or content without collection and trading authority.
- Buying a processed vendor score whose historical vintages and methodology changes cannot be audited.
- Selecting a source because it is novel, exclusive, expensive, or difficult to process rather than because it is incrementally predictive out of sample.
- Building the core partner-reproducible notebook around a live, changing website.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Chapter introduction and alternative-data scope | 95 |
| Alternative-data revolution, five Vs, and investment uses | 96-97 |
| Individual- and business-process-generated data | 98 |
| Sensors, satellites, geolocation, and evaluation transition | 99-100 |
| Signal, cost, capacity, quality, law, exclusivity, history, frequency, reliability, latency, and format criteria | 101-103 |
| Alternative-data market snapshot | 104 |
| Provider categories, named use cases, and OpenTable transition | 105-107 |
| Requests, Beautiful Soup, Selenium, and OpenTable dataset construction | 108-110 |
| OpenTable results and Scrapy/Splash workflow | 111-112 |
| Earnings-call transcript collection, parsing, and chapter summary | 113-114 |

## Unresolved ambiguities and extraction confidence

**Extraction confidence: high for the conceptual framework and workflow; medium for tabular/chart details and historical web code; low for present-day validity of provider facts, URLs, selectors, or access arrangements.**

- **Visual inspection pages:** PDF pp. 96, 104, 105, 111.

The assigned range was read sequentially. The following actual PDF pages were also rendered and visually inspected:

- **p. 96:** the five-V definitions and example alternative-data use cases.
- **p. 104:** Figure 3.1 on reported 2017 usefulness and adoption.
- **p. 105:** the provider-category/count table.
- **p. 111:** Figure 3.2 and the Scrapy/Splash transition and code.

Remaining ambiguities:

- Percentages and provider counts are survey/catalog snapshots with category definitions that are not fully specified in the chapter.
- Statements about historical signal correlation or profitable published strategies are summaries of cited work, not independently reproduced evidence in this chapter.
- The Dataminr total-funding amount on p. 106 appears internally inconsistent; this note preserves the ambiguity instead of guessing a correction.
- The rendered p. 96 contains unusual punctuation around “images” in the variety bullet; it does not change the underlying definition.
- Text extraction wraps code, selectors, and punctuation across lines. The summary captures the workflow, but exact historical code should be obtained from the companion repository and still would require modernization.
- The book presents web scraping as a technical demonstration. It does not provide sufficient current legal authority, terms-of-service analysis, privacy review, or license for any named site.
