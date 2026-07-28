# Chapter 2 - Market and Fundamental Data: Sources and Techniques

**Assigned source range:** PDF pages 59-94  
**Source identity:** supplied second-edition PDF; SHA-256 is recorded in `../mlat_book_map.md`  
**Extraction procedure:** sequential text extraction plus rendered-page checks; see `../README.md`  
**Stage:** 1 - book digestion only; no Project 1 feature or data source is approved by this note.

## Main argument

Financial data are produced by market institutions, reporting rules, and vendor transformations. A researcher therefore needs to understand how orders become quotes and trades, how irregular events become bars, when fundamental information became public, and what information was discarded before treating a dataset as model-ready. The chapter develops this argument from exchange market structure through NASDAQ TotalView-ITCH order messages, order-book reconstruction, bar sampling, consolidated minute data, remote data APIs, SEC XBRL filings, and storage choices (PDF pp. 59-94).

The practical lesson is that a clean table is not the same thing as a faithful historical observation. Quote fields can be carried forward, trade and quote OHLC values can describe different event streams, bars can conceal order splitting and within-bar paths, and a fundamental value can become usable only after its filing time. Reproducible research must preserve the source's semantics, event time, availability time, corporate-action treatment, and transformation history (PDF pp. 59-61, 69-82, 88-94).

## Concepts and definitions

- **Market microstructure:** the institutional and operational rules through which trading interest becomes orders, quotes, transactions, and prices. Venue design and order rules determine what the resulting market data mean (PDF pp. 59-65).
- **Market order:** an instruction to transact immediately at the best price available, accepting price uncertainty in exchange for execution priority (PDF pp. 62-63).
- **Limit order:** an instruction to buy no higher than, or sell no lower than, a specified price. It controls price but does not guarantee execution (PDF pp. 62-63).
- **Stop order:** an instruction that becomes active after a trigger price is reached; its execution behavior then depends on the order specification and market conditions (PDF p. 63).
- **Exchange and over-the-counter market:** an exchange centralizes trading under venue rules, whereas an OTC market relies on a dealer network rather than a single centralized exchange (PDF pp. 60-62).
- **Alternative trading system, electronic communication network, and dark pool:** electronic venues that match orders outside a traditional exchange structure. Dark pools limit pre-trade transparency and may report trades only after execution (PDF pp. 60-62).
- **National Best Bid and Offer (NBBO):** the best displayed bid and offer consolidated across US equity venues. The Securities Information Processor aggregates venue feeds for public dissemination, while proprietary direct feeds can be faster or more detailed (PDF pp. 64-65, 76-79).
- **Bid-ask spread:** the difference between the best offer and best bid. Alternating executions at bid and ask can create bid-ask bounce even if the underlying value is unchanged (PDF pp. 63-64, 69).
- **Level 1, Level 2, and Level 3 market data:** increasing levels of order-book detail. Level 1 supplies best prices and associated size; Level 2 adds price levels and market depth; Level 3 exposes individual order information where available (PDF pp. 64-65).
- **FIX and native exchange protocol:** FIX is an industry messaging standard used for order and trade communication; native feeds such as NASDAQ TotalView-ITCH encode venue-specific event detail and sequence (PDF pp. 65-67).
- **Order book:** the current set of active buy and sell limit orders. **Depth** describes the available quantity away from or at the best prices (PDF pp. 67-72).
- **Net Order Imbalance Indicator (NOII):** NASDAQ auction information about paired shares, imbalance direction and quantity, and indicative clearing prices (PDF pp. 66-67).
- **Tick data:** irregularly spaced market events such as quote changes or trades. A tick is an event, not a fixed unit of clock time (PDF pp. 59, 68-72).
- **OHLCV:** open, high, low, close, and volume summaries over an interval or event bucket. A bar records aggregates and loses the exact within-bar event path (PDF pp. 68-74).
- **VWAP:** volume-weighted average price, which weights each transaction price by its traded quantity (PDF pp. 68-69).
- **Time, tick, volume, and dollar bars:** alternative ways to regularize events. Time bars close at fixed clock intervals; tick bars after an event count; volume bars after a quantity threshold; dollar bars after a notional-value threshold (PDF pp. 69-74).
- **Carried-forward quote:** a quote repeated into a minute record when no new qualifying NBBO event occurs. Its timestamped row does not imply that the quote was newly observed in that minute (PDF pp. 76-80).
- **XBRL:** a structured business-reporting format whose taxonomy identifies financial concepts and relationships so filings can be processed programmatically (PDF pp. 87-93).
- **CIK and accession number:** the SEC's company identifier and a filing-specific identifier used to locate submissions and filing facts (PDF pp. 88-90).
- **Point-in-time fundamental data:** accounting information aligned to the time at which a filing became publicly available, not merely to the fiscal period it describes (PDF pp. 87-93).
- **Parquet and HDF5:** binary columnar or hierarchical storage formats that can preserve dtypes and offer substantially better analytical I/O than repeated CSV parsing (PDF pp. 80-86, 92-94).

## Formulas and notation

For transactions indexed by \(i\), with price \(p_i\) and traded shares or contracts \(q_i\), the chapter's VWAP calculation is:

\[
\mathrm{VWAP}=\frac{\sum_i p_iq_i}{\sum_i q_i}
\]

(PDF pp. 68-69).

The activity accumulated for a dollar bar is transaction notional:

\[
d_i=p_iq_i
\]

and the bar closes when cumulative notional reaches the selected threshold (PDF pp. 72-74).

The Apple fundamental-data example constructs a trailing price-to-earnings ratio from an adjusted price and trailing twelve-month diluted earnings per share:

\[
\mathrm{P/E}_t=
\frac{\mathrm{Adjusted\ Price}_t}
{\mathrm{TTM\ Diluted\ EPS}_t}
\]

where trailing EPS is assembled from the most recent four quarterly values after the example's split adjustment (PDF pp. 90-92).

The chapter also uses cumulative event counts, share volume, and dollar value as bar-closing rules, but it does not provide a universal statistical formula for choosing their thresholds. No unprinted microstructure or bar-sampling equations are reconstructed here.

## Assumptions

- Feed messages are complete, correctly sequenced, and decoded with the exact exchange protocol version in force.
- Venue timestamps, receive timestamps, and local processing times are not silently conflated.
- Order identifiers and event types are sufficient to add, replace, cancel, and execute orders without corrupting reconstructed state.
- A vendor's trade and quote conditions, correction logic, and consolidation rules are understood before aggregation.
- Corporate actions, contract changes, symbol mappings, and exchange calendars are handled consistently.
- Bar thresholds are selected using information available before the modeled decision and are not optimized on the full sample.
- A field carried into a minute is identified as stale or last-observation-carried-forward rather than treated as a new event.
- Fundamental facts are joined using their public availability date and time, including amendments or restatements where relevant.
- Storage conversion preserves precision, timezone, missing-value semantics, sort order, and schema.
- Vendor APIs and public endpoints supply the historical coverage the analysis assumes, subject to licensing and access limits.

## Procedures described by the chapter

### Reconstruct an order book from NASDAQ ITCH

1. Obtain the TotalView-ITCH binary event file and the matching message specification.
2. Read each message header to determine its type and byte length.
3. Decode the message with type-specific binary formats.
4. retain relevant system events, stock-directory records, order additions, replacements, cancellations, deletions, executions, and trade messages.
5. Use order reference numbers to maintain active buy and sell orders.
6. Apply messages in sequence so additions, size reductions, replacements, and executions update the correct prior state.
7. Store trades and selected order-book snapshots or depth summaries for downstream aggregation.
8. Validate message counts, stock mappings, and reconstructed quantities before calculating features (PDF pp. 65-72; the message-layout table on p. 66 was visually inspected).

### Convert irregular events to bars

1. Select the target instrument and trading-session rules.
2. Sort qualifying transactions by event time and apply corrections or condition filters.
3. For time bars, group trades by fixed clock interval.
4. For tick, volume, or dollar bars, accumulate event count, quantity, or notional until a predeclared threshold is crossed.
5. Compute open from the first eligible trade, high and low over eligible trades, close from the last, total volume, and VWAP.
6. Preserve bar start/end time, event count, and any threshold overshoot so the transformation can be audited.
7. Compare distributions and information loss across sampling methods rather than assuming one bar type is inherently superior (PDF pp. 68-74; p. 72 was visually inspected).

### Process consolidated minute quote and trade data

1. Read the vendor schema and distinguish quote-derived fields from trade-derived fields.
2. Convert each daily compressed source file into a typed columnar representation.
3. retain the exchange timestamp, ticker, quote OHLC, trade OHLC, sizes, volume, and vendor condition fields required by the analysis.
4. Record when NBBO values are carried forward because a minute contains no new quote.
5. Write partitioned daily Parquet files for efficient filtering, then combine only when a downstream task requires a broader HDF5 or table view.
6. verify that the resulting minute count, date coverage, symbols, dtypes, and missingness match the source documentation (PDF pp. 75-82; the schema display on p. 78 was visually inspected).

### Retrieve market data through APIs

The examples demonstrate `pandas-datareader`, Quandl/Nasdaq Data Link, and `yfinance`-style access for economic, market, and security data. The transferable procedure is to pin the library and endpoint version, cache immutable raw responses, store query parameters and retrieval dates, validate adjustment semantics, and convert the response into a locally versioned analytical dataset (PDF pp. 82-87).

### Build a point-in-time fundamental series from SEC data

1. Download the relevant SEC Financial Statement Data Set archive for each period.
2. use the submission table to identify the company and filing accession number.
3. use the tag, number, presentation, and calculation tables to identify the desired accounting concept and context.
4. select the intended statement, fiscal period, units, and diluted EPS observations.
5. account for stock splits consistently with the price series.
6. construct trailing-twelve-month EPS from four quarterly observations.
7. join the fact to prices only from the filing's public availability onward.
8. preserve filing identity and amendment/restatement lineage rather than overwriting history.
9. calculate the desired ratio and verify suspicious discontinuities against the underlying filing (PDF pp. 87-93; the example code and output on p. 91 were visually inspected).

### Choose a storage format

The chapter compares CSV, HDF5, and Parquet by file size and read/write performance. The general procedure is to preserve an immutable raw layer, use typed partitioned Parquet for interoperable columnar analytics, and consider HDF5 when its indexed table or hierarchical access pattern is specifically useful. Benchmarks must use the actual dataset and access pattern rather than the chapter's results as a universal ranking (PDF pp. 80-86, 92-94).

## Feature and model examples

- Trade OHLC, quote OHLC, traded volume, event count, and VWAP at a chosen bar frequency (PDF pp. 68-82).
- Spread, midpoint, quoted size, depth, and imbalance features when an authenticated quote/order-book source supplies them (PDF pp. 63-72).
- Event-time, volume-time, or dollar-time returns intended to regularize market activity rather than clock time alone (PDF pp. 69-74).
- Auction imbalance and indicative clearing information from NOII messages (PDF pp. 66-67).
- Fundamental valuation features such as trailing P/E built from filing facts and adjusted prices (PDF pp. 87-93).
- Cross-source features combining market observations with economic or security data retrieved through APIs (PDF pp. 82-87).

These are examples of data and transformations, not evidence that any resulting feature predicts returns.

## Statistical, validation, and backtesting warnings

- **Irregular sampling:** ticks cluster in active periods. Treating observations as equally spaced without an explicit event-time interpretation distorts horizons and volatility (PDF pp. 59, 68-74).
- **Bid-ask bounce:** alternating bid- and ask-side executions can create short-horizon return reversal unrelated to a change in economic value (PDF pp. 63-64, 69).
- **Aggregation loss:** identical OHLCV bars can arise from different event paths. A backtest cannot use information that the bar representation discarded (PDF pp. 68-74).
- **Fragmented execution:** a parent order may generate many prints, so tick bars can overweight execution mechanics rather than independent information. Volume and dollar bars mitigate different problems but do not guarantee independence (PDF pp. 69-74).
- **Threshold leakage:** choosing a bar threshold from future activity or optimizing it on the entire sample leaks later market conditions into earlier observations.
- **Quote/trade mismatch:** quote OHLC and trade OHLC summarize different processes. Quote values may be carried forward, and off-exchange trades may arrive under different reporting rules (PDF pp. 75-80).
- **Hidden or delayed activity:** dark-pool and FINRA reporting means displayed depth is not total executable interest and timing may differ from venue activity (PDF pp. 60-65, 75-80).
- **Feed survivorship and correction handling:** omitted broken trades, cancels, replacements, halts, or late corrections can make reconstructed history internally inconsistent.
- **Corporate actions and identifiers:** splits, symbol changes, and delistings can create false returns or valuation jumps if price, volume, and fundamental series use inconsistent adjustment rules (PDF pp. 82-93).
- **Fundamental look-ahead:** fiscal period end is not information availability. Forward-filling a value before its filing timestamp or silently using restated history is look-ahead bias (PDF pp. 87-93).
- **API/vendor bias:** convenient endpoints may change historical adjustments, universe coverage, symbol mapping, or returned fields without preserving prior versions (PDF pp. 75-87).
- **Session and timezone ambiguity:** bars spanning a session boundary, daylight-saving change, or maintenance break can combine economically different intervals.
- **Backtest execution:** a bar's final high, low, close, volume, or VWAP is not known at the bar's opening. Any strategy using completed-bar values must trade no earlier than its declared next decision/execution point.

## Implementation patterns worth preserving

- Keep raw downloads immutable and record checksums, protocol/vendor version, query parameters, retrieval date, and license constraints.
- Separate event time, receipt time, bar-close time, and information-availability time in the schema.
- Parse exchange binary messages with declarative layouts and test each message type against known records.
- Assert monotonic ordering, unique keys where required, valid prices and quantities, and state consistency after each order-book transition.
- Partition large event or minute datasets by stable keys such as date and instrument; avoid one monolithic reparsed CSV.
- Preserve quote and trade fields separately, together with a stale/carried-forward indicator.
- Make session calendar, timezone, adjustment policy, and bar-closing rule explicit configuration.
- Write deterministic bar builders with tests for empty intervals, threshold overshoot, corrected trades, and session resets.
- Join fundamentals by first-public availability and retain filing/accession lineage.
- Cache API responses and pin dependencies so partners do not rely on a live service returning the same history later.
- Produce a compact manifest containing row counts, date range, schema, missingness, and hashes for every analytical artifact.

## Dated APIs and examples

- NASDAQ TotalView-ITCH 5.0 and the sample files are protocol- and period-specific; production decoding must use the exact current or historical specification (PDF pp. 65-72).
- Quantopian, a recurring platform context in the book, has closed. Its hosted data and research environment are not a reproducibility target.
- The chapter's Zipline-related ecosystem reflects the book's publication period and should not be assumed compatible with current Python.
- `pandas-datareader` connectors for Yahoo and Quandl/Nasdaq Data Link have changed availability and authentication behavior since publication (PDF pp. 82-87).
- `yfinance` is an unofficial convenience interface whose syntax, adjustment defaults, and upstream behavior are not a stable data contract (PDF pp. 84-87).
- Older pandas patterns in the examples, including `DataFrame.info(null_counts=...)`, `fillna(method=...)`, and some `HDFStore.append` workflows, need version-specific review.
- Older Bokeh keyword names such as `plot_width` may not match current releases.
- SEC download locations, user-agent requirements, rate limits, archive formats, and taxonomy versions can change. Current SEC documentation must govern a live implementation.
- Vendor schemas and licenses such as AlgoSeek's are external contracts; the book's field list is descriptive of its edition, not authority for a current feed.

## Project 1 relevance

Project 1 already has trusted one-minute Databento CME GC/MGC bars, so NASDAQ ITCH decoding, US-equity NBBO consolidation, SEC XBRL parsing, and equity-data APIs are not required for the current notebook. The directly relevant lesson is provenance: a GC minute bar is a transformed summary of futures activity whose session, timestamp, contract, roll, price, and volume conventions must remain explicit.

The current bars do not contain authenticated bid, ask, individual trades, order identifiers, book depth, or auction messages. Consequently, the notebook must not label bar-derived proxies as true spread, order-flow imbalance, microprice, queue position, or market depth. Such features require a new data source and a separate data-contract review.

The chapter supports several current engineering choices:

- preserve the Parquet analytical layer and its hashes;
- keep GC and MGC identities and contract/roll treatment explicit;
- calculate completed-bar features causally and enter no earlier than the declared next bar;
- reset or flag session boundaries and abnormal gaps;
- retain volume and VWAP semantics rather than assuming they are comparable across all times of day;
- document which source columns are observed versus derived.

Before adding any bar-sampling experiment, the new notebook should audit overlap with the existing statistical feature matrix, which already includes volume- and VWAP-related transformations.

## One-minute GC adaptation

A bounded adaptation is a **sampling-robustness experiment**, not an automatic feature proposal:

1. Start from the existing trusted one-minute GC source and preserve its timestamp, session, and roll metadata.
2. Define causal activity buckets from completed observations only, such as a predeclared cumulative-volume or cumulative-dollar threshold reset at a documented boundary.
3. Map every event bucket back to its bar-close decision time.
4. form labels only after that close and preserve the Project 1 rule of decision at \(t\), entry no earlier than \(t+1\).
5. compare whether an already documented feature keeps its sign, coverage, and stability under clock-time and activity-time sampling.
6. normalize activity for time of day and contract regime where appropriate.
7. treat threshold selection as a train-fold parameter and report incomplete final buckets.

This experiment tests whether an effect is an artifact of clock-time sampling. It does not justify selecting the best threshold over the full sample, synthesizing tick-level order flow from minute OHLCV, or changing the production data contract.

## Unsuitable or deferred ideas

- NASDAQ TotalView-ITCH reconstruction for CME gold futures.
- FIX/native equity order-message features without the corresponding CME market-depth feed.
- NBBO, dark-pool, auction-NOII, queue-position, or individual-order features derived from minute OHLCV.
- SEC XBRL equity fundamentals or stock P/E ratios as direct GC one-minute predictors without a defensible economic and timing hypothesis.
- Cross-sectional stock-universe workflows applied to a single continuous futures series.
- Live dependence on Yahoo, Quandl/Nasdaq Data Link, or other convenience APIs for a reproducible core dataset.
- Intrabar execution claims based solely on bar high, low, or VWAP.
- Replacing the trusted Project 1 source simply because a book example uses a different vendor or format.

## Exact PDF page map

| Topic | PDF pages |
|---|---:|
| Market-data role, trading venues, orders, and market structure | 59-64 |
| FIX, ITCH, message types, and binary feed specification | 65-67 |
| Event parsing, order-book reconstruction, trades, and initial bar construction | 68-72 |
| Time, tick, volume, and dollar bars | 73-74 |
| AlgoSeek consolidated minute quote and trade data | 75-81 |
| Remote market-data APIs and their storage/access examples | 82-86 |
| SEC XBRL fundamentals and the Apple EPS/P-E example | 87-92 |
| CSV, HDF5, and Parquet benchmark | 93 |
| Chapter summary | 94 |

## Unresolved ambiguities and extraction confidence

**Extraction confidence: high for the chapter's arguments, terminology, procedures, and displayed code; medium for treating any example schema or API as current.**

- **Visual inspection pages:** PDF pp. 66, 72, 78, 91.

The assigned range was read sequentially. The following actual PDF pages were also rendered and visually inspected:

- **p. 66:** TotalView-ITCH message-type and binary-layout material.
- **p. 72:** reconstructed order-book/trade output and the transition from ticks to bars.
- **p. 78:** AlgoSeek/NBBO minute-field display and surrounding explanation.
- **p. 91:** Apple diluted-EPS split adjustment, trailing aggregation, and P/E example code/output.

Remaining ambiguities:

- Level 1/2/3 labels are market conventions whose exact fields vary by venue and vendor; the chapter's descriptions should not be treated as a CME feed schema.
- The text example carries quote values into inactive minutes, but a current implementation must confirm the exact vendor rules for stale quotes, crossed markets, corrections, and condition codes.
- The Apple example communicates the transformation mechanics, yet a production point-in-time implementation would need more explicit handling of filing timestamps, amendments, taxonomy changes, and when split information became known.
- The storage benchmark is hardware-, library-, compression-, schema-, and access-pattern-dependent.
- Page extraction occasionally wraps code and table cells across lines. The visually checked pages support the summarized meaning, but code should be taken from the book's companion repository and version-pinned before reuse.
