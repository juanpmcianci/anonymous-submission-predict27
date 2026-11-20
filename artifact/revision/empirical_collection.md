# Kalshi collection audit

Collected on 28 September 2026, 23:34:39–23:38:22 UTC, under the frozen `revision/empirical_protocol.md`. This report concerns collection and data integrity only. No outcome frequencies, forecasts, losses, or test-set scores were calculated by the collector.

## Collected cohort

The population is the exact UTC-hour closing subsample of `KXBTC15M` and `KXETH15M` contracts in [3 August 2026 00:00 UTC, 28 September 2026 00:00 UTC). The nominal grid contains 2,688 series/hour slots. The API returned **2,650 eligible contracts**, 1,325 per asset, across all 56 close dates.

| Integrity check | Result |
|---|---:|
| Raw candidate metadata records | 10,620 |
| Identical duplicate metadata rows | 0 |
| Candidate exclusions: not exact UTC hour | 7,968 |
| Candidate exclusions: upper endpoint outside interval | 2 |
| Eligible finalized binary markets | 2,650 |
| Nominal slots absent from returned metadata | 38 |
| Unique ticker/event mapping | 2,650 |
| Complete scheduled 900-second windows | 2,650 |
| Canonical one-minute candle rows | 39,750 |
| Full 15-candle windows | 2,650 / 2,650 |
| Exact decimal candle-volume/lifetime-volume matches | 2,650 / 2,650 |
| Raw extra candles outside trading windows | 2,317 |
| Extra candles with nonzero volume | 0 |

The 38 absent slots occur in both assets at 07:00, 08:00 and 09:00 UTC on 6, 13 and 20 August, and at 08:00 and 09:00 UTC on 27 August and 3, 10, 17 and 24 September. All dates are Thursdays. Their cause was not independently established; they are recorded as absent API metadata, not assumed maintenance. No replacement dates or markets were added. The exact slot list is in `code/empirical_data/collection_audit.json`.

Eligible metadata requires binary yes/no resolution, `status=finalized`, a present settlement timestamp, no true `is_provisional` flag, and a 900-second open-to-close window. The API commonly omits the optional provisional flag; the collector treats an absent flag as false and retains the raw metadata. Selection does not prefer either resolution. Candidate records excluded for deterministic time rules are preserved in full alongside the exclusion ledger.

## API request details and two transport corrections

The [official markets documentation](https://docs.kalshi.com/api-reference/market/get-markets) makes close-time filters incompatible with `status=settled`. Requests therefore use `series_ticker`, `min_close_ts`, `max_close_ts`, and `limit=1000`, without a status filter. A one-second lower-bound pad avoids losing the inclusive first scheduled close; local filtering implements the exact half-open date interval. Pagination follows every cursor to an empty terminal cursor, even if a page is short. Twelve metadata requests covered both assets. No repeated cursors or conflicting duplicate tickers were encountered.

The fresh historical-cutoff response placed both market-settlement and trade-created cutoffs at 30 July 2026 00:00 UTC, before the cohort. The collector checks this condition; it refuses to silently use current endpoints if a later cutoff makes historical endpoints necessary.

The [batch-candle endpoint](https://docs.kalshi.com/api-reference/market/batch-get-market-candlesticks) permits 100 tickers and 10,000 candles. An initial 48-market day request failed: the server reported 66,959 requested candles against the 10,000 limit. The cap is calculated over the entire requested time rectangle, including periods outside each contract's short lifetime. The collector now partitions each day using ticker count multiplied by common time-span minutes. This produced **168 successful candle requests**, three per close date. This changes transport only, not the cohort or forecast protocol.

Responses often include one candle after a contract's close. The first unadjusted diagnostic therefore reported only 336 full windows, although all contracts had their 15 valid in-window candles. The canonical file now retains exactly `open_time < end_period_ts <= close_time`, with **no timestamp shifts, interpolation, or invented rows**. All 2,317 removed extra rows remain in the raw responses; their timestamps are listed per ticker in `candle_audit.json`, and all have zero volume. The authoritative audit is `collection_audit.json` and `collection_offline_run.log`; `collection_run.log` retains the earlier unadjusted diagnostic for provenance.

## Bounded execution-level reconciliation

The deterministic trade audit selected the earliest training-close contract, `KXBTC15M-26AUG022000-00`, and its first minute, **2 August 2026 23:45:00–23:46:00 UTC**. Its close is 3 August 00:00 UTC, so it belongs to the frozen training cohort despite opening the previous date.

Two fully paginated [trade requests](https://docs.kalshi.com/api-reference/market/get-trades), with one-second query-boundary padding, returned 1,653 executions. Applying the local interval convention `[23:45:00, 23:46:00)` retained **1,643 unique executions**. Their `count_fp` sum, calculated with decimal arithmetic, was **131,205.35 contracts**, exactly the corresponding candle's `volume_fp`. Ten executions outside the local minute remain in the raw response. There were no duplicate trade IDs. This validates one selected interval, not all execution records in the cohort.

The preserved contract rules refer to the average of 60 CF Benchmarks reference-price observations before the scheduled endpoint. The example first contract uses 20:00 EDT, corresponding to 00:00 UTC the following day. The shared secondary rule describes the final averaging minute. Feature construction must continue to obey the protocol's pre-close horizons and availability buffer; lifetime volumes, settlement prices and final metadata quotes are only audit fields.

## Files, reproducibility and limits

All files are under `code/empirical_data/`:

- `markets.json`: canonical list of 2,650 eligible market objects, retaining source metadata fields.
- `candles.json`: mapping from ticker to its 15 chronologically sorted, in-window source candle objects.
- `candidate_markets.json` and `exclusion_ledger.json`: complete returned candidate metadata and deterministic exclusions.
- `candle_audit.json`, `collection_audit.json`, `trade_audit.json`: per-market window/volume diagnostics, nominal missing slots, and the bounded execution audit.
- `raw/` and `manifest.json`: 183 successful raw JSON responses, with exact URLs, request/retrieval times, byte counts and SHA256 hashes. The requests comprise one cutoff, twelve market pages, 168 candle batches and two trade pages.
- `derived_manifest.json`: byte counts and SHA256 hashes of the canonical files and audit artifacts at handoff.

Raw responses total **42,908,840 bytes**, below the enforced 100,000,000-byte raw-data cap. Requests used unauthenticated public endpoints, TLS verification, throttling, bounded retries, response-size limits and a persistent checksummed cache. No large third-party archive was downloaded. Existing cache files are reused only after checksum verification.

Reconstruct from cache without network access:

```sh
/opt/anaconda3/bin/python3 code/collect_kalshi.py --offline
```

Repeat collection with cache reuse and network requests only for absent URLs:

```sh
/opt/anaconda3/bin/python3 code/collect_kalshi.py
```

Ten collector tests pass, covering UTC/hour/date boundaries, neutral yes/no eligibility and finality, pagination after short pages, repeated-cursor failure, duplicate conflicts, batch coverage/caps, post-close rows, and exact volume checks against the real schema probe. Offline reconstruction succeeded, and all raw checksums and byte counts were independently rechecked. An initial concurrent project-wide test run saw three temporarily unimplemented `GaussianHistoryTests`; these were reported to the parent task. After the other work completed, the final full-project run passed **39 tests**. No model code was changed by the collector task.

The complete scheduled candle grid does not establish predictive feature eligibility: missing prices, zero-volume prefixes and other protocol conditions still need the separate analysis stage. Outcome fields are present in the public metadata and retained for later supervised analysis, but this collection stage did not summarize them or evaluate forecasts. Public API accessibility does not independently establish a licence to redistribute the exchange's complete raw data; this delivery preserves a local research snapshot and retrieval provenance.
