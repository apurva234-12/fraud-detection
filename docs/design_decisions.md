# Design Decisions — Phase 0/1

This document records the decisions made in Phase 0/1 and the reasoning behind
them, so they can be defended in a viva/interview and revisited deliberately
(not accidentally) in later phases.

## 1. Core library has zero web/DB dependencies

`core/` imports only `pandas`, `numpy`, `networkx`, `pytz`, `python-dateutil`,
`PyYAML`. No FastAPI, no SQLAlchemy, no psycopg. This means:

- The entire analytical pipeline is testable and demonstrable with `pytest`
  and a plain script — no server, no database, no frontend required.
- Later phases (API, DB) become **adapters around** this library, not a
  rewrite of it. `db/` will convert `core.data.schema.Transaction` objects to
  rows; `api/` will call `core` functions and serialise the results.

## 2. Timestamp convention: IST (Asia/Kolkata)

Locked in per project decision. All timestamps are normalised to
`Asia/Kolkata` inside `core.data.validators._parse_timestamp`:

- A timezone-**aware** input timestamp is converted to IST.
- A timezone-**naive** input timestamp is *assumed to already represent IST*
  and is localised (not shifted) to IST. This assumption is stated explicitly
  here because it is a real design choice with consequences — a naive
  timestamp that was actually recorded in UTC would be silently
  mis-localised. If a real dataset's naive-timestamp convention turns out to
  be UTC, this is the one place to change.

## 3. Validation never silently drops data

`core.data.validators.validate_batch` returns **both** the valid
`Transaction` objects **and** a `rejects` DataFrame with a `row_index`,
`transaction_id` (if available), machine-readable `reasons` (list of
`RejectReason` values), and human-readable `details` for every row that
failed. A row can fail for multiple reasons simultaneously — all of them are
recorded, not just the first.

This was a hard requirement in the project brief and is directly testable:
see `test_multiple_failures_all_collected` and
`test_invalid_rows_never_silently_dropped`.

## 4. Self-transactions: rejected by default, but never silently

`sender_account_id == receiver_account_id` is rejected by default
(`Config.allow_self_transactions = False`) with reason code
`SELF_TRANSACTION`, and is fully configurable. This is a deliberate scope
decision, not an oversight: our graph model treats an edge as *fund
movement between two distinct accounts*, and a self-loop conflicts with graph
metrics like "unique counterparties" and inflow/outflow ratio being
meaningfully distinct. If a real dataset legitimately contains
self-transactions that need investigating (e.g. internal ledger corrections),
flip the config flag rather than changing validator code.

## 5. Duplicate `transaction_id`: first-occurrence-wins

If the same `transaction_id` appears twice in a batch, the first occurrence
is validated normally; every later occurrence is rejected with
`DUPLICATE_ID`, **regardless of whether its content is otherwise valid**.
This is intentional — a duplicate ID is a data-integrity problem regardless
of payload, and "the second one looked fine so we kept it" would hide a bug
in whatever produced the file.

Duplicate **content** with **different** IDs is explicitly *not* treated as
an error at this layer — it may be a legitimate repeated transfer, or it may
itself be a signal worth flagging later (`REPEATED_TRANSACTIONS`-style
behaviour). That judgment belongs to the AML rule engine (a later phase), not
the data-validation layer.

## 6. Graph model: `MultiDiGraph` as source of truth

- **Node** = account (`account_id`).
- **Edge** = one transaction, keyed by `transaction_id`, directed
  sender → receiver, carrying `amount_inr`, `timestamp`, `transaction_type`.
- **Type**: `networkx.MultiDiGraph`. Multiple transactions between the same
  ordered pair are **not** collapsed — each is its own edge. This preserves
  full transaction-level detail; nothing about "how many times" or "when" is
  lost by construction. See `test_repeated_transactions_between_same_pair_create_multiple_edges`.
- **Derived aggregated view**: `core.graph.builder.to_weighted_digraph`
  collapses multi-edges into a single weighted `DiGraph` edge (sum of
  amounts, plus a separate `txn_count`) for metrics that need one edge per
  pair. The `MultiDiGraph` remains authoritative; the `DiGraph` is always
  *derived*, never edited directly.
- **Isolated accounts**: the graph builder does not invent isolated nodes —
  it only has transaction data to work from. An account with zero
  transactions in the window simply never becomes a node unless a caller
  explicitly passes it via `extra_isolated_accounts` (e.g. because it exists
  in an accounts table with no activity yet). This keeps the builder honest
  about what it actually knows.

## 7. Graph metrics implemented in Phase 1 (and what's deferred)

Implemented, each with a one-sentence justification:

| Metric | Why |
|---|---|
| in/out-degree | Raw transaction count in/out — cheapest possible connectivity signal. |
| weighted in/out-degree | Total amount in/out — degree alone hides a `1×₹10L` transfer vs `100×₹100`. |
| unique counterparties (in/out/total) | Distinguishes "20 transactions with 1 person" from "20 transactions with 20 people" — the latter is structurally different (see `test_unique_counterparties_vs_degree_with_repeated_edges`). |
| inflow/outflow ratio | Cheap, interpretable pass-through/sink/source signal. |

**Deliberately deferred** to the graph-analytics phase (Phase 4):
**betweenness centrality** and **PageRank**. Both are more expensive to
compute and need a specific justification tied to real graph structure
(intermediary/layering detection) that is easier to validate once we have
graph-analytics-specific test scenarios (`PASS_THROUGH`,
`LAYERED_MULTI_HOP`) already in place from the synthetic generator — which
they now are, ready for Phase 4.

## 8. Anomaly-score semantics for `inflow_outflow_ratio`

`inflow / outflow`, with two edge cases made explicit rather than left as
`NaN`:

- `outflow == 0` and `inflow > 0` → `float('inf')` (pure sink account).
- `outflow == 0` and `inflow == 0` → `0.0` (isolated node has no ratio, but
  the account-features table never contains a fully-inactive account by
  construction, so this only occurs at the graph-metrics layer for
  explicitly-added isolated nodes).

This is a real design decision worth defending: `inf` is semantically
correct ("infinitely more in than out") and trivially filterable
(`df[df.inflow_outflow_ratio == float('inf')]`), whereas silently using
`NaN` or `0` for a pure-sink account would be misleading.

## 9. Synthetic generator: scenarios are structural, not schema violations

Every injected scenario (`HIGH_VELOCITY`, `LARGE_TRANSFER`,
`REPEATED_TRANSACTIONS`, `HIGH_FANOUT`, `PASS_THROUGH`, `CIRCULAR_FLOW`,
`LAYERED_MULTI_HOP`, `BEHAVIOURAL_CHANGE`) produces ordinary, **schema-valid**
`Transaction` objects — see `test_all_generated_transactions_pass_validation`.
The scenarios encode *behavioural patterns* (timing, amount, fan-out,
topology) an anomaly/AML pipeline should later be able to distinguish from
`NORMAL` baseline activity — they do not encode malformed data. Malformed
data is tested separately and directly against the validators
(`tests/unit/data/test_validators.py`), independent of the generator.

Scenario labels (`transaction_scenarios`, `account_scenarios`) are returned
in a structure **separate from `Transaction`** and are never joined into
`core.data.schema`, `core.features`, or `core.graph` output — see
`test_scenario_labels_not_a_transaction_field`. This is the mechanism that
prevents them from ever being mistaken for, or leaking into, real fraud
labels.

**Explicit disclaimer, repeated here for the record**: these synthetic
scenarios are controlled academic test patterns designed to exercise the
pipeline. They are not derived from real fraud data and must never be
described as representative of real-world fraud in any report, viva, or
interview discussion of this project.

Reproducibility: the entire generator is driven by one
`numpy.random.default_rng(config.random_seed)` instance; same seed → 
byte-identical output (`test_fixture_dataset_reproducible`).

## 10. What was intentionally NOT built in Phase 0/1

Per explicit instruction: no FastAPI, no PostgreSQL, no React, no
authentication, no Docker, no anomaly detection, no AML rules, no risk
scoring. `core.features` stops at directly-interpretable, non-graph,
non-anomaly behavioural statistics (counts, sums, ratios, a simple
same-day-count velocity proxy) — nothing here presumes what "suspicious"
means, because that judgment belongs to later phases.
