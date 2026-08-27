"""
Baseline per-account behavioural features computed directly from validated
transaction data.

Scope decision (Phase 1): this module computes only features that are
directly interpretable from raw transaction data alone, with NO graph
dependency (see `core.graph.metrics` for network-structure features) and NO
anomaly/AML logic. Every feature here should be explainable in one sentence
to a non-technical reviewer -- that is the bar for inclusion in this phase.

Each account that appears as a sender and/or receiver in the input gets
exactly one row. An account with no transactions at all never appears here
(there is nothing to compute) -- that is a graph-level "isolated account"
concern, not a features concern; see `core.graph.builder`.
"""

from __future__ import annotations

import pandas as pd

from core.config import Config


ACCOUNT_FEATURE_COLUMNS: tuple[str, ...] = (
    "account_id",
    "txn_count_sent",
    "txn_count_received",
    "total_amount_sent",
    "total_amount_received",
    "avg_amount_sent",
    "avg_amount_received",
    "unique_counterparties_sent",
    "unique_counterparties_received",
    "inflow_outflow_ratio",
    "first_seen",
    "last_seen",
    "active_days",
    "max_daily_txn_count",
)


def compute_account_features(transactions: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute one feature row per account from a canonical transaction
    DataFrame (as produced by `core.data.schema.transactions_to_dataframe`).

    Returns an empty, correctly-columned DataFrame for an empty input --
    callers should not need to special-case "no transactions".
    """
    if transactions.empty:
        return pd.DataFrame(columns=ACCOUNT_FEATURE_COLUMNS)

    sent = transactions.groupby("sender_account_id").agg(
        txn_count_sent=("transaction_id", "count"),
        total_amount_sent=("amount_inr", "sum"),
        avg_amount_sent=("amount_inr", "mean"),
        unique_counterparties_sent=("receiver_account_id", "nunique"),
    )

    received = transactions.groupby("receiver_account_id").agg(
        txn_count_received=("transaction_id", "count"),
        total_amount_received=("amount_inr", "sum"),
        avg_amount_received=("amount_inr", "mean"),
        unique_counterparties_received=("sender_account_id", "nunique"),
    )

    # first_seen / last_seen / active_days consider an account's activity as
    # either sender or receiver.
    sender_ts = transactions[["sender_account_id", "timestamp"]].rename(columns={"sender_account_id": "account_id"})
    receiver_ts = transactions[["receiver_account_id", "timestamp"]].rename(
        columns={"receiver_account_id": "account_id"}
    )
    all_ts = pd.concat([sender_ts, receiver_ts], ignore_index=True)
    activity = all_ts.groupby("account_id")["timestamp"].agg(first_seen="min", last_seen="max")
    activity["active_days"] = (activity["last_seen"] - activity["first_seen"]).dt.total_seconds() / 86400.0

    # Velocity proxy: the maximum number of transactions (sent or received)
    # an account is involved in within any single calendar day of the
    # observed window, using the configured velocity window (in whole days).
    all_ts_local = all_ts.copy()
    all_ts_local["day_bucket"] = all_ts_local["timestamp"].dt.floor(f"{config.velocity_window_days}D")
    daily_counts = all_ts_local.groupby(["account_id", "day_bucket"]).size()
    max_daily = daily_counts.groupby("account_id").max().rename("max_daily_txn_count")

    accounts = sorted(set(transactions["sender_account_id"]) | set(transactions["receiver_account_id"]))
    features = pd.DataFrame(index=pd.Index(accounts, name="account_id"))
    features = features.join(sent).join(received).join(activity).join(max_daily)

    numeric_fill_zero = [
        "txn_count_sent",
        "txn_count_received",
        "total_amount_sent",
        "total_amount_received",
        "unique_counterparties_sent",
        "unique_counterparties_received",
        "max_daily_txn_count",
    ]
    for col in numeric_fill_zero:
        features[col] = features[col].fillna(0)

    features["inflow_outflow_ratio"] = features.apply(
        lambda row: _inflow_outflow_ratio(row["total_amount_received"], row["total_amount_sent"]),
        axis=1,
    )

    features = features.reset_index()
    features = features[list(ACCOUNT_FEATURE_COLUMNS)]
    return features


def _inflow_outflow_ratio(total_received: float, total_sent: float) -> float:
    """Inflow/outflow ratio, defined as total_received / total_sent.

    Design decision: when total_sent is 0 (account only ever receives),
    the ratio is undefined by simple division. We represent that case as
    `float('inf')` rather than 0 or NaN, since "infinitely more inflow than
    outflow" is the semantically correct description (pure sink account) and
    is easy to filter for explicitly downstream. An account with zero
    inflow AND zero outflow cannot occur here (see docstring: every row in
    this table has at least one transaction).
    """
    if total_sent == 0:
        return float("inf") if total_received > 0 else 0.0
    return total_received / total_sent
