"""
Transaction graph construction.

Graph model (locked in during architecture review):

- Node = an account (`account_id`).
- Edge = a single transaction flow, directed sender -> receiver.
- Graph type = `networkx.MultiDiGraph`, because multiple transactions between
  the same ordered pair of accounts are common and must NOT be collapsed into
  a single edge -- each transaction is individually preserved with its own
  amount, timestamp, and type as edge attributes. This is the source-of-truth
  graph; aggregated/weighted views for specific metrics are derived from it
  (see `core.graph.metrics`), never the other way round.
- Self-transactions never reach this stage: they are rejected during
  validation (see `core.data.validators`) unless explicitly allowed by
  config, so the graph builder does not need special-case handling for them.
- Isolated accounts (known accounts with no transactions in the window) are
  NOT part of this graph by construction, since the graph is built purely
  from transaction edges. Callers who need isolated accounts represented as
  nodes (e.g. because they exist in an accounts table with zero activity)
  can pass them explicitly via `extra_isolated_accounts`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx
import pandas as pd


def build_transaction_graph(
    transactions: pd.DataFrame,
    extra_isolated_accounts: Optional[Iterable[str]] = None,
) -> nx.MultiDiGraph:
    """Build the transaction-level MultiDiGraph from a canonical transaction
    DataFrame (see `core.data.schema.transactions_to_dataframe`).

    Each row becomes exactly one edge, keyed by `transaction_id`, carrying
    `amount_inr`, `timestamp`, and `transaction_type` as edge attributes.
    """
    graph = nx.MultiDiGraph()

    if extra_isolated_accounts:
        graph.add_nodes_from(extra_isolated_accounts)

    if transactions.empty:
        return graph

    for row in transactions.itertuples(index=False):
        graph.add_edge(
            row.sender_account_id,
            row.receiver_account_id,
            key=row.transaction_id,
            transaction_id=row.transaction_id,
            amount_inr=row.amount_inr,
            timestamp=row.timestamp,
            transaction_type=row.transaction_type,
        )

    return graph


def to_weighted_digraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Derive an aggregated, single-edge-per-pair `DiGraph` view from the
    transaction-level `MultiDiGraph`, for metrics that require a single edge
    weight per (sender, receiver) pair (e.g. some centrality algorithms).

    Edge weight = total amount_inr transferred sender -> receiver across all
    of that pair's transactions. Transaction count is kept as a separate
    `txn_count` attribute so no information is silently lost in the
    aggregation -- callers needing transaction-level detail should use the
    source `MultiDiGraph` instead.
    """
    weighted = nx.DiGraph()
    weighted.add_nodes_from(graph.nodes)

    for u, v, data in graph.edges(data=True):
        if weighted.has_edge(u, v):
            weighted[u][v]["weight"] += data["amount_inr"]
            weighted[u][v]["txn_count"] += 1
        else:
            weighted.add_edge(u, v, weight=data["amount_inr"], txn_count=1)

    return weighted
