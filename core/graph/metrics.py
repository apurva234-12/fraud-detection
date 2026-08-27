"""
Initial graph metrics.

Scope decision (Phase 1): only the metrics explicitly justified in the
architecture review are implemented here -- in-degree, out-degree, weighted
in-degree, weighted out-degree, unique counterparties, and inflow/outflow
ratio. Betweenness centrality and PageRank are deferred to the graph-analytics
phase (Phase 4): they are more expensive, need a documented justification for
this specific dataset, and are not needed to prove the pipeline works
end-to-end.

All metrics are computed directly from the transaction-level `MultiDiGraph`
so multi-edges (repeated transactions between the same pair) are correctly
reflected in counts and sums, not silently collapsed.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


GRAPH_METRICS_COLUMNS: tuple[str, ...] = (
    "account_id",
    "in_degree",
    "out_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "unique_counterparties_in",
    "unique_counterparties_out",
    "unique_counterparties_total",
    "inflow_outflow_ratio",
)


def compute_graph_metrics(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Compute one row of graph metrics per node (account) in `graph`.

    Nodes with zero edges (isolated accounts explicitly added to the graph)
    get all-zero metrics rather than being dropped, so an isolated account is
    still visible to downstream consumers.
    """
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=GRAPH_METRICS_COLUMNS)

    records = []
    for account in graph.nodes:
        in_edges = graph.in_edges(account, data=True)
        out_edges = graph.out_edges(account, data=True)

        in_degree = graph.in_degree(account)
        out_degree = graph.out_degree(account)

        weighted_in = sum(data["amount_inr"] for _, _, data in in_edges)
        weighted_out = sum(data["amount_inr"] for _, _, data in out_edges)

        unique_in = len({u for u, _, _ in graph.in_edges(account, data=True)})
        unique_out = len({v for _, v, _ in graph.out_edges(account, data=True)})
        unique_total = len(
            {u for u, _, _ in graph.in_edges(account, data=True)}
            | {v for _, v, _ in graph.out_edges(account, data=True)}
        )

        ratio = _inflow_outflow_ratio(weighted_in, weighted_out)

        records.append(
            {
                "account_id": account,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "weighted_in_degree": weighted_in,
                "weighted_out_degree": weighted_out,
                "unique_counterparties_in": unique_in,
                "unique_counterparties_out": unique_out,
                "unique_counterparties_total": unique_total,
                "inflow_outflow_ratio": ratio,
            }
        )

    df = pd.DataFrame.from_records(records, columns=list(GRAPH_METRICS_COLUMNS))
    return df.sort_values("account_id").reset_index(drop=True)


def _inflow_outflow_ratio(weighted_in: float, weighted_out: float) -> float:
    """Same convention as `core.features.transaction_features._inflow_outflow_ratio`:
    inflow / outflow, with `inf` representing a pure-sink account (receives
    but never sends) and `0.0` representing a node with no inflow at all
    (including isolated nodes)."""
    if weighted_out == 0:
        return float("inf") if weighted_in > 0 else 0.0
    return weighted_in / weighted_out
