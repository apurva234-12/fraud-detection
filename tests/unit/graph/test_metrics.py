from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
import pytz

from core.graph.builder import build_transaction_graph
from core.graph.metrics import GRAPH_METRICS_COLUMNS, compute_graph_metrics


IST = pytz.timezone("Asia/Kolkata")


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "transaction_id", "sender_account_id", "receiver_account_id",
        "amount_inr", "timestamp", "transaction_type",
    ])


def _ts(day: int = 1) -> pd.Timestamp:
    return pd.Timestamp(IST.localize(datetime(2024, 1, day, 10, 0, 0)))


def test_empty_graph_returns_empty_correctly_columned():
    graph = build_transaction_graph(_df([]))
    df = compute_graph_metrics(graph)
    assert df.empty
    assert list(df.columns) == list(GRAPH_METRICS_COLUMNS)


def test_isolated_account_gets_zeroed_metrics():
    graph = build_transaction_graph(_df([]), extra_isolated_accounts=["Z"])
    df = compute_graph_metrics(graph).set_index("account_id")
    assert df.loc["Z", "in_degree"] == 0
    assert df.loc["Z", "out_degree"] == 0
    assert df.loc["Z", "inflow_outflow_ratio"] == 0.0


def test_degree_and_weighted_degree():
    rows = [
        {"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 100.0, "timestamp": _ts(), "transaction_type": "UPI"},
        {"transaction_id": "T2", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 50.0, "timestamp": _ts(), "transaction_type": "UPI"},
        {"transaction_id": "T3", "sender_account_id": "C", "receiver_account_id": "A",
         "amount_inr": 30.0, "timestamp": _ts(), "transaction_type": "UPI"},
    ]
    graph = build_transaction_graph(_df(rows))
    df = compute_graph_metrics(graph).set_index("account_id")

    assert df.loc["A", "out_degree"] == 2  # 2 outgoing edges (multi-edge to B)
    assert df.loc["A", "in_degree"] == 1
    assert df.loc["A", "weighted_out_degree"] == 150.0
    assert df.loc["A", "weighted_in_degree"] == 30.0


def test_unique_counterparties_vs_degree_with_repeated_edges():
    rows = [
        {"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(), "transaction_type": "UPI"},
        {"transaction_id": "T2", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(), "transaction_type": "UPI"},
        {"transaction_id": "T3", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(), "transaction_type": "UPI"},
    ]
    graph = build_transaction_graph(_df(rows))
    df = compute_graph_metrics(graph).set_index("account_id")
    assert df.loc["A", "out_degree"] == 3          # 3 transaction edges
    assert df.loc["A", "unique_counterparties_out"] == 1  # but only 1 distinct counterparty


def test_inflow_outflow_ratio_pure_sink_and_pure_source():
    rows = [{"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
             "amount_inr": 100.0, "timestamp": _ts(), "transaction_type": "UPI"}]
    graph = build_transaction_graph(_df(rows))
    df = compute_graph_metrics(graph).set_index("account_id")
    assert df.loc["A", "inflow_outflow_ratio"] == 0.0
    assert df.loc["B", "inflow_outflow_ratio"] == float("inf")


def test_high_fanout_account_has_high_out_degree_low_reciprocity():
    rows = [
        {"transaction_id": f"T{i}", "sender_account_id": "HUB", "receiver_account_id": f"LEAF{i}",
         "amount_inr": 10.0, "timestamp": _ts(), "transaction_type": "UPI"}
        for i in range(20)
    ]
    graph = build_transaction_graph(_df(rows))
    df = compute_graph_metrics(graph).set_index("account_id")
    assert df.loc["HUB", "out_degree"] == 20
    assert df.loc["HUB", "unique_counterparties_out"] == 20
    assert df.loc["HUB", "in_degree"] == 0
