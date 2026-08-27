#!/usr/bin/env python3
"""
End-to-end Phase 0/1 demonstration:

    synthetic data -> validation -> feature generation -> graph construction
        -> graph metrics -> summary report

This script proves the analytical core works end-to-end without any web
framework, database, or frontend involved. Run from the project root:

    python scripts/run_pipeline.py
"""

from __future__ import annotations

from core.config import Config
from core.data.synthetic import generate_synthetic_dataset
from core.data.validators import validate_batch
from core.data.schema import transactions_to_dataframe
from core.features.transaction_features import compute_account_features
from core.graph.builder import build_transaction_graph
from core.graph.metrics import compute_graph_metrics


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    config = Config()

    # ------------------------------------------------------------------
    # 1. Synthetic data generation
    # ------------------------------------------------------------------
    _section("1. SYNTHETIC DATA GENERATION")
    dataset = generate_synthetic_dataset(config)
    print(f"Generated {len(dataset.transactions)} synthetic transactions "
          f"(random_seed={config.random_seed}, reproducible).")
    print("\nScenario counts (NOT fraud labels -- controlled academic test patterns):")
    print(dataset.scenario_counts().to_string())

    # ------------------------------------------------------------------
    # 2. Validation (same path a real CSV upload would take)
    # ------------------------------------------------------------------
    _section("2. VALIDATION")
    raw_rows = [
        {
            "transaction_id": t.transaction_id,
            "sender_account_id": t.sender_account_id,
            "receiver_account_id": t.receiver_account_id,
            "amount_inr": t.amount_inr,
            "timestamp": t.timestamp,
            "transaction_type": t.transaction_type.value,
        }
        for t in dataset.transactions
    ]
    outcome = validate_batch(raw_rows, config)
    print(f"Valid transactions:   {len(outcome.valid_transactions)}")
    print(f"Rejected rows:        {len(outcome.rejects)}")
    if not outcome.rejects.empty:
        print(outcome.rejects.head())
    transactions_df = transactions_to_dataframe(outcome.valid_transactions)

    # ------------------------------------------------------------------
    # 3. Feature generation
    # ------------------------------------------------------------------
    _section("3. ACCOUNT FEATURE GENERATION")
    features_df = compute_account_features(transactions_df, config)
    num_accounts = features_df["account_id"].nunique()
    print(f"Accounts with features: {num_accounts}")
    print("\nSample features (5 accounts with the most sent transactions):")
    print(
        features_df.sort_values("txn_count_sent", ascending=False)
        .head(5)[
            [
                "account_id",
                "txn_count_sent",
                "txn_count_received",
                "unique_counterparties_sent",
                "inflow_outflow_ratio",
                "max_daily_txn_count",
            ]
        ]
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 4. Graph construction
    # ------------------------------------------------------------------
    _section("4. TRANSACTION GRAPH CONSTRUCTION")
    graph = build_transaction_graph(transactions_df)
    print(f"Graph nodes (accounts): {graph.number_of_nodes()}")
    print(f"Graph edges (transactions): {graph.number_of_edges()}")

    # ------------------------------------------------------------------
    # 5. Graph metrics
    # ------------------------------------------------------------------
    _section("5. GRAPH METRICS")
    metrics_df = compute_graph_metrics(graph)
    print("\nSample graph metrics (5 accounts with the highest out-degree):")
    print(
        metrics_df.sort_values("out_degree", ascending=False)
        .head(5)[
            [
                "account_id",
                "in_degree",
                "out_degree",
                "weighted_in_degree",
                "weighted_out_degree",
                "unique_counterparties_total",
            ]
        ]
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    _section("6. SUMMARY")
    print(f"Transactions (raw synthetic):     {len(dataset.transactions)}")
    print(f"Transactions (valid, post-check): {len(outcome.valid_transactions)}")
    print(f"Distinct accounts:                {num_accounts}")
    print(f"Graph nodes:                      {graph.number_of_nodes()}")
    print(f"Graph edges:                      {graph.number_of_edges()}")
    print(f"Scenario types injected:          {dataset.transaction_scenarios['scenario'].nunique()}")
    print()
    print("NOTE: scenario labels above are controlled synthetic test patterns for")
    print("later evaluating the detection pipeline (Phases 5-7). They are NOT real")
    print("fraud labels and no anomaly/AML/risk scoring has been performed yet --")
    print("that is out of scope for Phase 0/1.")


if __name__ == "__main__":
    main()
