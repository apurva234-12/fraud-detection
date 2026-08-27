# Fraudulent Transaction Detection — Graph-Based Anomaly Detection & AML Risk Scoring

B.Tech CS minor project. INR-denominated digital transaction data is modelled as a graph
of accounts and transaction flows, then analysed for **anomalous / suspicious / high-risk**
activity for human investigation. The system never claims to conclusively prove fraud.

## Status: Phase 0/1 complete

This phase establishes the foundation only:

- Canonical transaction schema + strict validation (`core/data`)
- Deterministic, scenario-labelled synthetic transaction generator (`core/data/synthetic.py`)
- CSV loader built on the same validation path (`core/data/loader.py`)
- Baseline per-account behavioural features (`core/features`)
- Transaction graph construction on `networkx.MultiDiGraph` (`core/graph/builder.py`)
- Initial graph metrics: degree, weighted degree, unique counterparties, inflow/outflow ratio
  (`core/graph/metrics.py`)
- Unit tests for every module above, including edge cases
- An end-to-end demonstration script (`scripts/run_pipeline.py`)

**Not implemented yet** (by design — later phases): FastAPI, PostgreSQL, React,
authentication, anomaly detection, AML rules, risk scoring. See `docs/design_decisions.md`
for the full architecture and phase plan.

## Project layout

```
core/
  data/       canonical schema, validation, synthetic generator, CSV loader
  features/   per-account behavioural feature engineering
  graph/      MultiDiGraph construction + graph metrics
config/       central, non-hardcoded configuration (timezone, thresholds, allowed types)
tests/unit/   pytest suite, mirrors core/ structure
scripts/      run_pipeline.py — synthetic data -> validation -> features -> graph -> metrics
docs/         design decisions and rationale
```

The `core` package has **no dependency on any web framework or database** — it is a plain
Python library, importable and testable on its own. This is intentional: FastAPI/PostgreSQL
(later phases) will be thin adapters around it, not the other way round.

## Setup

```bash
cd fraud-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the demonstration pipeline

```bash
python scripts/run_pipeline.py
```

This generates a reproducible synthetic dataset, validates it, computes account features,
builds the transaction graph, computes graph metrics, and prints a summary report.

## Running tests

```bash
pytest -v
```

## Design conventions (see `docs/design_decisions.md` for full rationale)

- All timestamps are interpreted and stored in **IST (Asia/Kolkata)**.
- Currency is **INR only**; no multi-currency/crypto/card-instrument handling in this phase.
- Invalid records are never silently dropped — they are collected into a **rejects report**
  with a machine-readable reason code, so nothing disappears unexplained.
- Synthetic scenario labels (e.g. `HIGH_FANOUT`, `CIRCULAR_FLOW`) are **not** part of the
  transaction schema and are **not** fraud labels — they are kept in a separate table for
  later evaluation of the detection pipeline, so they can never leak into features or models.
