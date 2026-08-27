"""
CSV ingestion for transaction data.

Deliberately thin: this module's only job is to turn a CSV file into an
iterable of raw row dicts and hand them to `core.data.validators.validate_batch`.
All validation logic lives in one place (`validators.py`) so the synthetic
generator's output and real CSV uploads go through exactly the same checks --
there is no separate "trusted" ingestion path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import Config
from core.data.schema import REQUIRED_COLUMNS, transactions_to_dataframe
from core.data.validators import ValidationOutcome, validate_batch


class SchemaError(ValueError):
    """Raised when a CSV is missing required columns entirely -- this is a
    structural problem with the file, not a per-row data-quality issue, so it
    is raised immediately rather than quarantining every row."""


def load_transactions_csv(path: str | Path, config: Config) -> ValidationOutcome:
    """Load a CSV file of raw transactions and validate every row.

    Returns a `ValidationOutcome` with the valid `Transaction` objects and a
    rejects DataFrame explaining every row that failed validation (including
    duplicate transaction_id detection across the whole file). Nothing is
    silently dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=True)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise SchemaError(
            f"CSV {path} is missing required column(s): {sorted(missing_columns)}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    rows = df.to_dict(orient="records")
    return validate_batch(rows, config)


def valid_transactions_dataframe(outcome: ValidationOutcome) -> pd.DataFrame:
    """Convenience helper: materialise the valid transactions from a
    `ValidationOutcome` as a canonical DataFrame."""
    return transactions_to_dataframe(outcome.valid_transactions)
