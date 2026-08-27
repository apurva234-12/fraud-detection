"""
Central configuration for the fraud-detection core library.

Design decision: configuration is a plain, typed, immutable dataclass with sane
defaults, optionally overridden by a YAML file. Nothing downstream (validators,
generator, features, graph) should hardcode a threshold, allowed value, or
timezone string directly -- it should read it from a `Config` instance.

This keeps every "magic number" (velocity window, allowed transaction types,
timezone, quarantine behaviour) in exactly one place, which is what the project
brief explicitly asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass(frozen=True)
class Config:
    # --- Timezone convention -------------------------------------------------
    # All timestamps are interpreted and stored in this timezone. This is a
    # project-level decision (locked in for Phase 1): IST, since the dataset
    # is INR / India-context transaction data.
    timezone: str = "Asia/Kolkata"

    # --- Domain / schema constraints -----------------------------------------
    # Closed set of allowed transaction types. Deliberately conservative and
    # specific to Indian digital/electronic banking rails -- no cards, no
    # crypto, no stock/investment instruments in this phase.
    allowed_transaction_types: tuple[str, ...] = (
        "UPI",
        "NEFT",
        "IMPS",
        "RTGS",
        "CASH_DEPOSIT",
        "CASH_WITHDRAWAL",
    )

    # Whether a transaction where sender_account_id == receiver_account_id is
    # accepted into the "clean" dataset. Default: False. Self-transactions are
    # not silently dropped either way -- when disallowed, they are routed to
    # the rejects report with reason SELF_TRANSACTION so they remain visible
    # and auditable rather than disappearing.
    allow_self_transactions: bool = False

    # Minimum transaction amount in INR (exclusive). Zero and negative amounts
    # are always invalid; this additionally lets us reject sub-paisa noise if
    # a real dataset ever has it. Kept at the smallest valid currency unit by
    # default so it does not silently reject legitimate small transactions.
    min_amount_inr: float = 0.01

    # --- Behavioural feature parameters --------------------------------------
    # Rolling window (in whole days) used to compute a simple transaction
    # velocity proxy (max transactions by an account in any single day within
    # the observed period). This is intentionally simple in Phase 1 -- a
    # richer, configurable multi-window velocity feature belongs to the AML
    # rule engine (later phase), not here.
    velocity_window_days: int = 1

    # --- Synthetic data generator ---------------------------------------------
    random_seed: int = 42
    synthetic_num_accounts: int = 200
    synthetic_num_normal_transactions: int = 3000

    # --- Validation / ingestion behaviour -------------------------------------
    # Invalid records are always quarantined (never silently dropped); this
    # flag only controls whether the loader raises if *any* record is invalid,
    # useful for strict test fixtures vs. exploratory runs.
    raise_on_any_invalid: bool = False


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration, optionally overridden by a YAML file.

    If `path` is None or does not exist, the built-in defaults are returned.
    Only keys present in the YAML file override the corresponding defaults;
    unknown keys in the YAML raise an error immediately (fail loud on typos
    in config rather than silently ignoring a misspelled key).
    """
    defaults = Config()
    if path is None:
        return defaults

    path = Path(path)
    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    valid_keys = set(defaults.__dataclass_fields__.keys())
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown configuration key(s) in {path}: {sorted(unknown)}")

    # Tuples need explicit conversion since YAML gives lists.
    if "allowed_transaction_types" in raw:
        raw["allowed_transaction_types"] = tuple(raw["allowed_transaction_types"])

    merged = {**defaults.__dict__, **raw}
    return Config(**merged)
