"""
Deterministic synthetic INR transaction generator.

IMPORTANT (per project scope): the scenarios produced here are controlled,
hand-designed academic test patterns used to exercise and later evaluate the
detection pipeline (Phases 5-7). They are NOT a claim about what real-world
fraud looks like, and they are NOT fraud labels. Scenario labels are kept in
a structure entirely separate from `Transaction` (see `SyntheticDataset`) so
they can never leak into feature engineering or model input.

Reproducibility: the entire generator is driven by a single
`numpy.random.Generator` seeded from `Config.random_seed`. Given the same
config, the output is byte-for-byte identical across runs.

Scenarios implemented (each returns real, schema-valid `Transaction` objects
plus labels -- injected scenarios are additional transactions layered on top
of a normal baseline, not corruptions of the schema):

- NORMAL: baseline random transaction activity between accounts.
- HIGH_VELOCITY: one account issues a burst of many transactions in a short
  time window.
- LARGE_TRANSFER: a single transaction far above the normal amount
  distribution.
- REPEATED_TRANSACTIONS: the same sender/receiver pair transacts a fixed
  amount many times (structuring-like repetition).
- HIGH_FANOUT: one source account sends to a large number of unique
  receivers.
- PASS_THROUGH: an account receives funds and forwards a near-identical
  amount onward shortly after, repeatedly (mule/intermediary behaviour).
- CIRCULAR_FLOW: funds move A -> B -> C -> A, returning to the origin.
- LAYERED_MULTI_HOP: funds move through a chain of several accounts without
  returning to the origin (layering, no closed loop).
- BEHAVIOURAL_CHANGE: an account's transaction pattern (amount and/or
  frequency) shifts sharply partway through the observed window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from core.config import Config
from core.data.schema import Transaction, TransactionType, transactions_to_dataframe


@dataclass
class SyntheticDataset:
    """Output of the synthetic generator.

    `transactions` are ordinary, schema-valid `Transaction` objects -- nothing
    downstream needs to know these are synthetic.

    `transaction_scenarios` maps transaction_id -> scenario name, for
    transactions that were deliberately injected as part of a named scenario
    (baseline/normal transactions are also labelled, as "NORMAL").

    `account_scenarios` maps account_id -> list of scenario roles that
    account plays (an account can participate in more than one scenario,
    e.g. be the source of both a HIGH_FANOUT and a HIGH_VELOCITY scenario).

    Both label structures are kept separate from `Transaction` /
    `core.data.schema` on purpose -- see module docstring.
    """

    transactions: list[Transaction]
    transaction_scenarios: pd.DataFrame  # columns: transaction_id, scenario
    account_scenarios: pd.DataFrame  # columns: account_id, scenario

    def transactions_df(self) -> pd.DataFrame:
        return transactions_to_dataframe(self.transactions)

    def scenario_counts(self) -> pd.Series:
        return self.transaction_scenarios["scenario"].value_counts()


class _TransactionFactory:
    """Small stateful helper that assigns unique, sequential transaction IDs
    and centralises random-choice helpers so scenario generators stay
    readable. Not part of the public API."""

    def __init__(self, rng: np.random.Generator, config: Config, tz: "pytz.BaseTzInfo"):
        self._rng = rng
        self._config = config
        self._tz = tz
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"TXN{self._counter:08d}"

    def random_amount(self, low: float = 100.0, high: float = 50_000.0) -> float:
        # Log-uniform-ish distribution: most transactions small, occasional
        # larger ones, without an artificial hard ceiling shape.
        return float(round(np.exp(self._rng.uniform(np.log(low), np.log(high))), 2))

    def random_type(self) -> TransactionType:
        choice = self._rng.choice(self._config.allowed_transaction_types)
        return TransactionType(choice)

    def make(
        self,
        sender: str,
        receiver: str,
        timestamp: datetime,
        amount: Optional[float] = None,
        txn_type: Optional[TransactionType] = None,
    ) -> Transaction:
        return Transaction(
            transaction_id=self.next_id(),
            sender_account_id=sender,
            receiver_account_id=receiver,
            amount_inr=amount if amount is not None else self.random_amount(),
            timestamp=timestamp,
            transaction_type=txn_type if txn_type is not None else self.random_type(),
        )


def generate_synthetic_dataset(
    config: Config,
    num_accounts: Optional[int] = None,
    num_normal_transactions: Optional[int] = None,
    window_days: int = 30,
) -> SyntheticDataset:
    """Generate a full, reproducible synthetic dataset: a normal baseline plus
    every scenario listed in the module docstring.

    `num_accounts` / `num_normal_transactions` default to the values in
    `config` (so a small fixture dataset can be produced simply by passing a
    `Config` with small values, without duplicating generator logic).
    """
    num_accounts = num_accounts if num_accounts is not None else config.synthetic_num_accounts
    num_normal_transactions = (
        num_normal_transactions if num_normal_transactions is not None else config.synthetic_num_normal_transactions
    )
    _MIN_ACCOUNTS_FOR_SCENARIOS = 20  # sum of accounts reserved by every scenario below
    if num_accounts < _MIN_ACCOUNTS_FOR_SCENARIOS:
        raise ValueError(
            f"generate_synthetic_dataset requires at least {_MIN_ACCOUNTS_FOR_SCENARIOS} "
            f"accounts to place all scenarios, got {num_accounts}"
        )

    rng = np.random.default_rng(config.random_seed)
    tz = pytz.timezone(config.timezone)
    factory = _TransactionFactory(rng, config, tz)

    window_start = tz.localize(datetime(2024, 1, 1, 9, 0, 0))
    window_end = window_start + timedelta(days=window_days)

    accounts = [f"ACC{i:05d}" for i in range(num_accounts)]

    transactions: list[Transaction] = []
    txn_scenario_records: list[dict] = []
    account_scenario_records: list[dict] = []

    def tag_txn(txn: Transaction, scenario: str) -> None:
        txn_scenario_records.append({"transaction_id": txn.transaction_id, "scenario": scenario})

    def tag_account(account_id: str, scenario: str) -> None:
        account_scenario_records.append({"account_id": account_id, "scenario": scenario})

    def random_timestamp() -> datetime:
        offset_seconds = rng.uniform(0, (window_end - window_start).total_seconds())
        return window_start + timedelta(seconds=float(offset_seconds))

    # ------------------------------------------------------------------
    # 1. NORMAL baseline: random pairs, random amounts, spread across window.
    # ------------------------------------------------------------------
    for _ in range(num_normal_transactions):
        sender, receiver = rng.choice(accounts, size=2, replace=False)
        txn = factory.make(sender, receiver, random_timestamp())
        transactions.append(txn)
        tag_txn(txn, "NORMAL")

    # Reserve a handful of accounts per scenario so scenarios don't overlap
    # unpredictably and remain easy to inspect/debug.
    reserved = accounts[: min(40, num_accounts)]
    it = iter(reserved)

    def take(n: int) -> list[str]:
        return [next(it) for _ in range(n)]

    # ------------------------------------------------------------------
    # 2. HIGH_VELOCITY: one account fires a burst of transactions in <1 hour.
    # ------------------------------------------------------------------
    (velocity_account,) = take(1)
    burst_start = random_timestamp()
    receivers = rng.choice([a for a in accounts if a != velocity_account], size=15, replace=False)
    for i, receiver in enumerate(receivers):
        ts = burst_start + timedelta(minutes=float(rng.uniform(0, 45)))
        txn = factory.make(velocity_account, receiver, ts, amount=factory.random_amount(500, 5000))
        transactions.append(txn)
        tag_txn(txn, "HIGH_VELOCITY")
    tag_account(velocity_account, "HIGH_VELOCITY_SOURCE")

    # ------------------------------------------------------------------
    # 3. LARGE_TRANSFER: one transaction far above the normal distribution.
    # ------------------------------------------------------------------
    large_sender, large_receiver = take(2)
    txn = factory.make(large_sender, large_receiver, random_timestamp(), amount=2_500_000.0)
    transactions.append(txn)
    tag_txn(txn, "LARGE_TRANSFER")
    tag_account(large_sender, "LARGE_TRANSFER_SENDER")
    tag_account(large_receiver, "LARGE_TRANSFER_RECEIVER")

    # ------------------------------------------------------------------
    # 4. REPEATED_TRANSACTIONS: same pair, same amount, many times
    #    (structuring-like repetition).
    # ------------------------------------------------------------------
    rep_sender, rep_receiver = take(2)
    repeat_amount = 9_500.0  # deliberately just under a round figure
    base_ts = random_timestamp()
    for i in range(12):
        ts = base_ts + timedelta(hours=float(i * 6))
        txn = factory.make(rep_sender, rep_receiver, ts, amount=repeat_amount)
        transactions.append(txn)
        tag_txn(txn, "REPEATED_TRANSACTIONS")
    tag_account(rep_sender, "REPEATED_TRANSACTIONS_SENDER")
    tag_account(rep_receiver, "REPEATED_TRANSACTIONS_RECEIVER")

    # ------------------------------------------------------------------
    # 5. HIGH_FANOUT: one source sends to many unique receivers (not
    #    necessarily in a short window, unlike HIGH_VELOCITY).
    # ------------------------------------------------------------------
    (fanout_account,) = take(1)
    fanout_receivers = rng.choice([a for a in accounts if a != fanout_account], size=30, replace=False)
    for receiver in fanout_receivers:
        txn = factory.make(fanout_account, receiver, random_timestamp(), amount=factory.random_amount(200, 3000))
        transactions.append(txn)
        tag_txn(txn, "HIGH_FANOUT")
    tag_account(fanout_account, "HIGH_FANOUT_SOURCE")

    # ------------------------------------------------------------------
    # 6. PASS_THROUGH: intermediary receives then forwards a near-identical
    #    amount shortly after, repeated across several inflow/outflow pairs.
    # ------------------------------------------------------------------
    origin_a, intermediary, dest_a = take(3)
    origin_b, dest_b = take(2)
    for origin, dest in [(origin_a, dest_a), (origin_b, dest_b)]:
        inflow_amount = factory.random_amount(10_000, 80_000)
        inflow_ts = random_timestamp()
        outflow_ts = inflow_ts + timedelta(minutes=float(rng.uniform(5, 60)))
        in_txn = factory.make(origin, intermediary, inflow_ts, amount=inflow_amount)
        out_txn = factory.make(intermediary, dest, outflow_ts, amount=round(inflow_amount * 0.98, 2))
        transactions.extend([in_txn, out_txn])
        tag_txn(in_txn, "PASS_THROUGH")
        tag_txn(out_txn, "PASS_THROUGH")
    tag_account(intermediary, "PASS_THROUGH_INTERMEDIARY")

    # ------------------------------------------------------------------
    # 7. CIRCULAR_FLOW: A -> B -> C -> A, funds return to origin.
    # ------------------------------------------------------------------
    circ_a, circ_b, circ_c = take(3)
    circ_amount = factory.random_amount(20_000, 60_000)
    circ_ts = random_timestamp()
    for i, (s, r) in enumerate([(circ_a, circ_b), (circ_b, circ_c), (circ_c, circ_a)]):
        ts = circ_ts + timedelta(hours=float(i * 2))
        txn = factory.make(s, r, ts, amount=round(circ_amount * (0.97**i), 2))
        transactions.append(txn)
        tag_txn(txn, "CIRCULAR_FLOW")
    for acc in (circ_a, circ_b, circ_c):
        tag_account(acc, "CIRCULAR_FLOW_PARTICIPANT")

    # ------------------------------------------------------------------
    # 8. LAYERED_MULTI_HOP: chain through several accounts, no return to
    #    origin (distinguishes layering from a closed circular loop).
    # ------------------------------------------------------------------
    chain = take(5)
    layer_amount = factory.random_amount(30_000, 100_000)
    layer_ts = random_timestamp()
    for i in range(len(chain) - 1):
        ts = layer_ts + timedelta(hours=float(i * 3))
        txn = factory.make(chain[i], chain[i + 1], ts, amount=round(layer_amount * (0.95**i), 2))
        transactions.append(txn)
        tag_txn(txn, "LAYERED_MULTI_HOP")
    for acc in chain:
        tag_account(acc, "LAYERED_MULTI_HOP_PARTICIPANT")

    # ------------------------------------------------------------------
    # 9. BEHAVIOURAL_CHANGE: quiet, small-amount activity in the first half
    #    of the window, then a sharp shift to frequent, large-amount
    #    activity in the second half.
    # ------------------------------------------------------------------
    (change_account,) = take(1)
    midpoint = window_start + (window_end - window_start) / 2
    other_accounts = [a for a in accounts if a != change_account]

    for _ in range(4):  # quiet phase
        ts = window_start + timedelta(seconds=float(rng.uniform(0, (midpoint - window_start).total_seconds())))
        receiver = rng.choice(other_accounts)
        txn = factory.make(change_account, receiver, ts, amount=factory.random_amount(100, 1000))
        transactions.append(txn)
        tag_txn(txn, "BEHAVIOURAL_CHANGE_BEFORE")

    for _ in range(25):  # shifted phase
        ts = midpoint + timedelta(seconds=float(rng.uniform(0, (window_end - midpoint).total_seconds())))
        receiver = rng.choice(other_accounts)
        txn = factory.make(change_account, receiver, ts, amount=factory.random_amount(20_000, 90_000))
        transactions.append(txn)
        tag_txn(txn, "BEHAVIOURAL_CHANGE_AFTER")
    tag_account(change_account, "BEHAVIOURAL_CHANGE")

    # ------------------------------------------------------------------
    # Sort by timestamp so the resulting dataset reads like a real feed.
    # ------------------------------------------------------------------
    transactions.sort(key=lambda t: t.timestamp)

    txn_scenarios_df = pd.DataFrame(txn_scenario_records, columns=["transaction_id", "scenario"])
    account_scenarios_df = pd.DataFrame(account_scenario_records, columns=["account_id", "scenario"]).drop_duplicates()

    return SyntheticDataset(
        transactions=transactions,
        transaction_scenarios=txn_scenarios_df,
        account_scenarios=account_scenarios_df,
    )


def generate_fixture_dataset(config: Config) -> SyntheticDataset:
    """Small, fast dataset for unit tests: same scenarios as the full
    generator, just far fewer normal-baseline transactions and accounts, so
    tests run quickly while still exercising every scenario type."""
    return generate_synthetic_dataset(
        config,
        num_accounts=40,
        num_normal_transactions=60,
        window_days=14,
    )
