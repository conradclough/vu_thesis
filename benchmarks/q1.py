"""
Q1 - grouped summary statistics, sweep parameter is the shipdate cutoff.

    SELECT l_returnflag, l_linestatus,
           SUM(l_quantity), SUM(l_extendedprice),
           SUM(l_extendedprice * (1 - l_discount))               AS sum_disc_price,
           SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
           AVG(l_quantity), AVG(l_extendedprice), AVG(l_discount), COUNT(*)
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL N DAY
    GROUP BY l_returnflag, l_linestatus

Steps:
  [S] shipdate : l_shipdate <= cutoff  (the sweep parameter: N days before 1998-12-01)
  [G] groupby  : encode (returnflag, linestatus) as a composite uint8 key
  [D] daily    : GROUP BY (group_key, shipdate) + SUM per (group, date)
  [A] agg      : range sum across dates in the window + AVG

When S is in Python, we bulk-fetch once and sweep N cutoffs cheaply.
With D also in SQL, we reduce the amount to fetch.
Python builds prefix sums at fetch time; each query is then 4 binary searches +
4 index lookups with O(log n) per query instead of O(n).

Validity:
  D in SQL requires G in SQL (D groups by group_key; G must compute it first).
  A in SQL requires S, G, D all in SQL (A collapses rows; Python cannot post-aggregate).

Opt flags (ON by default):
  --opt_presort      sort by (group_key, shipdate) once on the raw path - date filter
                     becomes searchsorted + reduceat, no per-query mask allocation
  --opt_precompute   hoist ep*(1-disc) and disc*(1+tax) out of the query loop
  --opt_encode_group encode (returnflag, linestatus) to uint8 at fetch time -
                     groupby sees 4 distinct integers not byte strings
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
from dataclasses import dataclass

import duckdb
import numpy as np

from benchmarks.benchmark_sweep import (
    QueryBenchmark,
    SweepResult,
    make_base_parser,
    setup_logging,
)
from scripts.setup_db import setup_db

logger = logging.getLogger(__name__)

# S/G/A - the three steps and all valid subsets.

ALL_STEPS = (
    "S",
    "G",
    "D",
    "A",
)  # shipdate filter, group encoding, daily pre-agg, range-agg

# One result row per (returnflag, linestatus) group.

Q1Row = tuple[str, str, float, float, float, float, float, float, float, int]
Q1Result = list[Q1Row]

# Generate N evenly-spaced cutoff dates (days before 1998-12-01).

_CUTOFF_BASE = np.datetime64("1998-12-01", "D")
Q1_SINGLE_N = 90  # TPC-H standard


def generate_day_params(n: int) -> list[int]:
    return [int(x) for x in np.linspace(60, 120, n, dtype=int)]


def single_query_params() -> list[int]:
    return [Q1_SINGLE_N]


def cutoff_date(days: int) -> np.datetime64:
    return _CUTOFF_BASE - np.timedelta64(days, "D")


def cutoff_literal(days: int) -> str:
    return f"DATE '{str(cutoff_date(days))}'"


def cutoff_int64(days: int) -> int:
    return int(cutoff_date(days).astype("datetime64[D]").view(np.int64))


# Encode (returnflag, linestatus) pairs to uint8 so the groupby sees 4 values not strings.

_RETURNFLAG_ORD = {ord("A"): 0, ord("N"): 1, ord("R"): 2}
_LINESTATUS_ORD = {ord("F"): 0, ord("O"): 1}
_ORD_TO_RF = {0: "A", 1: "N", 2: "R"}
_ORD_TO_LS = {0: "F", 1: "O"}


def encode_groups_numpy(rf_bytes: np.ndarray, ls_bytes: np.ndarray) -> np.ndarray:
    rf_ord = np.frompyfunc(lambda s: _RETURNFLAG_ORD.get(ord(s[0]), 0), 1, 1)(
        rf_bytes
    ).astype(np.uint8)
    ls_ord = np.frompyfunc(lambda s: _LINESTATUS_ORD.get(ord(s[0]), 0), 1, 1)(
        ls_bytes
    ).astype(np.uint8)
    return (rf_ord * 3 + ls_ord).astype(np.uint8)


def decode_group_key(key: int) -> tuple[str, str]:
    return _ORD_TO_RF.get(key // 3, "?"), _ORD_TO_LS.get(key % 3, "?")


@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_precompute: bool = True
    opt_encode_group: bool = True
    opt_presort: bool = True

    @property
    def key(self) -> str:
        return "".join(s for s in ALL_STEPS if s in self.sql_steps) or "(none)"

    @property
    def python_steps(self) -> frozenset:
        return frozenset(ALL_STEPS) - self.sql_steps

    def sql_handles(self, step: str) -> bool:
        return step in self.sql_steps

    def python_handles(self, step: str) -> bool:
        return step in self.python_steps


# Validity constraints on step assignments.
# D groups by group_key, so G must be in SQL whenever D is.
# A collapses all rows to per-group aggregates, so S, G, D must all be in SQL too.


def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("D") and cfg.python_handles("G"):
        return False, (
            "D in SQL requires G in SQL: daily pre-aggregation groups by group_key, "
            "which must be computed by G."
        )
    if cfg.sql_handles("A") and cfg.python_steps - {"A"}:
        missing = sorted(cfg.python_steps - {"A"})
        return False, (
            f"A in SQL requires S, G, D also in SQL (missing: {missing}). "
            "SQL returns grouped aggregates; Python cannot post-filter or re-aggregate."
        )
    return True, ""


# Build the SELECT/WHERE dynamically - what SQL sees changes per combo.


def _shipdate_where(days: int) -> str:
    return f"l_shipdate <= {cutoff_literal(days)}"


def build_fetch_sql(cfg: BenchConfig) -> str:
    """Bulk fetch SQL (S in Python - no shipdate filter embedded)."""
    assert cfg.python_handles("S")

    cols = ["l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_shipdate"]

    if cfg.sql_handles("G"):
        cols.append(
            "CASE l_returnflag "
            "WHEN 'A' THEN 0 WHEN 'N' THEN 3 WHEN 'R' THEN 6 ELSE 9 END "
            "+ CASE l_linestatus WHEN 'F' THEN 0 WHEN 'O' THEN 1 ELSE 2 END "
            "AS group_key"
        )
    else:
        cols += ["l_returnflag", "l_linestatus"]

    return f"SELECT {', '.join(cols)}\nFROM lineitem"


def build_fetch_sql_preagg(cfg: BenchConfig) -> str:
    """
    G+D in SQL, S+A in Python: pre-aggregate the full lineitem by (group_key, shipdate).

    Returns ~10K rows (4 groups * ~2500 unique ship dates at SF=1) instead of ~6M.
    Ordered by (group_key, shipdate) so Python can build prefix sums in one pass
    and answer each date-cutoff query with 4 binary searches + 4 index lookups.
    """
    assert cfg.sql_handles("G") and cfg.sql_handles("D") and cfg.python_handles("S")
    return (
        "SELECT\n"
        "  CASE l_returnflag WHEN 'A' THEN 0 WHEN 'N' THEN 3 WHEN 'R' THEN 6 ELSE 9 END\n"
        "  + CASE l_linestatus WHEN 'F' THEN 0 WHEN 'O' THEN 1 ELSE 2 END AS group_key,\n"
        "  l_shipdate,\n"
        "  SUM(l_quantity)                                        AS sum_qty,\n"
        "  SUM(l_extendedprice)                                   AS sum_ep,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount))             AS sum_dp,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_ch,\n"
        "  SUM(l_discount)                                        AS sum_disc,\n"
        "  COUNT(*)                                               AS cnt\n"
        "FROM lineitem\n"
        "GROUP BY group_key, l_shipdate\n"
        "ORDER BY group_key, l_shipdate"
    )


def build_per_query_sql_sgd(days: int) -> str:
    """S+G+D in SQL, A in Python: GROUP BY group (returns 4 rows), Python computes AVGs."""
    return (
        "SELECT\n"
        "  l_returnflag, l_linestatus,\n"
        "  SUM(l_quantity)                                        AS sum_qty,\n"
        "  SUM(l_extendedprice)                                   AS sum_base_price,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount))             AS sum_disc_price,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_charge,\n"
        "  SUM(l_discount)                                        AS sum_disc,\n"
        "  COUNT(*)                                               AS cnt\n"
        "FROM lineitem\n"
        f"WHERE {_shipdate_where(days)}\n"
        "GROUP BY l_returnflag, l_linestatus\n"
        "ORDER BY l_returnflag, l_linestatus"
    )


def build_per_query_sql_sg_raw(days: int) -> str:
    """S+G in SQL, D+A in Python: filter + encode groups, raw rows for Python to aggregate."""
    return (
        "SELECT\n"
        "  CASE l_returnflag WHEN 'A' THEN 0 WHEN 'N' THEN 3 WHEN 'R' THEN 6 ELSE 9 END\n"
        "  + CASE l_linestatus WHEN 'F' THEN 0 WHEN 'O' THEN 1 ELSE 2 END AS group_key,\n"
        "  l_quantity, l_extendedprice, l_discount, l_tax\n"
        "FROM lineitem\n"
        f"WHERE {_shipdate_where(days)}"
    )


def build_per_query_sql_s_raw(days: int) -> str:
    """S in SQL only, G+D+A in Python: filter only, Python encodes groups and aggregates."""
    return (
        "SELECT l_quantity, l_extendedprice, l_discount, l_tax,\n"
        "       l_returnflag, l_linestatus\n"
        "FROM lineitem\n"
        f"WHERE {_shipdate_where(days)}"
    )


def build_pure_sql(days: int) -> str:
    return (
        "SELECT\n"
        "  l_returnflag, l_linestatus,\n"
        "  SUM(l_quantity)                                        AS sum_qty,\n"
        "  SUM(l_extendedprice)                                   AS sum_base_price,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount))             AS sum_disc_price,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_charge,\n"
        "  AVG(l_quantity)                                        AS avg_qty,\n"
        "  AVG(l_extendedprice)                                   AS avg_price,\n"
        "  AVG(l_discount)                                        AS avg_disc,\n"
        "  COUNT(*)                                               AS count_order\n"
        "FROM lineitem\n"
        f"WHERE {_shipdate_where(days)}\n"
        "GROUP BY l_returnflag, l_linestatus\n"
        "ORDER BY l_returnflag, l_linestatus"
    )


@dataclass
class PreAggData:
    """
    Per-group daily aggregates with prefix sums; built once in fetch_and_prepare.

    Flat arrays sorted by (group_key ASC, shipdate ASC).  For each group g,
    the sum of any aggregate over dates in (−inf, cutoff] is:
        prefix_x[group_starts[g] + k − 1] − offset_x[g]
    where k = searchsorted(dates[group_starts[g]:group_ends[g]], cutoff, 'right').
    """

    dates: np.ndarray  # int64, flat across all groups
    unique_keys: np.ndarray  # uint8, at most 4 values
    group_starts: np.ndarray  # int64 index where each group begins
    group_ends: np.ndarray  # int64 index where each group ends (exclusive)
    prefix_qty: np.ndarray
    prefix_ep: np.ndarray
    prefix_dp: np.ndarray
    prefix_ch: np.ndarray
    prefix_disc: np.ndarray
    prefix_cnt: np.ndarray
    offset_qty: np.ndarray  # prefix value just before each group starts
    offset_ep: np.ndarray
    offset_dp: np.ndarray
    offset_ch: np.ndarray
    offset_disc: np.ndarray
    offset_cnt: np.ndarray


@dataclass
class FetchedArrays:
    quantity: np.ndarray
    ep: np.ndarray
    discount: np.ndarray
    tax: np.ndarray
    shipdate: np.ndarray  # int64 days-since-epoch
    group_key: np.ndarray | None  # uint8 composite key, or tuple of raw arrays
    disc_price: np.ndarray | None  # ep * (1 - disc)
    charge: np.ndarray | None  # ep * (1 - disc) * (1 + tax)
    presorted: bool = False
    preagg_data: PreAggData | None = None  # set when G+D are in SQL


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    if cfg.sql_handles("S"):
        return None, 0.0

    # SQL={G,D}: pre-aggregate to (group_key, shipdate) level, build prefix sums.
    # ~10K rows instead of ~6M; each per-query sweep is then O(log n).
    if cfg.sql_handles("G") and cfg.sql_handles("D"):
        t0 = time.perf_counter()
        raw = con.execute(build_fetch_sql_preagg(cfg)).fetchnumpy()
        fetch_time = time.perf_counter() - t0

        gk = raw["group_key"].astype(np.uint8)
        sd_raw = raw["l_shipdate"]
        dates = (
            sd_raw.astype("datetime64[D]").view(np.int64)
            if sd_raw.dtype.kind == "M"
            else sd_raw.astype(np.int64)
        )
        sum_qty = raw["sum_qty"].astype(np.float64)
        sum_ep = raw["sum_ep"].astype(np.float64)
        sum_dp = raw["sum_dp"].astype(np.float64)
        sum_ch = raw["sum_ch"].astype(np.float64)
        sum_disc = raw["sum_disc"].astype(np.float64)
        cnt = raw["cnt"].astype(np.int64)

        # SQL ORDER BY group_key, shipdate guarantees sort; verify group boundaries.
        unique_keys, group_starts, group_counts = np.unique(
            gk, return_index=True, return_counts=True
        )
        group_ends = group_starts + group_counts

        # Global cumulative sums (across all groups in sorted order).
        prefix_qty = np.cumsum(sum_qty)
        prefix_ep = np.cumsum(sum_ep)
        prefix_dp = np.cumsum(sum_dp)
        prefix_ch = np.cumsum(sum_ch)
        prefix_disc = np.cumsum(sum_disc)
        prefix_cnt = np.cumsum(cnt)

        # Per-group offsets: the prefix value just before this group's first row.
        # Range sum for group g, k rows: prefix_x[start+k-1] - offset_x[g].
        n_groups = len(unique_keys)
        offset_qty = np.zeros(n_groups, dtype=np.float64)
        offset_ep = np.zeros(n_groups, dtype=np.float64)
        offset_dp = np.zeros(n_groups, dtype=np.float64)
        offset_ch = np.zeros(n_groups, dtype=np.float64)
        offset_disc = np.zeros(n_groups, dtype=np.float64)
        offset_cnt = np.zeros(n_groups, dtype=np.int64)
        for i, start in enumerate(group_starts):
            if start > 0:
                offset_qty[i] = prefix_qty[start - 1]
                offset_ep[i] = prefix_ep[start - 1]
                offset_dp[i] = prefix_dp[start - 1]
                offset_ch[i] = prefix_ch[start - 1]
                offset_disc[i] = prefix_disc[start - 1]
                offset_cnt[i] = prefix_cnt[start - 1]

        preagg = PreAggData(
            dates=dates,
            unique_keys=unique_keys,
            group_starts=group_starts,
            group_ends=group_ends,
            prefix_qty=prefix_qty,
            prefix_ep=prefix_ep,
            prefix_dp=prefix_dp,
            prefix_ch=prefix_ch,
            prefix_disc=prefix_disc,
            prefix_cnt=prefix_cnt,
            offset_qty=offset_qty,
            offset_ep=offset_ep,
            offset_dp=offset_dp,
            offset_ch=offset_ch,
            offset_disc=offset_disc,
            offset_cnt=offset_cnt,
        )
        return FetchedArrays(
            quantity=np.empty(0),
            ep=np.empty(0),
            discount=np.empty(0),
            tax=np.empty(0),
            shipdate=np.empty(0, dtype=np.int64),
            group_key=None,
            disc_price=None,
            charge=None,
            presorted=False,
            preagg_data=preagg,
        ), fetch_time

    sql = build_fetch_sql(cfg)
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    fetch_time = time.perf_counter() - t0

    quantity = raw["l_quantity"].astype(np.float64)
    ep = raw["l_extendedprice"].astype(np.float64)
    discount = raw["l_discount"].astype(np.float64)
    tax = raw["l_tax"].astype(np.float64)

    sd_raw = raw["l_shipdate"]
    shipdate = (
        sd_raw.astype("datetime64[D]").view(np.int64)
        if sd_raw.dtype.kind == "M"
        else sd_raw.astype(np.int64)
    )

    # Encode (returnflag, linestatus) to a uint8 composite - 4 distinct values
    # instead of per-row string comparisons in the groupby.
    group_key: np.ndarray | None = None
    if cfg.sql_handles("G"):
        group_key = raw["group_key"].astype(np.uint8)
    elif cfg.opt_encode_group:
        group_key = encode_groups_numpy(raw["l_returnflag"], raw["l_linestatus"])
    else:
        group_key = (raw["l_returnflag"], raw["l_linestatus"])

    # Hoist ep*(1-disc) and disc_price*(1+tax) out of the per-query loop.
    # DuckDB recomputes these expressions on every query execution.
    disc_price: np.ndarray | None = None
    charge: np.ndarray | None = None
    if cfg.opt_precompute:
        disc_price = ep * (1.0 - discount)
        charge = disc_price * (1.0 + tax)

    # Sort by (group, shipdate) once so each query is a searchsorted slice per group
    # + a single reduceat pass - no per-query boolean mask or gather operation.
    if cfg.opt_presort and isinstance(group_key, np.ndarray):
        sort_idx = np.lexsort((shipdate, group_key))
        quantity = quantity[sort_idx]
        ep = ep[sort_idx]
        discount = discount[sort_idx]
        tax = tax[sort_idx]
        shipdate = shipdate[sort_idx]
        group_key = group_key[sort_idx]
        if disc_price is not None:
            disc_price = disc_price[sort_idx]
        if charge is not None:
            charge = charge[sort_idx]

    return FetchedArrays(
        quantity=quantity,
        ep=ep,
        discount=discount,
        tax=tax,
        shipdate=shipdate,
        group_key=group_key,
        disc_price=disc_price,
        charge=charge,
        presorted=cfg.opt_presort and isinstance(group_key, np.ndarray),
    ), fetch_time


# Grouped aggregation in numpy - presorted path uses reduceat+searchsorted, unsorted uses mask+sort.
def _numpy_groupby_agg(
    group_key: np.ndarray,
    shipdate: np.ndarray,
    quantity: np.ndarray,
    ep: np.ndarray,
    discount: np.ndarray,
    disc_price: np.ndarray | None,
    charge: np.ndarray | None,
    cutoff_i64: int,
    presorted: bool,
    tax: np.ndarray | None = None,
) -> Q1Result:
    """
    Core grouped aggregation.

    PRESORTED PATH: arrays sorted by (group_key ASC, shipdate ASC).
      1. reduceat over full arrays -> full-group sums (one sequential pass).
      2. Per group (4 total), searchsorted -> prefix length k.
      3. Subtract suffix [k:end] (contiguous slice) from full-group sums.
      No index gather - maximally cache-friendly.

    UNSORTED PATH: boolean mask -> sort by group key -> unique + reduceat.
    """
    if presorted:
        unique_keys, group_starts, group_counts = np.unique(
            group_key, return_index=True, return_counts=True
        )

        dp = disc_price if disc_price is not None else ep * (1.0 - discount)
        ch = (
            charge
            if charge is not None
            else dp * (1.0 + (tax if tax is not None else 0.0))
        )

        full_sum_qty = np.add.reduceat(quantity, group_starts)
        full_sum_base = np.add.reduceat(ep, group_starts)
        full_sum_dp = np.add.reduceat(dp, group_starts)
        full_sum_ch = np.add.reduceat(ch, group_starts)
        full_sum_disc = np.add.reduceat(discount, group_starts)

        sum_qty = full_sum_qty.copy().astype(np.float64)
        sum_base = full_sum_base.copy().astype(np.float64)
        sum_disc_price = full_sum_dp.copy().astype(np.float64)
        sum_charge_arr = full_sum_ch.copy().astype(np.float64)
        sum_disc = full_sum_disc.copy().astype(np.float64)
        counts = group_counts.copy().astype(np.int64)

        for i in range(len(unique_keys)):
            start = int(group_starts[i])
            count = int(group_counts[i])
            k = int(
                np.searchsorted(
                    shipdate[start : start + count], cutoff_i64, side="right"
                )
            )
            if k == 0:
                sum_qty[i] = sum_base[i] = sum_disc_price[i] = 0.0
                sum_charge_arr[i] = sum_disc[i] = 0.0
                counts[i] = 0
            elif k < count:
                s = slice(start + k, start + count)
                sum_qty[i] -= quantity[s].sum()
                sum_base[i] -= ep[s].sum()
                sum_disc_price[i] -= dp[s].sum()
                sum_charge_arr[i] -= ch[s].sum()
                sum_disc[i] -= discount[s].sum()
                counts[i] = k

        active = counts > 0
        active_keys = unique_keys[active]
        sum_qty = sum_qty[active]
        sum_base = sum_base[active]
        sum_disc_price = sum_disc_price[active]
        sum_charge_arr = sum_charge_arr[active]
        sum_disc = sum_disc[active]
        counts = counts[active]

        if len(active_keys) == 0:
            return []

    else:
        mask = shipdate <= cutoff_i64
        if not mask.any():
            return []

        dp = (
            disc_price[mask]
            if disc_price is not None
            else ep[mask] * (1.0 - discount[mask])
        )
        ch = (
            charge[mask]
            if charge is not None
            else dp * (1.0 + (tax[mask] if tax is not None else 0.0))
        )

        gk_m = group_key[mask]
        qty_m = quantity[mask]
        ep_m = ep[mask]
        disc_m = discount[mask]

        sort_idx = np.argsort(gk_m, kind="stable")
        gk_s = gk_m[sort_idx]
        qty_s = qty_m[sort_idx]
        ep_s = ep_m[sort_idx]
        disc_s = disc_m[sort_idx]
        dp_s = dp[sort_idx]
        ch_s = ch[sort_idx]

        active_keys, reduceat_starts, active_counts = np.unique(
            gk_s, return_index=True, return_counts=True
        )
        sum_qty = np.add.reduceat(qty_s, reduceat_starts)
        sum_base = np.add.reduceat(ep_s, reduceat_starts)
        sum_disc_price = np.add.reduceat(dp_s, reduceat_starts)
        sum_charge_arr = np.add.reduceat(ch_s, reduceat_starts)
        sum_disc = np.add.reduceat(disc_s, reduceat_starts)
        counts = active_counts

    result: Q1Result = []
    for i, key in enumerate(active_keys):
        rf, ls = decode_group_key(int(key))
        cnt = int(counts[i])
        result.append(
            (
                rf,
                ls,
                round(float(sum_qty[i]), 4),
                round(float(sum_base[i]), 4),
                round(float(sum_disc_price[i]), 4),
                round(float(sum_charge_arr[i]), 4),
                round(float(sum_qty[i]) / cnt, 6),
                round(float(sum_base[i]) / cnt, 6),
                round(float(sum_disc[i]) / cnt, 6),
                cnt,
            )
        )
    return sorted(result, key=lambda r: (r[0], r[1]))


# Convert raw numpy arrays or SQL group rows to the Q1Result format.
def _rows_to_q1result(raw: dict) -> Q1Result:
    qty = raw["l_quantity"].astype(np.float64)
    ep = raw["l_extendedprice"].astype(np.float64)
    disc = raw["l_discount"].astype(np.float64)
    tax = raw["l_tax"].astype(np.float64)
    dp = ep * (1.0 - disc)
    ch = dp * (1.0 + tax)
    gk = encode_groups_numpy(raw["l_returnflag"], raw["l_linestatus"])

    sort_idx = np.argsort(gk, kind="stable")
    gk = gk[sort_idx]
    qty = qty[sort_idx]
    ep = ep[sort_idx]
    disc = disc[sort_idx]
    dp = dp[sort_idx]
    ch = ch[sort_idx]

    unique_keys, starts, counts = np.unique(gk, return_index=True, return_counts=True)
    sum_qty = np.add.reduceat(qty, starts)
    sum_base = np.add.reduceat(ep, starts)
    sum_dp = np.add.reduceat(dp, starts)
    sum_ch = np.add.reduceat(ch, starts)
    sum_disc = np.add.reduceat(disc, starts)

    result: Q1Result = []
    for i, key in enumerate(unique_keys):
        rf, ls = decode_group_key(int(key))
        cnt = int(counts[i])
        result.append(
            (
                rf,
                ls,
                round(float(sum_qty[i]), 4),
                round(float(sum_base[i]), 4),
                round(float(sum_dp[i]), 4),
                round(float(sum_ch[i]), 4),
                round(float(sum_qty[i]) / cnt, 6),
                round(float(sum_base[i]) / cnt, 6),
                round(float(sum_disc[i]) / cnt, 6),
                cnt,
            )
        )
    return sorted(result, key=lambda r: (r[0], r[1]))


def _sql_groups_to_q1result(rows) -> Q1Result:
    result: Q1Result = []
    for row in rows:
        rf, ls = str(row[0]), str(row[1])
        sq, sb, sdp, sch, sdisc, cnt = (
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            int(row[7]),
        )
        result.append(
            (
                rf,
                ls,
                round(sq, 4),
                round(sb, 4),
                round(sdp, 4),
                round(sch, 4),
                round(sq / cnt, 6),
                round(sb / cnt, 6),
                round(sdisc / cnt, 6),
                cnt,
            )
        )
    return sorted(result, key=lambda r: (r[0], r[1]))


# O(log n) per-query sweep on the pre-aggregated prefix sums.
def _query_from_preagg(pa: PreAggData, cutoff_i64: int) -> Q1Result:
    """
    Answer one date-cutoff query using the pre-built prefix sums.

    For each of the (at most 4) groups: binary search for the cutoff in that
    group's sorted date array, then read the prefix sum at that position.
    Total work: 4 binary searches + 4 subtractions + 4 divisions.
    """
    result: Q1Result = []
    for i, key in enumerate(pa.unique_keys):
        start = int(pa.group_starts[i])
        end = int(pa.group_ends[i])
        k = int(np.searchsorted(pa.dates[start:end], cutoff_i64, side="right"))
        if k == 0:
            continue
        idx = start + k - 1
        sum_qty = float(pa.prefix_qty[idx]) - float(pa.offset_qty[i])
        sum_ep = float(pa.prefix_ep[idx]) - float(pa.offset_ep[i])
        sum_dp = float(pa.prefix_dp[idx]) - float(pa.offset_dp[i])
        sum_ch = float(pa.prefix_ch[idx]) - float(pa.offset_ch[i])
        sum_disc = float(pa.prefix_disc[idx]) - float(pa.offset_disc[i])
        cnt = int(pa.prefix_cnt[idx]) - int(pa.offset_cnt[i])
        if cnt == 0:
            continue
        rf, ls = decode_group_key(int(key))
        result.append(
            (
                rf,
                ls,
                round(sum_qty, 4),
                round(sum_ep, 4),
                round(sum_dp, 4),
                round(sum_ch, 4),
                round(sum_qty / cnt, 6),
                round(sum_ep / cnt, 6),
                round(sum_disc / cnt, 6),
                cnt,
            )
        )
    return sorted(result, key=lambda r: (r[0], r[1]))


def _encoded_rows_to_q1result(raw: dict) -> Q1Result:
    """Aggregate raw rows where group_key is already encoded as uint8."""
    gk = raw["group_key"].astype(np.uint8)
    qty = raw["l_quantity"].astype(np.float64)
    ep = raw["l_extendedprice"].astype(np.float64)
    disc = raw["l_discount"].astype(np.float64)
    tax = raw["l_tax"].astype(np.float64)
    dp = ep * (1.0 - disc)
    ch = dp * (1.0 + tax)

    sort_idx = np.argsort(gk, kind="stable")
    gk = gk[sort_idx]
    qty = qty[sort_idx]
    ep = ep[sort_idx]
    disc = disc[sort_idx]
    dp = dp[sort_idx]
    ch = ch[sort_idx]

    unique_keys, starts, counts = np.unique(gk, return_index=True, return_counts=True)
    sum_qty = np.add.reduceat(qty, starts)
    sum_base = np.add.reduceat(ep, starts)
    sum_dp = np.add.reduceat(dp, starts)
    sum_ch = np.add.reduceat(ch, starts)
    sum_disc = np.add.reduceat(disc, starts)

    result: Q1Result = []
    for i, key in enumerate(unique_keys):
        rf, ls = decode_group_key(int(key))
        cnt = int(counts[i])
        result.append(
            (
                rf,
                ls,
                round(float(sum_qty[i]), 4),
                round(float(sum_base[i]), 4),
                round(float(sum_dp[i]), 4),
                round(float(sum_ch[i]), 4),
                round(float(sum_qty[i]) / cnt, 6),
                round(float(sum_base[i]) / cnt, 6),
                round(float(sum_disc[i]) / cnt, 6),
                cnt,
            )
        )
    return sorted(result, key=lambda r: (r[0], r[1]))


# Route each query to the right path depending on which steps SQL owns.
def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    days: int,
) -> Q1Result:

    if cfg.sql_handles("S"):
        # Pure SQL: S+G+D+A all in SQL
        if all(cfg.sql_handles(s) for s in ("G", "D", "A")):
            rows = con.execute(build_pure_sql(days)).fetchall()
            return [
                (
                    str(r[0]),
                    str(r[1]),
                    round(float(r[2]), 4),
                    round(float(r[3]), 4),
                    round(float(r[4]), 4),
                    round(float(r[5]), 4),
                    round(float(r[6]), 6),
                    round(float(r[7]), 6),
                    round(float(r[8]), 6),
                    int(r[9]),
                )
                for r in rows
            ]
        # S+G+D in SQL, A in Python: SQL returns one row per group, Python computes AVGs
        if cfg.sql_handles("G") and cfg.sql_handles("D") and cfg.python_handles("A"):
            rows = con.execute(build_per_query_sql_sgd(days)).fetchall()
            return _sql_groups_to_q1result(rows)
        # S+G in SQL, D+A in Python: raw date-filtered rows with group encoding
        if cfg.sql_handles("G") and cfg.python_handles("D"):
            raw = con.execute(build_per_query_sql_sg_raw(days)).fetchnumpy()
            return _encoded_rows_to_q1result(raw)
        # S in SQL only, G+D+A in Python: raw date-filtered rows, Python encodes + aggregates
        raw = con.execute(build_per_query_sql_s_raw(days)).fetchnumpy()
        return _rows_to_q1result(raw)

    # S in Python: use pre-fetched arrays
    assert arrays is not None

    # SQL={G,D}: prefix-sum sweep — O(log n) per query
    if arrays.preagg_data is not None:
        return _query_from_preagg(arrays.preagg_data, cutoff_int64(days))

    # Raw path (SQL={G} or SQL={}): numpy grouped aggregation
    cutoff_i64 = cutoff_int64(days)
    disc_price = arrays.disc_price
    charge = arrays.charge
    if disc_price is None:
        disc_price = arrays.ep * (1.0 - arrays.discount)
    if charge is None:
        charge = disc_price * (1.0 + arrays.tax)

    if isinstance(arrays.group_key, np.ndarray):
        gk = arrays.group_key
    else:
        rf_arr, ls_arr = arrays.group_key
        gk = encode_groups_numpy(rf_arr, ls_arr)

    return _numpy_groupby_agg(
        group_key=gk,
        shipdate=arrays.shipdate,
        quantity=arrays.quantity,
        ep=arrays.ep,
        discount=arrays.discount,
        disc_price=disc_price,
        charge=charge,
        cutoff_i64=cutoff_i64,
        presorted=arrays.presorted,
        tax=arrays.tax,
    )


# Fetch + N-query loop for one combo; return a timed SweepResult.
def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[int],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key or '(none)'}: {reason}")

    arrays, fetch_time = fetch_and_prepare(con, cfg)

    results = []
    t1 = time.perf_counter()
    for days in params:
        results.append(run_one_query(con, cfg, arrays, days))
    logic_time = time.perf_counter() - t1

    return SweepResult(
        key=cfg.key,
        sql_steps_str=cfg.key if cfg.sql_steps else "(none)",
        python_steps_str="".join(s for s in ALL_STEPS if cfg.python_handles(s))
        or "(none)",
        values=results,
        fetch_time=fetch_time,
        logic_time=logic_time,
        total_time=fetch_time + logic_time,
    )


# Check Python results against the SQL reference
def _result_close(a: Q1Result, b: Q1Result, tol: float = 1.0) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra[0] != rb[0] or ra[1] != rb[1]:
            return False
        if any(abs(fa - fb) > tol for fa, fb in zip(ra[2:9], rb[2:9])):
            return False
        if ra[9] != rb[9]:
            return False
    return True


def validate(
    reference: list[Q1Result],
    candidate: list[Q1Result],
    params: list[int],
    label: str,
    tol: float = 1.0,
) -> bool:
    mismatches = [
        (params[i], reference[i], candidate[i])
        for i in range(len(params))
        if not _result_close(reference[i], candidate[i], tol)
    ]
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for days, ref, cand in mismatches[:3]:
            logger.warning(
                f"    N={days} days  ref_groups={len(ref)}  got_groups={len(cand)}"
            )
            for rr, rc in zip(ref[:2], cand[:2]):
                logger.warning(f"      ref: {rr}")
                logger.warning(f"      got: {rc}")
        return False
    logger.info(f"  [{label}] all results match reference")
    return True


# Plug Q1 into the shared benchmark harness.
class Q1Benchmark(QueryBenchmark):
    NAME = "Q1"
    ALL_STEPS = ("S", "G", "D", "A")
    N_APPLICABLE = True
    N_HELP = "Number of day-offset sweep values in [60, 120] (default: 60)"
    N_DEFAULT = 60

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_precompute=args.opt_precompute,
            opt_encode_group=args.opt_encode_group,
            opt_presort=args.opt_presort,
        )

    def generate_params(self, n: int) -> list[int]:
        return generate_day_params(n)

    def single_params(self) -> list[int]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query (N={Q1_SINGLE_N} days, N=1)"

    def is_valid_combo(self, cfg: BenchConfig) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg: BenchConfig, params: list[int]) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"precompute={'ON' if args.opt_precompute else 'OFF'}  "
            f"encode_group={'ON' if args.opt_encode_group else 'OFF'}  "
            f"presort={'ON' if args.opt_presort else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_precompute",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Precompute disc_price and charge at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_group",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Encode (returnflag, linestatus) as uint8 at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort arrays by (group_key ASC, shipdate ASC) at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        return f"sweep  N={len(params)} day-offset values in [60, 120]"


# Run directly: python3 -m benchmarks.q1
def main() -> None:
    bench = Q1Benchmark()
    parser = make_base_parser(
        "Q1 predicate mix-and-match: benchmark every SQL/Python split"
    )
    bench.add_query_args(parser)
    args = parser.parse_args()

    if args.n is None:
        args.n = bench.N_DEFAULT

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "single" if args.single else f"n{args.n}"
    log_filename = os.path.join(
        log_dir, f"q1_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
