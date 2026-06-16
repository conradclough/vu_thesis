"""
Q4 - order counts by priority, sweep over (start date, window width).

    SELECT o_orderpriority, COUNT(*) AS order_count
    FROM orders
    WHERE o_orderdate >= DATE '[DATE]'
      AND o_orderdate < DATE '[DATE]' + INTERVAL '3' MONTH
      AND EXISTS (
          SELECT 1 FROM lineitem
          WHERE l_orderkey = o_orderkey
            AND l_commitdate < l_receiptdate
      )
    GROUP BY o_orderpriority
    ORDER BY o_orderpriority

The interesting predicate is the EXISTS correlated subquery: for each order,
SQL checks whether any lineitem was received late (l_commitdate < l_receiptdate).
That's a per-order correlated scan - DuckDB re-runs it for every (date, width) pair.
Python precomputes the set of late orderkeys once from lineitem (sorted int64 array),
then sweeps (date, width) pairs as a date-window slice + np.isin membership test.

Steps (each independently assigned to SQL or Python):
  [E] exists : precompute the set of late orderkeys (l_commitdate < l_receiptdate)
  [J] join   : semi-join orders against the late-key set means late orders only
  [D] date   : o_orderdate in [start_date, start_date + months)
  [G] group  : GROUP BY o_orderpriority + COUNT(*)

Validity:
  J in SQL requires E in SQL (SQL needs the late-key set to do the EXISTS semi-join).
  G in SQL requires E, J, D also in SQL (GROUP BY collapses rows; Python can't post-count).

The J step is the key new addition. When D is in Python (bulk-fetch path), J is applied
ONCE at fetch time rather than once per query:
  sql=EJ: SQL runs EXISTS once → late orders array. Python sweeps dates (searchsorted)
    and counts. No per-query isin at all.
  sql=E, Python J: SQL returns late-key set + raw orders; Python applies isin once to
    build the late-orders array. Per query: searchsorted + count only.

Opt flags (all ON by default):
  --opt_presort_orders   sort orders by orderdate - date window becomes searchsorted slice
  --opt_encode_priority  encode o_orderpriority as uint8 at fetch time

Usage:
"""

from __future__ import annotations

import argparse
import datetime
import itertools
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

# E/D/G - the three steps and all valid subsets.

ALL_STEPS = (
    "E",
    "J",
    "D",
    "G",
)  # exists/late-key set, join/semi-join, date window, group+count

# Priority encoding and date range constants from the TPC-H Q4 spec.

_SWEEP_DATE_LO = np.datetime64("1993-01-01", "D")
_SWEEP_DATE_HI = np.datetime64("1997-10-01", "D")  # leave room for 4-month windows
ALL_WIDTHS = [1, 2, 3, 4]  # window widths in months
SINGLE_DATE = np.datetime64("1993-07-01", "D")
SINGLE_WIDTH = 3  # Q4 spec uses 3 months

# Priority encoding: TPC-H has exactly 5 priorities in alpha order
ALL_PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
_PRI_TO_U8: dict[str, int] = {p: i for i, p in enumerate(ALL_PRIORITIES)}
_U8_TO_PRI: dict[int, str] = {i: p for p, i in _PRI_TO_U8.items()}

# One result row per order priority.

Q4Row = tuple[str, int]  # (o_orderpriority, order_count)
Q4Result = list[Q4Row]
SweepParam = tuple[np.datetime64, int]  # (start_date, width_months)


# Generate (start_date, window_months) parameter grid.
def _month_add(d: np.datetime64, months: int) -> np.datetime64:
    """Add *months* calendar months to a datetime64[D] date."""
    dt = d.astype("datetime64[D]").astype(object)  # -> datetime.date
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    import calendar

    day = min(dt.day, calendar.monthrange(y, m)[1])
    return np.datetime64(datetime.date(y, m, day), "D")


def generate_sweep_params(n_dates: int) -> list[SweepParam]:
    """n_dates evenly-spaced start dates * 4 window widths."""
    span = int((_SWEEP_DATE_HI - _SWEEP_DATE_LO) / np.timedelta64(1, "D"))
    offsets = np.linspace(0, span, n_dates, dtype=int)
    dates = [
        (_SWEEP_DATE_LO + np.timedelta64(int(o), "D"))
        .astype("datetime64[M]")
        .astype("datetime64[D]")  # snap to month start
        for o in offsets
    ]
    # Deduplicate dates while preserving order
    seen = set()
    unique_dates = []
    for d in dates:
        k = str(d)
        if k not in seen:
            seen.add(k)
            unique_dates.append(d)
    return [(d, w) for d in unique_dates for w in ALL_WIDTHS]


def single_query_params() -> list[SweepParam]:
    return [(SINGLE_DATE, SINGLE_WIDTH)]


def _date_i64(d: np.datetime64) -> int:
    return int(d.astype("datetime64[D]").view(np.int64))


def _date_str(d: np.datetime64) -> str:
    return str(d.astype("datetime64[D]"))


# Which steps go to SQL vs Python, and which numpy opts are active.
@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_presort_orders: bool = True  # sort orders by orderdate for searchsorted
    opt_encode_priority: bool = True  # encode priority string -> uint8

    @property
    def key(self) -> str:
        return "".join(s for s in ALL_STEPS if s in self.sql_steps) or "(none)"

    @property
    def python_steps(self) -> frozenset:
        return frozenset(ALL_STEPS) - self.sql_steps

    def sql_handles(self, step: str) -> bool:
        return step in self.sql_steps

    def python_handles(self, step: str) -> bool:
        return step not in self.sql_steps


# Validity constraints on step assignments.
# J (semi-join) requires E (late-key set) to already be in SQL.
# G collapses rows - Python can't post-count, so G in SQL forces E, J, D into SQL too.
def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("J") and cfg.python_handles("E"):
        return False, (
            "J in SQL requires E in SQL: the EXISTS semi-join needs the late-key "
            "set, which must be computed by E first."
        )
    if cfg.sql_handles("G") and cfg.python_steps - {"G"}:
        missing = sorted(cfg.python_steps - {"G"})
        return False, (
            f"G in SQL requires E, J, D also in SQL (missing: {missing}). "
            "SQL GROUP BY collapses rows; Python cannot post-filter a count."
        )
    return True, ""


# Build the SELECT/WHERE/EXISTS dynamically - what SQL sees changes per combo.
_LATE_SUBQUERY = (
    "SELECT DISTINCT l_orderkey\nFROM lineitem\nWHERE l_commitdate < l_receiptdate"
)


def _date_where(d: np.datetime64, months: int) -> str:
    lo = _date_str(d)
    hi = _date_str(_month_add(d, months))
    return f"o_orderdate >= DATE '{lo}' AND o_orderdate < DATE '{hi}'"


def _priority_encode_sql() -> str:
    """SQL CASE expression encoding o_orderpriority -> uint8."""
    cases = " ".join(f"WHEN '{p}' THEN {i}" for p, i in _PRI_TO_U8.items())
    return f"CASE o_orderpriority {cases} ELSE 255 END AS pri_key"


def build_fetch_sql_orders(cfg: BenchConfig) -> str:
    """
    Bulk fetch of orders (D in Python - no date filter embedded).
    Always fetches o_orderkey, o_orderdate, and priority.
    """
    assert cfg.python_handles("D")
    pri_col = _priority_encode_sql() if cfg.opt_encode_priority else "o_orderpriority"
    return f"SELECT o_orderkey, o_orderdate, {pri_col}\nFROM orders"


def build_fetch_sql_late_keys() -> str:
    """Fetch the set of late orderkeys from lineitem (E in Python)."""
    return _LATE_SUBQUERY


def build_per_query_sql_full(d: np.datetime64, months: int) -> str:
    """Pure SQL: E+D+G all in SQL."""
    lo = _date_str(d)
    hi = _date_str(_month_add(d, months))
    return (
        "SELECT o_orderpriority, COUNT(*) AS order_count\n"
        "FROM orders\n"
        f"WHERE o_orderdate >= DATE '{lo}'\n"
        f"  AND o_orderdate <  DATE '{hi}'\n"
        "  AND EXISTS (\n"
        "      SELECT 1 FROM lineitem\n"
        "      WHERE l_orderkey = o_orderkey\n"
        "        AND l_commitdate < l_receiptdate\n"
        "  )\n"
        "GROUP BY o_orderpriority\n"
        "ORDER BY o_orderpriority"
    )


def build_fetch_sql_late_orders(cfg: BenchConfig) -> str:
    """
    E+J in SQL, D+G in Python: fetch all late orders via EXISTS, no date filter.

    Returns ~1.1M rows at SF=1 (orders with at least one late lineitem) instead of
    running the correlated EXISTS subquery for every (date, width) pair.  Ordered by
    o_orderdate so Python can use searchsorted for the date window without re-sorting.
    """
    assert cfg.sql_handles("E") and cfg.sql_handles("J") and cfg.python_handles("D")
    pri_col = _priority_encode_sql() if cfg.opt_encode_priority else "o_orderpriority"
    return (
        f"SELECT o_orderkey, o_orderdate, {pri_col}\n"
        "FROM orders\n"
        "WHERE EXISTS (\n"
        "    SELECT 1 FROM lineitem\n"
        "    WHERE l_orderkey = o_orderkey\n"
        "      AND l_commitdate < l_receiptdate\n"
        ")\n"
        "ORDER BY o_orderdate"
    )


def build_per_query_sql_d_only(cfg: BenchConfig, d: np.datetime64, months: int) -> str:
    """D in SQL, E+J+G in Python: return date-filtered orders; Python applies late-key test."""
    assert cfg.sql_handles("D") and cfg.python_handles("J")
    pri_col = _priority_encode_sql() if cfg.opt_encode_priority else "o_orderpriority"
    return f"SELECT o_orderkey, {pri_col}\nFROM orders\nWHERE {_date_where(d, months)}"


def build_per_query_sql_ed_raw(cfg: BenchConfig, d: np.datetime64, months: int) -> str:
    """E+D in SQL, J+G in Python: date-filtered orders without EXISTS; Python applies isin."""
    assert cfg.sql_handles("E") and cfg.sql_handles("D") and cfg.python_handles("J")
    pri_col = _priority_encode_sql() if cfg.opt_encode_priority else "o_orderpriority"
    return f"SELECT o_orderkey, {pri_col}\nFROM orders\nWHERE {_date_where(d, months)}"


def build_per_query_sql_ejd(cfg: BenchConfig, d: np.datetime64, months: int) -> str:
    """E+J+D in SQL, G in Python: date-filtered late orders via EXISTS; Python counts."""
    assert (
        cfg.sql_handles("E")
        and cfg.sql_handles("J")
        and cfg.sql_handles("D")
        and cfg.python_handles("G")
    )
    pri_col = _priority_encode_sql() if cfg.opt_encode_priority else "o_orderpriority"
    return (
        f"SELECT {pri_col}\n"
        "FROM orders\n"
        f"WHERE {_date_where(d, months)}\n"
        "  AND EXISTS (\n"
        "      SELECT 1 FROM lineitem\n"
        "      WHERE l_orderkey = o_orderkey\n"
        "        AND l_commitdate < l_receiptdate\n"
        "  )"
    )


# Arrays holding bulk-fetched orders and late-orderkey sets.
@dataclass
class LateKeys:
    """Sorted int64 array of orderkeys where at least one lineitem was late."""

    orderkeys: np.ndarray  # int64, sorted ascending


@dataclass
class OrderArrays:
    """Bulk-fetched orders table (no date filter)."""

    orderkey: np.ndarray  # int64
    orderdate: np.ndarray  # int64 days-since-epoch
    pri_key: np.ndarray | None  # uint8 (opt_encode_priority)
    pri_raw: np.ndarray | None  # object strings
    presorted: bool = False  # sorted by orderdate ASC
    late_filtered: bool = (
        False  # True when J already applied (only late orders present)
    )


def _to_i64(arr: np.ndarray) -> np.ndarray:
    return (
        arr.astype("datetime64[D]").view(np.int64)
        if arr.dtype.kind == "M"
        else arr.astype(np.int64)
    )


def fetch_late_orders_sql(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[OrderArrays, float]:
    """
    sql={E,J}: one SQL fetch returns all late orders via EXISTS, sorted by orderdate.

    Eliminates the per-query isin: the resulting OrderArrays has late_filtered=True,
    so _numpy_query skips the membership test entirely and just slices by date.
    """
    sql = build_fetch_sql_late_orders(cfg)
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    ft = time.perf_counter() - t0

    orderkey = raw["o_orderkey"].astype(np.int64)
    orderdate = _to_i64(raw["o_orderdate"])
    pri_key = raw["pri_key"].astype(np.uint8) if cfg.opt_encode_priority else None
    pri_raw = None if cfg.opt_encode_priority else raw["o_orderpriority"]

    # SQL ORDER BY o_orderdate already sorted; skip re-sort unless opt is off.
    presorted = True  # guaranteed by ORDER BY in the SQL
    return OrderArrays(
        orderkey=orderkey,
        orderdate=orderdate,
        pri_key=pri_key,
        pri_raw=pri_raw,
        presorted=presorted,
        late_filtered=True,
    ), ft


def _apply_late_filter(orders: OrderArrays, late_keys: LateKeys) -> OrderArrays:
    """
    J step in Python: apply isin ONCE to keep only late orders.

    Boolean masking preserves array order, so if orders was presorted by date the
    result is also presorted - no re-sort needed.
    """
    late_mask = np.isin(orders.orderkey, late_keys.orderkeys, assume_unique=False)
    ok = orders.orderkey[late_mask]
    od = orders.orderdate[late_mask]
    pk = orders.pri_key[late_mask] if orders.pri_key is not None else None
    pr = orders.pri_raw[late_mask] if orders.pri_raw is not None else None
    return OrderArrays(
        orderkey=ok,
        orderdate=od,
        pri_key=pk,
        pri_raw=pr,
        presorted=orders.presorted,  # masking preserves sort order
        late_filtered=True,
    )


def fetch_late_keys(con: duckdb.DuckDBPyConnection) -> tuple[LateKeys, float]:
    # One SQL scan over lineitem to get the distinct orderkeys where commitdate < receiptdate.
    # DuckDB re-runs this correlated scan per (date, width) pair; we pay it once.
    t0 = time.perf_counter()
    raw = con.execute(build_fetch_sql_late_keys()).fetchnumpy()
    ft = time.perf_counter() - t0
    # Sort ascending so np.isin (which uses searchsorted internally) is fast.
    keys = np.sort(raw["l_orderkey"].astype(np.int64))
    return LateKeys(orderkeys=keys), ft


def fetch_orders(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[OrderArrays, float]:
    # Bulk fetch all orders - no date filter embedded.
    # DuckDB pushes the date window into every query; we fetch once and sweep.
    sql = build_fetch_sql_orders(cfg)
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    ft = time.perf_counter() - t0

    orderkey = raw["o_orderkey"].astype(np.int64)
    orderdate = _to_i64(raw["o_orderdate"])

    # SQL already encoded priority as uint8 - per-query groupby is np.unique on integers.
    if cfg.opt_encode_priority and "pri_key" in raw:
        pri_key = raw["pri_key"].astype(np.uint8)
        pri_raw = None
    else:
        pri_key = None
        pri_raw = raw["o_orderpriority"]

    presorted = False
    if cfg.opt_presort_orders:
        # Sort by orderdate once - per-query date window becomes two searchsorted calls.
        # DuckDB applies the date filter on every (date, width) pair.
        sort_idx = np.argsort(orderdate, kind="stable")
        orderkey = orderkey[sort_idx]
        orderdate = orderdate[sort_idx]
        if pri_key is not None:
            pri_key = pri_key[sort_idx]
        if pri_raw is not None:
            pri_raw = pri_raw[sort_idx]
        presorted = True

    return OrderArrays(
        orderkey=orderkey,
        orderdate=orderdate,
        pri_key=pri_key,
        pri_raw=pri_raw,
        presorted=presorted,
    ), ft


# N tight loops - no SQL calls from here. The bulk fetch bought us that.
def _count_by_priority(
    pri: np.ndarray,  # uint8 or object, already filtered to qualifying rows
    encoded: bool,
) -> Q4Result:
    """Group by priority and count, return sorted by priority name."""
    if encoded:
        unique_keys, counts = np.unique(pri, return_counts=True)
        result = [
            (_U8_TO_PRI.get(int(k), "?"), int(c)) for k, c in zip(unique_keys, counts)
        ]
    else:
        unique_vals, counts = np.unique(pri, return_counts=True)
        result = [(str(v), int(c)) for v, c in zip(unique_vals, counts)]
    return sorted(result, key=lambda r: r[0])


def _numpy_query(
    late_keys: LateKeys,
    orders: OrderArrays,
    d: np.datetime64,
    months: int,
) -> Q4Result:
    """
    Apply date window + late-key membership in numpy, count by priority.

    Date window:
      - presorted: searchsorted on orderdate -> contiguous slice [lo_idx, hi_idx)
      - unsorted:  boolean mask

    Late-key membership:
      - np.searchsorted on sorted late_keys.orderkeys -> O(log n) per key
        implemented as np.isin with assume_unique hints for batch efficiency
    """
    lo_i64 = _date_i64(d)
    hi_i64 = _date_i64(_month_add(d, months))

    if orders.presorted:
        lo_idx = int(np.searchsorted(orders.orderdate, lo_i64, side="left"))
        hi_idx = int(np.searchsorted(orders.orderdate, hi_i64, side="left"))
        if lo_idx >= hi_idx:
            return []
        ok_slice = orders.orderkey[lo_idx:hi_idx]
        pri_slice = (
            orders.pri_key[lo_idx:hi_idx]
            if orders.pri_key is not None
            else orders.pri_raw[lo_idx:hi_idx]
        )
    else:
        date_mask = (orders.orderdate >= lo_i64) & (orders.orderdate < hi_i64)
        if not date_mask.any():
            return []
        ok_slice = orders.orderkey[date_mask]
        pri_slice = (
            orders.pri_key[date_mask]
            if orders.pri_key is not None
            else orders.pri_raw[date_mask]
        )

    if orders.late_filtered:
        # J already applied at fetch time (sql=EJ or Python one-time isin): skip membership test.
        if len(ok_slice) == 0:
            return []
        return _count_by_priority(pri_slice, orders.pri_key is not None)

    # J not pre-applied: late-key membership test per query (O(n log m))
    assert late_keys is not None
    late_mask = np.isin(ok_slice, late_keys.orderkeys, assume_unique=False)
    if not late_mask.any():
        return []
    return _count_by_priority(pri_slice[late_mask], orders.pri_key is not None)


# Route each query to the right path depending on which steps SQL owns.
def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    late_keys: LateKeys | None,
    orders: OrderArrays | None,
    d: np.datetime64,
    months: int,
) -> Q4Result:

    # Pure SQL: E+J+D+G all in SQL
    if all(cfg.sql_handles(s) for s in ("E", "J", "D", "G")):
        rows = con.execute(build_per_query_sql_full(d, months)).fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    # E+J+D in SQL, G in Python: SQL does EXISTS + date filter; Python counts
    if cfg.sql_handles("E") and cfg.sql_handles("J") and cfg.sql_handles("D"):
        raw = con.execute(build_per_query_sql_ejd(cfg, d, months)).fetchnumpy()
        col = "pri_key" if cfg.opt_encode_priority else "o_orderpriority"
        if len(raw.get(col, [])) == 0:
            return []
        return _count_by_priority(
            raw["pri_key"].astype(np.uint8)
            if cfg.opt_encode_priority
            else raw["o_orderpriority"],
            encoded=cfg.opt_encode_priority,
        )

    # E+D in SQL, J+G in Python: SQL date-filters (no EXISTS); Python applies isin + counts
    if cfg.sql_handles("E") and cfg.sql_handles("D") and cfg.python_handles("J"):
        assert late_keys is not None
        raw = con.execute(build_per_query_sql_ed_raw(cfg, d, months)).fetchnumpy()
        ok = raw["o_orderkey"].astype(np.int64)
        pri = (
            raw["pri_key"].astype(np.uint8)
            if "pri_key" in raw
            else raw["o_orderpriority"]
        )
        late_mask = np.isin(ok, late_keys.orderkeys, assume_unique=False)
        if not late_mask.any():
            return []
        return _count_by_priority(pri[late_mask], "pri_key" in raw)

    # D in SQL, J in Python: SQL date-filters; Python applies isin + counts
    if cfg.sql_handles("D") and cfg.python_handles("J"):
        assert late_keys is not None
        raw = con.execute(build_per_query_sql_d_only(cfg, d, months)).fetchnumpy()
        ok = raw["o_orderkey"].astype(np.int64)
        pri = (
            raw["pri_key"].astype(np.uint8)
            if "pri_key" in raw
            else raw["o_orderpriority"]
        )
        late_mask = np.isin(ok, late_keys.orderkeys, assume_unique=False)
        if not late_mask.any():
            return []
        return _count_by_priority(pri[late_mask], "pri_key" in raw)

    # D in Python: use pre-fetched orders (late_filtered=True skips isin inside _numpy_query)
    assert orders is not None
    return _numpy_query(late_keys, orders, d, months)


# Orchestrate fetches + N-query loop for one combo; return a timed SweepResult.
def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[SweepParam],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key}: {reason}")

    fetch_time = 0.0
    late_keys: LateKeys | None = None
    orders: OrderArrays | None = None

    if cfg.python_handles("D"):
        if cfg.sql_handles("J"):
            # sql=EJ: one SQL fetch returns all late orders via EXISTS (no date filter).
            # No late_keys needed - J is fully handled by SQL.
            orders, ft = fetch_late_orders_sql(con, cfg)
            fetch_time += ft
        else:
            # J in Python (one-time): need late_keys + raw orders, then apply isin once.
            late_keys, ft = fetch_late_keys(con)
            fetch_time += ft
            orders_raw, ft2 = fetch_orders(con, cfg)
            fetch_time += ft2
            orders = _apply_late_filter(orders_raw, late_keys)
    elif cfg.python_handles("J"):
        # D in SQL, J in Python: need late_keys for per-query isin.
        late_keys, ft = fetch_late_keys(con)
        fetch_time += ft
    # else: D+J both in SQL (per-query EXISTS path) - no bulk fetch needed.

    results = []
    t1 = time.perf_counter()
    for d, months in params:
        results.append(run_one_query(con, cfg, late_keys, orders, d, months))
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
def validate(
    reference: list[Q4Result],
    candidate: list[Q4Result],
    params: list[SweepParam],
    label: str,
) -> bool:
    mismatches = []
    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        if len(ref) != len(cand):
            mismatches.append((params[i], ref, cand, "length"))
            continue
        for rr, rc in zip(ref, cand):
            if rr[0] != rc[0] or rr[1] != rc[1]:
                mismatches.append((params[i], ref, cand, "value"))
                break
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, ref, cand, kind in mismatches[:3]:
            logger.warning(f"    date={_date_str(p[0])} months={p[1]} ({kind})")
            logger.warning(f"      ref: {ref}")
            logger.warning(f"      got: {cand}")
        return False
    logger.info(f"  [{label}] all results match reference")
    return True


# Plug Q4 into the shared benchmark harness.
class Q4Benchmark(QueryBenchmark):
    NAME = "Q4"
    ALL_STEPS = ("E", "J", "D", "G")
    N_APPLICABLE = True
    N_HELP = "Number of start dates to sweep (default: 10); total params = n * 4 window widths"
    N_DEFAULT = 10

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_presort_orders=args.opt_presort_orders,
            opt_encode_priority=args.opt_encode_priority,
        )

    def generate_params(self, n: int) -> list[SweepParam]:
        return generate_sweep_params(n)

    def single_params(self) -> list[SweepParam]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query ({_date_str(SINGLE_DATE)}, {SINGLE_WIDTH} months)"

    def is_valid_combo(self, cfg: BenchConfig) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg: BenchConfig, params) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"presort_orders={'ON' if args.opt_presort_orders else 'OFF'}  "
            f"encode_priority={'ON' if args.opt_encode_priority else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_presort_orders",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort orders by orderdate for searchsorted window (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_priority",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Encode o_orderpriority -> uint8 at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        n = args.n if args.n is not None else self.N_DEFAULT
        return (
            f"sweep  N={len(params)} params "
            f"({min(n, len(params) // len(ALL_WIDTHS))} dates * {len(ALL_WIDTHS)} widths)"
        )


# Run directly: python3 -m benchmarks.q4


def main() -> None:
    bench = Q4Benchmark()
    parser = make_base_parser(
        "Q4 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q4_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
