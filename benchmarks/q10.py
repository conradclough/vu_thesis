"""
Q10 - top-20 customers by returned-item revenue, sweep over (date, window width).

    SELECT c_custkey, c_name,
           SUM(l_extendedprice * (1 - l_discount)) AS revenue,
           c_acctbal, n_name, c_address, c_comment
    FROM customer, orders, lineitem, nation
    WHERE c_custkey = o_custkey
      AND l_orderkey = o_orderkey
      AND o_orderdate >= DATE '[DATE]'
      AND o_orderdate < DATE '[DATE]' + INTERVAL '3' MONTH
      AND l_returnflag = 'R'
      AND c_nationkey = n_nationkey
    GROUP BY c_custkey, c_name, c_acctbal, c_phone,
             n_name, c_address, c_comment
    ORDER BY revenue DESC
    LIMIT 20

The 4-table join is always in SQL. DuckDB re-runs the full join and aggregation
for every (date, width) pair. Python bulk-fetches the join once and sweeps pairs
as cheap array slices - R-filtered rows are a contiguous prefix after presort,
so the date window is a searchsorted slice within that prefix.

Steps (each independently assigned to SQL or Python):
  [R] returnflag : l_returnflag = 'R' - filter to returned lineitems only
  [D] date       : o_orderdate in [date, date + months)
  [A] pre-agg    : GROUP BY (customer, orderdate) + SUM(ep*(1-disc)) per (customer, date)
  [G] group+top20: GROUP BY customer + SUM(rev_day) + TOP-20 by np.argpartition

Validity:
  A in SQL requires R in SQL: pre-aggregation must follow the returnflag filter.
  G in SQL requires R, D, A in SQL: GROUP BY collapses rows; Python cannot post-filter
  or re-aggregate.

Opt flags (all ON by default):
  --opt_precompute    compute ep*(1-disc) at fetch time - DuckDB recomputes it per query
  --opt_presort       sort by (returnflag_bool, orderdate, custkey) at fetch time
  --opt_encode_return encode l_returnflag as bool at fetch time (R=True)
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

# R/D/G - the three steps and all valid subsets.

ALL_STEPS = ("R", "D", "A", "G")  # returnflag, date window, pre-agg by day, group+top20

# Date range and window widths for the sweep.

_SWEEP_DATE_LO = np.datetime64("1993-10-01", "D")
_SWEEP_DATE_HI = np.datetime64("1995-01-01", "D")
ALL_WIDTHS = [2, 3, 4]
SINGLE_DATE = np.datetime64("1993-10-01", "D")
SINGLE_WIDTH = 3
LIMIT = 20

# Top-20 customers by lost revenue from returned items.

# (c_custkey, c_name, revenue, c_acctbal, n_name, c_address, c_comment)
Q10Row = tuple[int, str, float, float, str, str, str]
Q10Result = list[Q10Row]
SweepParam = tuple[np.datetime64, int]  # (start_date, width_months)

# Generate (start_date, window_months) parameter grid.


def _month_add(d: np.datetime64, months: int) -> np.datetime64:
    dt = d.astype("datetime64[D]").astype(object)
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    import calendar

    day = min(dt.day, calendar.monthrange(y, m)[1])
    return np.datetime64(datetime.date(y, m, day), "D")


def _date_str(d: np.datetime64) -> str:
    return str(d.astype("datetime64[D]"))


def _date_i64(d: np.datetime64) -> int:
    return int(d.astype("datetime64[D]").view(np.int64))


def generate_sweep_params(n_dates: int) -> list[SweepParam]:
    span = int((_SWEEP_DATE_HI - _SWEEP_DATE_LO) / np.timedelta64(1, "D"))
    offsets = np.linspace(0, span, n_dates, dtype=int)
    dates = [
        (_SWEEP_DATE_LO + np.timedelta64(int(o), "D"))
        .astype("datetime64[M]")
        .astype("datetime64[D]")
        for o in offsets
    ]
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


# Which steps go to SQL vs Python, and which numpy opts are active.


@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_precompute: bool = True
    opt_presort: bool = True
    opt_encode_return: bool = True

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
# A in SQL requires R in SQL: pre-agg must follow the returnflag filter.
# G in SQL collapses rows - Python can't post-aggregate, so G forces R+D+A into SQL too.


def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("A") and cfg.python_handles("R"):
        return False, (
            "A in SQL requires R in SQL: pre-aggregation must follow the returnflag filter."
        )
    if cfg.sql_handles("G") and cfg.python_steps - {"G"}:
        missing = sorted(cfg.python_steps - {"G"})
        return False, (
            f"G in SQL requires R, D, A also in SQL (missing: {missing}). "
            "SQL GROUP BY collapses rows; Python cannot post-filter or re-aggregate."
        )
    return True, ""


# Build the SELECT/WHERE/JOIN dynamically - what SQL sees changes per combo.

_BASE_JOIN = (
    "FROM customer\n"
    "JOIN orders   ON c_custkey   = o_custkey\n"
    "JOIN lineitem ON l_orderkey  = o_orderkey\n"
    "JOIN nation   ON c_nationkey = n_nationkey"
)

_SELECT_COLS = (
    "c_custkey, c_name,\n"
    "  SUM(l_extendedprice * (1.0 - l_discount)) AS revenue,\n"
    "  c_acctbal, n_name, c_address, c_comment"
)

_GROUP_ORDER = (
    "GROUP BY c_custkey, c_name, c_acctbal, c_phone,\n"
    "         n_name, c_address, c_comment\n"
    "ORDER BY revenue DESC\n"
    f"LIMIT {LIMIT}"
)


def _date_where(d: np.datetime64, months: int) -> str:
    return (
        f"o_orderdate >= DATE '{_date_str(d)}' "
        f"AND o_orderdate < DATE '{_date_str(_month_add(d, months))}'"
    )


def build_pure_sql(d: np.datetime64, months: int) -> str:
    return (
        f"SELECT {_SELECT_COLS}\n"
        f"{_BASE_JOIN}\n"
        f"WHERE l_returnflag = 'R'\n"
        f"  AND {_date_where(d, months)}\n"
        f"{_GROUP_ORDER}"
    )


def build_fetch_sql_r_only(cfg: BenchConfig) -> str:
    """R in SQL, A+D+G in Python: fetch all returned-order rows, one row per lineitem."""
    assert cfg.sql_handles("R") and cfg.python_handles("A") and cfg.python_handles("D")
    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    return (
        f"SELECT c_custkey, c_name, {ep_col},\n"
        "  c_acctbal, n_name, c_address, c_comment, o_orderdate\n"
        f"{_BASE_JOIN}\n"
        "WHERE l_returnflag = 'R'"
    )


def build_fetch_sql_full(cfg: BenchConfig) -> str:
    """
    R, A, D all in Python: fetch rows pre-filtered to l_returnflag = 'R'.

    Non-returned lineitems contribute zero to Q10 revenue and are pure
    transfer waste - we always push this filter to SQL regardless of which
    step R is assigned to.  "R in Python" means Python *could* vary the
    filter value; in Q10 it is always 'R', so the SQL filter is safe.
    We omit the is_return column since every fetched row satisfies it.
    """
    assert (
        cfg.python_handles("R") and cfg.python_handles("A") and cfg.python_handles("D")
    )
    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    return (
        f"SELECT c_custkey, c_name, {ep_col},\n"
        "  c_acctbal, n_name, c_address, c_comment, o_orderdate\n"
        f"{_BASE_JOIN}\n"
        "WHERE l_returnflag = 'R'"
    )


def build_fetch_sql_preagg(cfg: BenchConfig) -> str:
    """
    R+A in SQL, D+G in Python: pre-aggregate by (customer, orderdate), no date filter.

    Returns one row per (customer, date) instead of one per lineitem - much smaller
    result for the sweep. Ordered by (orderdate, custkey) so Python can use
    searchsorted for date windows without a secondary sort.
    """
    assert cfg.sql_handles("R") and cfg.sql_handles("A") and cfg.python_handles("D")
    return (
        "SELECT c_custkey, c_name,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount)) AS rev_day,\n"
        "  c_acctbal, n_name, c_address, c_comment, o_orderdate\n"
        f"{_BASE_JOIN}\n"
        "WHERE l_returnflag = 'R'\n"
        "GROUP BY c_custkey, c_name, c_acctbal, c_phone,\n"
        "         n_name, c_address, c_comment, o_orderdate\n"
        "ORDER BY o_orderdate, c_custkey"
    )


def build_per_query_sql_d_only(cfg: BenchConfig, d: np.datetime64, months: int) -> str:
    """
    D in SQL, R+A+G in Python.
    We push l_returnflag='R' to SQL even here - non-R rows are waste.
    """
    assert cfg.sql_handles("D") and cfg.python_handles("R") and cfg.python_handles("A")
    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    return (
        f"SELECT c_custkey, c_name, {ep_col},\n"
        "  c_acctbal, n_name, c_address, c_comment\n"
        f"{_BASE_JOIN}\n"
        f"WHERE l_returnflag = 'R'\n"
        f"  AND {_date_where(d, months)}"
    )


def build_per_query_sql_rd(cfg: BenchConfig, d: np.datetime64, months: int) -> str:
    """R+D in SQL, A+G in Python: raw filtered rows for Python pre-agg + groupby."""
    assert cfg.sql_handles("R") and cfg.sql_handles("D") and cfg.python_handles("A")
    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    return (
        f"SELECT c_custkey, c_name, {ep_col},\n"
        "  c_acctbal, n_name, c_address, c_comment\n"
        f"{_BASE_JOIN}\n"
        f"WHERE l_returnflag = 'R'\n"
        f"  AND {_date_where(d, months)}"
    )


def build_per_query_sql_rda(cfg: BenchConfig, d: np.datetime64, months: int) -> str:
    """
    R+D+A in SQL, G in Python: pre-aggregate within the date window, one row per customer.

    SQL does the heavy GROUP BY; Python only needs argpartition for top-20, which is
    O(n) vs SQL's ORDER BY LIMIT (O(n log n)).  Ordered by c_custkey so the result
    is ready for _groupby_presorted without an inner argsort.
    """
    assert (
        cfg.sql_handles("R")
        and cfg.sql_handles("D")
        and cfg.sql_handles("A")
        and cfg.python_handles("G")
    )
    return (
        "SELECT c_custkey, c_name,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount)) AS rev_day,\n"
        "  c_acctbal, n_name, c_address, c_comment\n"
        f"{_BASE_JOIN}\n"
        f"WHERE l_returnflag = 'R'\n"
        f"  AND {_date_where(d, months)}\n"
        "GROUP BY c_custkey, c_name, c_acctbal, c_phone,\n"
        "         n_name, c_address, c_comment\n"
        "ORDER BY c_custkey"
    )


# Arrays holding the bulk-fetched 4-table join result.
@dataclass
class FetchedArrays:
    custkey: np.ndarray  # int64
    cname: np.ndarray  # object
    ep_disc: np.ndarray  # float64
    acctbal: np.ndarray  # float64
    nname: np.ndarray  # object
    address: np.ndarray  # object
    comment: np.ndarray  # object
    orderdate: np.ndarray  # int64 days-since-epoch
    is_return: np.ndarray | None  # bool (R in Python)
    # Presorted by (orderdate ASC, custkey ASC) - enables O(log n) searchsorted per query
    presorted: bool = False
    # Index where returned rows end (all rows[:r_end] have is_return=True)
    r_end: int = 0
    # Pre-aggregated: ep_disc holds SUM(ep*(1-disc)) per (customer, date), not per lineitem
    preagg: bool = False


def _to_i64(arr: np.ndarray) -> np.ndarray:
    return (
        arr.astype("datetime64[D]").view(np.int64)
        if arr.dtype.kind == "M"
        else arr.astype(np.int64)
    )


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    """Bulk fetch when D is in Python. Returns None when D is in SQL (per-query path)."""
    if cfg.sql_handles("D"):
        return None, 0.0

    # SQL={R,A}: pre-aggregate by (customer, date) - much smaller result for the sweep.
    # SQL ORDER BY (orderdate, custkey) makes it presorted for free.
    if cfg.sql_handles("R") and cfg.sql_handles("A"):
        sql = build_fetch_sql_preagg(cfg)
        t0 = time.perf_counter()
        raw = con.execute(sql).fetchnumpy()
        ft = time.perf_counter() - t0
        custkey = raw["c_custkey"].astype(np.int64)
        ep_disc = raw["rev_day"].astype(np.float64)
        acctbal = raw["c_acctbal"].astype(np.float64)
        orderdate = _to_i64(raw["o_orderdate"])
        return FetchedArrays(
            custkey=custkey,
            cname=raw["c_name"],
            ep_disc=ep_disc,
            acctbal=acctbal,
            nname=raw["n_name"],
            address=raw["c_address"],
            comment=raw["c_comment"],
            orderdate=orderdate,
            is_return=None,
            presorted=True,  # guaranteed by SQL ORDER BY o_orderdate, c_custkey
            r_end=len(custkey),
            preagg=True,
        ), ft

    # SQL={R} or SQL={}: fetch raw lineitem rows (one per returned lineitem).
    if cfg.sql_handles("R"):
        sql = build_fetch_sql_r_only(cfg)
    else:
        sql = build_fetch_sql_full(cfg)

    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    ft = time.perf_counter() - t0

    custkey = raw["c_custkey"].astype(np.int64)
    ep_disc = raw["ep_disc"].astype(np.float64)
    acctbal = raw["c_acctbal"].astype(np.float64)
    orderdate = _to_i64(raw["o_orderdate"])
    cname = raw["c_name"]
    nname = raw["n_name"]
    address = raw["c_address"]
    comment = raw["c_comment"]

    # All fetched rows are l_returnflag='R' (pushed to SQL in both fetch paths).
    # is_return is always True for every row - no need to store or filter it.
    is_return = None
    presorted = False
    r_end = len(custkey)

    if cfg.opt_presort:
        # Sort by orderdate ASC for O(log n) date-window searchsorted per query.
        # Secondary sort by custkey ASC so reduceat within the date slice needs
        # no inner argsort - the groupby is already in order.
        sort_idx = np.lexsort((custkey, orderdate))
        custkey = custkey[sort_idx]
        ep_disc = ep_disc[sort_idx]
        acctbal = acctbal[sort_idx]
        orderdate = orderdate[sort_idx]
        cname = cname[sort_idx]
        nname = nname[sort_idx]
        address = address[sort_idx]
        comment = comment[sort_idx]
        presorted = True
        r_end = len(custkey)

    return FetchedArrays(
        custkey=custkey,
        cname=cname,
        ep_disc=ep_disc,
        acctbal=acctbal,
        nname=nname,
        address=address,
        comment=comment,
        orderdate=orderdate,
        is_return=is_return,
        presorted=presorted,
        r_end=r_end,
        preagg=False,
    ), ft


# Group by customer and pick top-20 by revenue - np.argpartition keeps this O(n).
def _groupby_custkey_top20(
    custkey: np.ndarray,
    ep_disc: np.ndarray,
    cname: np.ndarray,
    acctbal: np.ndarray,
    nname: np.ndarray,
    address: np.ndarray,
    comment: np.ndarray,
) -> Q10Result:
    """
    Group by custkey, sum ep_disc, return top-20 by revenue DESC.
    Uses np.unique + np.add.reduceat for groupby, np.argpartition for top-20.
    """
    if len(custkey) == 0:
        return []

    sort_idx = np.argsort(custkey, kind="stable")
    ck_s = custkey[sort_idx]
    ep_s = ep_disc[sort_idx]
    cn_s = cname[sort_idx]
    ab_s = acctbal[sort_idx]
    nn_s = nname[sort_idx]
    ad_s = address[sort_idx]
    co_s = comment[sort_idx]

    unique_ck, starts, _ = np.unique(ck_s, return_index=True, return_counts=True)
    rev = np.add.reduceat(ep_s, starts)

    # Top-20 by revenue DESC using argpartition (O(n) vs O(n log n) full sort)
    n = len(unique_ck)
    if n <= LIMIT:
        top_idx = np.arange(n)
    else:
        top_idx = np.argpartition(-rev, LIMIT)[:LIMIT]
    top_idx = top_idx[np.argsort(-rev[top_idx])]

    result: Q10Result = []
    for i in top_idx:
        si = starts[i]
        result.append(
            (
                int(unique_ck[i]),
                str(cn_s[si]),
                round(float(rev[i]), 4),
                round(float(ab_s[si]), 2),
                str(nn_s[si]),
                str(ad_s[si]),
                str(co_s[si]),
            )
        )
    return result


def _groupby_presorted(
    custkey: np.ndarray,
    ep_disc: np.ndarray,
    cname: np.ndarray,
    acctbal: np.ndarray,
    nname: np.ndarray,
    address: np.ndarray,
    comment: np.ndarray,
) -> Q10Result:
    """
    Fast groupby when input is already sorted by custkey (presorted path).
    Skips the inner argsort - just unique + reduceat directly.
    """
    if len(custkey) == 0:
        return []
    unique_ck, starts, _ = np.unique(custkey, return_index=True, return_counts=True)
    rev = np.add.reduceat(ep_disc, starts)

    n = len(unique_ck)
    if n <= LIMIT:
        top_idx = np.arange(n)
    else:
        top_idx = np.argpartition(-rev, LIMIT)[:LIMIT]
    top_idx = top_idx[np.argsort(-rev[top_idx])]

    result: Q10Result = []
    for i in top_idx:
        si = starts[i]
        result.append(
            (
                int(unique_ck[i]),
                str(cname[si]),
                round(float(rev[i]), 4),
                round(float(acctbal[si]), 2),
                str(nname[si]),
                str(address[si]),
                str(comment[si]),
            )
        )
    return result


def _numpy_query(
    arrays: FetchedArrays,
    d: np.datetime64,
    months: int,
) -> Q10Result:
    lo_i64 = _date_i64(d)
    hi_i64 = _date_i64(_month_add(d, months))

    if arrays.presorted:
        # Arrays sorted by (orderdate ASC, custkey ASC).
        # Date window is a contiguous slice - searchsorted on orderdate.
        # Within the slice, custkey is NOT globally sorted (custkey sort is
        # secondary to orderdate), so we still need a groupby sort.
        # However, using np.unique directly is faster than a full argsort
        # because np.unique uses a sort internally but only on the slice.
        lo_idx = int(np.searchsorted(arrays.orderdate, lo_i64, side="left"))
        hi_idx = int(np.searchsorted(arrays.orderdate, hi_i64, side="left"))
        if lo_idx >= hi_idx:
            return []
        sl = slice(lo_idx, hi_idx)
        return _groupby_custkey_top20(
            arrays.custkey[sl],
            arrays.ep_disc[sl],
            arrays.cname[sl],
            arrays.acctbal[sl],
            arrays.nname[sl],
            arrays.address[sl],
            arrays.comment[sl],
        )

    # Unsorted fallback - also no returnflag filter needed (always R)
    date_mask = (arrays.orderdate >= lo_i64) & (arrays.orderdate < hi_i64)
    if not date_mask.any():
        return []
    return _groupby_custkey_top20(
        arrays.custkey[date_mask],
        arrays.ep_disc[date_mask],
        arrays.cname[date_mask],
        arrays.acctbal[date_mask],
        arrays.nname[date_mask],
        arrays.address[date_mask],
        arrays.comment[date_mask],
    )


# Route each query to the right path depending on which steps SQL owns.
def _top20_from_grouped(
    custkey: np.ndarray,
    rev: np.ndarray,
    cname: np.ndarray,
    acctbal: np.ndarray,
    nname: np.ndarray,
    address: np.ndarray,
    comment: np.ndarray,
) -> Q10Result:
    """
    Top-20 from already-grouped-by-customer data (one row per customer).

    Used when SQL does R+D+A (GROUP BY customer within the date window) and Python
    only needs to pick the top-20 by argpartition - O(n) vs SQL ORDER BY LIMIT O(n log n).
    """
    n = len(custkey)
    if n == 0:
        return []
    if n <= LIMIT:
        top_idx = np.arange(n)
    else:
        top_idx = np.argpartition(-rev, LIMIT)[:LIMIT]
    top_idx = top_idx[np.argsort(-rev[top_idx])]
    return [
        (
            int(custkey[i]),
            str(cname[i]),
            round(float(rev[i]), 4),
            round(float(acctbal[i]), 2),
            str(nname[i]),
            str(address[i]),
            str(comment[i]),
        )
        for i in top_idx
    ]


def _rows_to_result(raw: dict) -> Q10Result:
    """Group per-row SQL result by custkey and return top-20."""
    return _groupby_custkey_top20(
        raw["c_custkey"].astype(np.int64),
        raw["ep_disc"].astype(np.float64),
        raw["c_name"],
        raw["c_acctbal"].astype(np.float64),
        raw["n_name"],
        raw["c_address"],
        raw["c_comment"],
    )


def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    d: np.datetime64,
    months: int,
) -> Q10Result:

    # Pure SQL: R+D+A+G all in SQL
    if all(cfg.sql_handles(s) for s in ("R", "D", "A", "G")):
        rows = con.execute(build_pure_sql(d, months)).fetchall()
        return [
            (
                int(r[0]),
                str(r[1]),
                round(float(r[2]), 4),
                round(float(r[3]), 2),
                str(r[4]),
                str(r[5]),
                str(r[6]),
            )
            for r in rows
        ]

    # R+D+A in SQL, G in Python: SQL pre-aggs by customer within the window;
    # Python picks top-20 with argpartition (O(n)) instead of SQL ORDER BY (O(n log n)).
    if (
        cfg.sql_handles("R")
        and cfg.sql_handles("D")
        and cfg.sql_handles("A")
        and cfg.python_handles("G")
    ):
        raw = con.execute(build_per_query_sql_rda(cfg, d, months)).fetchnumpy()
        return _top20_from_grouped(
            raw["c_custkey"].astype(np.int64),
            raw["rev_day"].astype(np.float64),
            raw["c_name"],
            raw["c_acctbal"].astype(np.float64),
            raw["n_name"],
            raw["c_address"],
            raw["c_comment"],
        )

    # R+D in SQL, A+G in Python: raw filtered rows, Python does pre-agg + groupby
    if cfg.sql_handles("R") and cfg.sql_handles("D") and cfg.python_handles("A"):
        raw = con.execute(build_per_query_sql_rd(cfg, d, months)).fetchnumpy()
        return _rows_to_result(raw)

    # D in SQL only (+ R always pushed), A+G in Python
    if cfg.sql_handles("D") and cfg.python_handles("R") and cfg.python_handles("A"):
        raw = con.execute(build_per_query_sql_d_only(cfg, d, months)).fetchnumpy()
        if len(raw.get("c_custkey", [])) == 0:
            return []
        return _rows_to_result(raw)

    # D in Python - use pre-fetched arrays (preagg or raw, _numpy_query handles both)
    assert arrays is not None
    return _numpy_query(arrays, d, months)


# Orchestrate fetch + N-query loop for one combo; return a timed SweepResult.


def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[SweepParam],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key}: {reason}")

    arrays, fetch_time = fetch_and_prepare(con, cfg)

    results = []
    t1 = time.perf_counter()
    for d, months in params:
        results.append(run_one_query(con, cfg, arrays, d, months))
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


# Spot-check Python results against the SQL reference - catches silent numerical drift.
def _rows_match(a: Q10Row, b: Q10Row, tol: float = 1.0) -> bool:
    return a[0] == b[0] and abs(a[2] - b[2]) <= tol and abs(a[3] - b[3]) <= 0.01


def validate(
    reference: list[Q10Result],
    candidate: list[Q10Result],
    params: list[SweepParam],
    label: str,
) -> bool:
    mismatches = []
    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        if len(ref) != len(cand):
            mismatches.append((params[i], ref, cand, "length"))
            continue
        for rr, rc in zip(ref, cand):
            if not _rows_match(rr, rc):
                mismatches.append((params[i], ref, cand, "value"))
                break
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, ref, cand, kind in mismatches[:2]:
            logger.warning(
                f"    date={_date_str(p[0])} months={p[1]} ({kind}): "
                f"ref_rows={len(ref)} got_rows={len(cand)}"
            )
            for rr, rc in zip(ref[:2], cand[:2]):
                logger.warning(f"      ref: {rr[:4]}")
                logger.warning(f"      got: {rc[:4]}")
        return False
    logger.info(f"  [{label}]  all results match reference")
    return True


# Plug Q10 into the shared benchmark harness.
class Q10Benchmark(QueryBenchmark):
    NAME = "Q10"
    ALL_STEPS = ("R", "D", "A", "G")
    N_APPLICABLE = True
    N_HELP = "Number of start dates (default: 10); total params = n * 3 window widths"
    N_DEFAULT = 10

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_precompute=args.opt_precompute,
            opt_presort=args.opt_presort,
            opt_encode_return=args.opt_encode_return,
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
            f"precompute={'ON' if args.opt_precompute else 'OFF'}  "
            f"presort={'ON' if args.opt_presort else 'OFF'}  "
            f"encode_return={'ON' if args.opt_encode_return else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_precompute",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Precompute ep*(1-disc) at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort by (is_return DESC, orderdate ASC) at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_return",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Encode returnflag as bool at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        n = args.n if args.n is not None else self.N_DEFAULT
        return (
            f"sweep  N={len(params)} params "
            f"({min(n, len(params) // len(ALL_WIDTHS))} dates * {len(ALL_WIDTHS)} widths)"
        )


# Run directly: python3 -m benchmarks.q10
def main() -> None:
    bench = Q10Benchmark()
    parser = make_base_parser(
        "Q10 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q10_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
