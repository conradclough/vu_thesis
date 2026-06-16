"""
Q18 - large-volume orders (sum_qty > threshold), sweep over threshold values in [250, 450].

    SELECT c_name, c_custkey, o_orderkey, o_orderdate,
           o_totalprice, SUM(l_quantity)
    FROM customer, orders, lineitem
    WHERE o_orderkey IN (
        SELECT l_orderkey
        FROM lineitem
        GROUP BY l_orderkey
        HAVING SUM(l_quantity) > [THRESHOLD]
    )
      AND c_custkey = o_custkey
      AND o_orderkey = l_orderkey
    GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
    ORDER BY o_totalprice DESC, o_orderdate
    LIMIT 100

SQL must re-scan all lineitem rows to compute per-orderkey quantity sums for every threshold value.
Python precomputes those sums once at fetch time (sort by orderkey + reduceat) and sweeps the threshold as a single
boolean comparison - O(unique_orderkeys) per query instead of O(total_lineitem_rows).

Steps:
  [T] threshold : HAVING SUM(l_quantity) > threshold (the subquery)
  [J] semi-join : filter the orders join to rows whose orderkey qualifies
  [G] group+agg : GROUP BY order + SUM(l_quantity) per qualifying order + sort

Validity: G collapses rows - Python can't post-aggregate, so G in SQL forces T and J
into SQL too. J in SQL requires T also in SQL (SQL needs the qualifying set to do the
semi-join; if T is in Python, SQL has no set to join against). J in Python with T in SQL
is fine: SQL returns (orderkey, sum_qty) pairs; Python thresholds them and filters orders.

Full Python path makes two bulk fetches: lineitem for T, and the orders*customer*lineitem
join for J+G. Per-query cost after that is a boolean mask + np.isin - essentially free.

Opt flags (all ON by default):
  --opt_precompute_sums   sort lineitem by orderkey, reduceat → per-key qty sums at fetch
  --opt_presort_orders    sort orders by (totalprice DESC, orderdate) → result already sorted

Usage:
  python3 -m benchmarks.benchmark_sweep q18 --sf 1 --repeats 5
  python3 -m benchmarks.benchmark_sweep q18 --no_opt_precompute_sums --no_opt_presort_orders
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass, field

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

# T/J/G - the three steps and all valid subsets.

ALL_STEPS = ("T", "J", "G")  # threshold subquery, semi-join, group+agg

# Threshold range and result shape from the TPC-H Q18 spec.

SINGLE_THRESHOLD = 300
THRESHOLD_LO = 250
THRESHOLD_HI = 450

# One result row per qualifying order: (c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum_qty).

# Each row: (c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum_qty)
Q18Row = tuple[str, int, int, str, float, float]
Q18Result = list[Q18Row]


# N evenly-spaced threshold values over [250, 450].
def generate_thresholds(n: int) -> list[int]:
    return [int(x) for x in np.linspace(THRESHOLD_LO, THRESHOLD_HI, n, dtype=int)]


def single_query_params() -> list[int]:
    return [SINGLE_THRESHOLD]


# Which steps go to SQL vs Python, and which numpy opts are active.
@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_precompute_sums: bool = True  # per-orderkey qty sums at fetch time
    opt_presort_orders: bool = True  # sort orders by (totalprice DESC, date)

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


# G in SQL collapses rows - Python can't post-aggregate, so G forces T+J. J in SQL requires T in SQL (the qualifying set).
def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    """
    All combos are valid via dynamic IN-list injection.

    When T is in Python: Python computes per-orderkey qty sums, applies the
    threshold, and produces a qualifying key set.  That set is injected into
    SQL as a literal IN list (e.g. WHERE o_orderkey IN (1,2,3,...)) so SQL
    can perform J and/or G without knowing the threshold at query-build time.

    The only genuinely invalid combo is T+G in SQL with J in Python:
    SQL runs the subquery AND the GROUP BY, but with J in Python SQL would
    need to return ungrouped rows for Python to filter - contradicting G
    being in SQL.  Python cannot filter rows that SQL has already collapsed
    into group aggregates.
    """
    if cfg.sql_handles("T") and cfg.sql_handles("G") and cfg.python_handles("J"):
        return False, (
            "T+G in SQL with J in Python is contradictory: SQL would need to "
            "both aggregate (G) and leave rows ungrouped for Python to filter (J). "
            "Cannot have G in SQL without J applied first."
        )
    return True, ""


# Build the SELECT/WHERE/subquery dynamically - what SQL sees changes per combo.

_OUTER_JOIN = """\
FROM customer
JOIN orders   ON c_custkey  = o_custkey
JOIN lineitem ON o_orderkey = l_orderkey"""

_SUBQUERY = "(\n    SELECT l_orderkey\n    FROM lineitem\n    GROUP BY l_orderkey\n    HAVING SUM(l_quantity) > {threshold}\n)"


def build_pure_sql(threshold: int) -> str:
    """Full Q18 in SQL."""
    return (
        "SELECT c_name, c_custkey, o_orderkey,\n"
        "       o_orderdate, o_totalprice,\n"
        "       SUM(l_quantity) AS sum_qty\n"
        f"{_OUTER_JOIN}\n"
        f"WHERE o_orderkey IN {_SUBQUERY.format(threshold=threshold)}\n"
        "GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice\n"
        "ORDER BY o_totalprice DESC, o_orderdate"
    )


def build_sql_tj_g_python(threshold: int) -> str:
    """T+J in SQL, G in Python: SQL returns filtered per-row lineitem data."""
    return (
        "SELECT c_name, c_custkey, o_orderkey,\n"
        "       o_orderdate, o_totalprice, l_quantity\n"
        f"{_OUTER_JOIN}\n"
        f"WHERE o_orderkey IN {_SUBQUERY.format(threshold=threshold)}"
    )


def build_sql_t_only() -> str:
    """T in SQL: return per-orderkey quantity sums from lineitem subquery."""
    return (
        "SELECT l_orderkey, SUM(l_quantity) AS sum_qty\n"
        "FROM lineitem\n"
        "GROUP BY l_orderkey"
    )


def build_sql_orders_fetch() -> str:
    """
    Bulk fetch of orders+customer+lineitem for the J+G-in-Python paths.
    Returns all rows; Python applies the qualifying-key filter per threshold.
    """
    return (
        "SELECT c_name, c_custkey, o_orderkey,\n"
        "       o_orderdate, o_totalprice, l_quantity\n"
        f"{_OUTER_JOIN}"
    )


def _in_list(keys: np.ndarray) -> str:
    """Render an int64 array as a SQL IN list literal: (1,2,3,...)."""
    if len(keys) == 0:
        return "(NULL)"  # IN (NULL) matches nothing
    return "(" + ",".join(str(int(k)) for k in keys) + ")"


def build_sql_j_only(qualifying_keys: np.ndarray) -> str:
    """
    J in SQL, T+G in Python: SQL filters orders to qualifying keys via
    a dynamic IN list; returns per-row data for Python to group+aggregate.
    """
    return (
        "SELECT c_name, c_custkey, o_orderkey,\n"
        "       o_orderdate, o_totalprice, l_quantity\n"
        f"{_OUTER_JOIN}\n"
        f"WHERE o_orderkey IN {_in_list(qualifying_keys)}"
    )


def build_sql_jg_only(qualifying_keys: np.ndarray) -> str:
    """
    J+G in SQL, T in Python: SQL filters by IN list and does the GROUP BY.
    Returns one row per qualifying order.
    """
    return (
        "SELECT c_name, c_custkey, o_orderkey,\n"
        "       o_orderdate, o_totalprice,\n"
        "       SUM(l_quantity) AS sum_qty\n"
        f"{_OUTER_JOIN}\n"
        f"WHERE o_orderkey IN {_in_list(qualifying_keys)}\n"
        "GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice\n"
        "ORDER BY o_totalprice DESC, o_orderdate"
    )


def build_sql_g_only(qualifying_keys: np.ndarray) -> str:
    """
    G in SQL, T+J in Python (qualifying_keys already filtered by Python):
    same as build_sql_jg_only - SQL receives the already-filtered key set
    and performs the GROUP BY.  Structurally identical to JG-in-SQL since
    the IN list encodes the J step.
    """
    return build_sql_jg_only(qualifying_keys)


# Arrays holding bulk-fetched lineitem sums and orders join, with optional precomputed aggregates.
@dataclass
class LineitemSums:
    """Per-orderkey quantity sums, sorted by orderkey for fast np.isin."""

    orderkeys: np.ndarray  # int64, sorted ascending
    qty_sums: np.ndarray  # float64, qty_sums[i] = SUM(l_quantity) for orderkeys[i]


@dataclass
class OrdersArrays:
    """
    Bulk-fetched orders+customer+lineitem rows, optionally presorted.
    Used when J and G are in Python.
    """

    c_name: np.ndarray  # object array of strings
    c_custkey: np.ndarray  # int64
    o_orderkey: np.ndarray  # int64
    o_orderdate: np.ndarray  # object array of date strings (keep as str for output)
    o_totalprice: np.ndarray  # float64
    l_quantity: np.ndarray  # float64

    # Per-order aggregates (precomputed when opt_precompute_sums=True)
    # order_sum_qty[i] = SUM(l_quantity) for rows where o_orderkey == unique_orderkeys[i]
    unique_orderkeys: np.ndarray | None = None  # int64, sorted
    order_sum_qty: np.ndarray | None = None  # float64
    order_totalprice: np.ndarray | None = None  # float64, one per unique order
    order_orderdate: np.ndarray | None = None  # object, one per unique order
    order_custkey: np.ndarray | None = None  # int64
    order_cname: np.ndarray | None = None  # object

    presorted: bool = False  # sorted by (totalprice DESC, orderdate ASC)


def _fetchnumpy_lineitem_sums(con: duckdb.DuckDBPyConnection) -> LineitemSums:
    """Fetch and compute per-orderkey quantity sums from lineitem."""
    raw = con.execute("SELECT l_orderkey, l_quantity FROM lineitem").fetchnumpy()
    okey = raw["l_orderkey"].astype(np.int64)
    qty = raw["l_quantity"].astype(np.float64)

    sort_idx = np.argsort(okey, kind="stable")
    okey_s = okey[sort_idx]
    qty_s = qty[sort_idx]

    unique_keys, starts, counts = np.unique(
        okey_s, return_index=True, return_counts=True
    )
    sums = np.add.reduceat(qty_s, starts)
    return LineitemSums(orderkeys=unique_keys, qty_sums=sums)


def _fetchnumpy_orders(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> OrdersArrays:
    """Bulk fetch orders+customer+lineitem rows."""
    raw = con.execute(build_sql_orders_fetch()).fetchnumpy()

    c_name = raw["c_name"]
    c_custkey = raw["c_custkey"].astype(np.int64)
    o_orderkey = raw["o_orderkey"].astype(np.int64)
    o_orderdate = raw["o_orderdate"]
    o_totalprice = raw["o_totalprice"].astype(np.float64)
    l_quantity = raw["l_quantity"].astype(np.float64)

    # Sort lineitem rows by orderkey and reduceat -> per-order sum_qty in one pass.
    # DuckDB recomputes this subquery scan for every threshold value; we pay it once.
    unique_orderkeys = order_sum_qty = order_totalprice = None
    order_orderdate = order_custkey = order_cname = None

    if cfg.opt_precompute_sums:
        sort_idx = np.argsort(o_orderkey, kind="stable")
        ok_s = o_orderkey[sort_idx]
        qty_s = l_quantity[sort_idx]
        tp_s = o_totalprice[sort_idx]
        od_s = o_orderdate[sort_idx]
        ck_s = c_custkey[sort_idx]
        cn_s = c_name[sort_idx]

        unique_orderkeys, starts, counts = np.unique(
            ok_s, return_index=True, return_counts=True
        )
        order_sum_qty = np.add.reduceat(qty_s, starts)
        # Scalar-per-order fields: one value per unique orderkey, take the first row.
        order_totalprice = tp_s[starts]
        order_orderdate = od_s[starts]
        order_custkey = ck_s[starts]
        order_cname = cn_s[starts]

    # Sort per-order arrays by (totalprice DESC, orderdate ASC) now so per-query
    # result needs no sort - just filter by threshold and return the already-ordered rows.
    presorted = False
    if cfg.opt_presort_orders and unique_orderkeys is not None:
        od_raw = order_orderdate
        if od_raw.dtype.kind == "M":
            od_i64 = od_raw.astype("datetime64[D]").view(np.int64)
        else:
            od_i64 = np.array(
                [int(np.datetime64(str(d), "D").view(np.int64)) for d in od_raw],
                dtype=np.int64,
            )
        sort_idx = np.lexsort((od_i64, -order_totalprice))
        unique_orderkeys = unique_orderkeys[sort_idx]
        order_sum_qty = order_sum_qty[sort_idx]
        order_totalprice = order_totalprice[sort_idx]
        order_orderdate = order_orderdate[sort_idx]
        order_custkey = order_custkey[sort_idx]
        order_cname = order_cname[sort_idx]
        presorted = True

    return OrdersArrays(
        c_name=c_name,
        c_custkey=c_custkey,
        o_orderkey=o_orderkey,
        o_orderdate=o_orderdate,
        o_totalprice=o_totalprice,
        l_quantity=l_quantity,
        unique_orderkeys=unique_orderkeys,
        order_sum_qty=order_sum_qty,
        order_totalprice=order_totalprice,
        order_orderdate=order_orderdate,
        order_custkey=order_custkey,
        order_cname=order_cname,
        presorted=presorted,
    )


# N tight loops - no SQL calls from here. The bulk fetch bought us that.
def _format_date(d) -> str:
    """Convert numpy datetime64 or string to YYYY-MM-DD string."""
    if hasattr(d, "astype"):
        return str(d.astype("datetime64[D]"))
    return str(d)[:10]


def _numpy_query_precomputed(
    li_sums: LineitemSums,
    orders: OrdersArrays,
    threshold: int,
) -> Q18Result:
    """
    Per-query path for the (none) combo.

    Fast path (opt_precompute_sums=ON): per-order arrays were built at fetch
    time, so we filter unique_orderkeys via isin and read scalar fields by index.

    Fallback (opt_precompute_sums=OFF): no per-order arrays were precomputed;
    we filter the raw lineitem-level join rows by qualifying_keys, then
    argsort + reduceat to compute SUM(l_quantity) per order on the fly. This
    is slower but correct - without it, np.isin(None, ...) silently returns an
    empty mask and every query returns [].
    """
    qualifying_mask = li_sums.qty_sums > threshold
    qualifying_keys = li_sums.orderkeys[qualifying_mask]

    if len(qualifying_keys) == 0:
        return []

    if orders.unique_orderkeys is not None:
        # Fast path: per-order arrays precomputed at fetch time.
        uk = orders.unique_orderkeys
        in_mask = np.isin(uk, qualifying_keys)

        if not in_mask.any():
            return []

        result: Q18Result = []
        for i in np.where(in_mask)[0]:
            okey = int(uk[i])
            idx = int(np.searchsorted(li_sums.orderkeys, okey))
            sq = float(li_sums.qty_sums[idx])
            result.append(
                (
                    str(orders.order_cname[i]),
                    int(orders.order_custkey[i]),
                    okey,
                    _format_date(orders.order_orderdate[i]),
                    round(float(orders.order_totalprice[i]), 2),
                    round(sq, 2),
                )
            )

        if not orders.presorted:
            result.sort(key=lambda r: (-r[4], r[3]))
        return result

    # Fallback: precompute_sums was OFF. Group raw rows per query.
    in_mask = np.isin(orders.o_orderkey, qualifying_keys)
    if not in_mask.any():
        return []

    ok_m = orders.o_orderkey[in_mask]
    qty_m = orders.l_quantity[in_mask]
    tp_m = orders.o_totalprice[in_mask]
    od_m = orders.o_orderdate[in_mask]
    ck_m = orders.c_custkey[in_mask]
    cn_m = orders.c_name[in_mask]

    sort_idx = np.argsort(ok_m, kind="stable")
    ok_s = ok_m[sort_idx]
    qty_s = qty_m[sort_idx]
    tp_s = tp_m[sort_idx]
    od_s = od_m[sort_idx]
    ck_s = ck_m[sort_idx]
    cn_s = cn_m[sort_idx]

    unique_ok, starts, _ = np.unique(ok_s, return_index=True, return_counts=True)
    sum_qty = np.add.reduceat(qty_s, starts)
    result: Q18Result = []
    for i, okey in enumerate(unique_ok):
        result.append(
            (
                str(cn_s[starts[i]]),
                int(ck_s[starts[i]]),
                int(okey),
                _format_date(od_s[starts[i]]),
                round(float(tp_s[starts[i]]), 2),
                round(float(sum_qty[i]), 2),
            )
        )
    result.sort(key=lambda r: (-r[4], r[3]))
    return result


def _numpy_query_t_in_sql(
    key_sums_raw: dict,
    orders: OrdersArrays,
    threshold: int,
) -> Q18Result:
    """
    T in SQL, J+G in Python: SQL returned (orderkey, sum_qty) pairs.
    Python applies threshold, filters orders, aggregates.
    """
    okey_sql = key_sums_raw["l_orderkey"].astype(np.int64)
    qty_sql = key_sums_raw["sum_qty"].astype(np.float64)

    qualifying_mask = qty_sql > threshold
    qualifying_keys = okey_sql[qualifying_mask]
    qualifying_qty = qty_sql[qualifying_mask]

    if len(qualifying_keys) == 0:
        return []

    if orders.unique_orderkeys is not None:
        uk = orders.unique_orderkeys
        in_mask = np.isin(uk, qualifying_keys)
        # Build qty lookup from SQL result
        qty_lookup = dict(zip(qualifying_keys.tolist(), qualifying_qty.tolist()))

        result: Q18Result = []
        for i in np.where(in_mask)[0]:
            okey = int(uk[i])
            result.append(
                (
                    str(orders.order_cname[i]),
                    int(orders.order_custkey[i]),
                    okey,
                    _format_date(orders.order_orderdate[i]),
                    round(float(orders.order_totalprice[i]), 2),
                    round(qty_lookup.get(okey, 0.0), 2),
                )
            )
        if not orders.presorted:
            result.sort(key=lambda r: (-r[4], r[3]))
        return result

    # Fallback: no precomputed per-order arrays
    ok = orders.o_orderkey
    in_mask = np.isin(ok, qualifying_keys)
    if not in_mask.any():
        return []

    qty_lookup = dict(zip(qualifying_keys.tolist(), qualifying_qty.tolist()))
    ok_m = ok[in_mask]
    tp_m = orders.o_totalprice[in_mask]
    od_m = orders.o_orderdate[in_mask]
    ck_m = orders.c_custkey[in_mask]
    cn_m = orders.c_name[in_mask]

    # Group by orderkey (take first row for scalar fields, sum qty from SQL)
    unique_ok, first_idx = np.unique(ok_m, return_index=True)
    result = []
    for j, okey in enumerate(unique_ok):
        fi = first_idx[j]
        result.append(
            (
                str(cn_m[fi]),
                int(ck_m[fi]),
                int(okey),
                _format_date(od_m[fi]),
                round(float(tp_m[fi]), 2),
                round(qty_lookup.get(int(okey), 0.0), 2),
            )
        )
    result.sort(key=lambda r: (-r[4], r[3]))
    return result


# Route each query to the right path depending on which steps SQL owns.
def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    li_sums: LineitemSums | None,
    orders: OrdersArrays | None,
    key_sums_raw: dict | None,
    threshold: int,
) -> Q18Result:

    # All steps in SQL - pure DuckDB, no Python involvement.
    if cfg.sql_handles("T") and cfg.sql_handles("J") and cfg.sql_handles("G"):
        rows = con.execute(build_pure_sql(threshold)).fetchall()
        return [
            (
                str(r[0]),
                int(r[1]),
                int(r[2]),
                _format_date(r[3]),
                round(float(r[4]), 2),
                round(float(r[5]), 2),
            )
            for r in rows
        ]

    # T+J in SQL, G in Python: SQL returns filtered per-row data; Python groups.
    if cfg.sql_handles("T") and cfg.sql_handles("J") and cfg.python_handles("G"):
        raw = con.execute(build_sql_tj_g_python(threshold)).fetchnumpy()
        ok = raw["o_orderkey"].astype(np.int64)
        qty = raw["l_quantity"].astype(np.float64)
        tp = raw["o_totalprice"].astype(np.float64)
        od = raw["o_orderdate"]
        ck = raw["c_custkey"].astype(np.int64)
        cn = raw["c_name"]

        sort_idx = np.argsort(ok, kind="stable")
        ok_s = ok[sort_idx]
        qty_s = qty[sort_idx]
        tp_s = tp[sort_idx]
        od_s = od[sort_idx]
        ck_s = ck[sort_idx]
        cn_s = cn[sort_idx]

        unique_ok, starts, _ = np.unique(ok_s, return_index=True, return_counts=True)
        sum_qty = np.add.reduceat(qty_s, starts)
        result: Q18Result = []
        for i, okey in enumerate(unique_ok):
            result.append(
                (
                    str(cn_s[starts[i]]),
                    int(ck_s[starts[i]]),
                    int(okey),
                    _format_date(od_s[starts[i]]),
                    round(float(tp_s[starts[i]]), 2),
                    round(float(sum_qty[i]), 2),
                )
            )
        result.sort(key=lambda r: (-r[4], r[3]))
        return result

    # T in SQL, J+G in Python: SQL returns (orderkey, sum_qty) per order; Python thresholds.
    if cfg.sql_handles("T") and cfg.python_handles("J"):
        assert key_sums_raw is not None and orders is not None
        return _numpy_query_t_in_sql(key_sums_raw, orders, threshold)

    # T in Python, J in SQL, G in Python: inject qualifying keys as IN list; SQL filters.
    if cfg.python_handles("T") and cfg.sql_handles("J") and cfg.python_handles("G"):
        assert li_sums is not None
        qualifying_mask = li_sums.qty_sums > threshold
        qualifying_keys = li_sums.orderkeys[qualifying_mask]
        if len(qualifying_keys) == 0:
            return []
        raw = con.execute(build_sql_j_only(qualifying_keys)).fetchnumpy()
        ok = raw["o_orderkey"].astype(np.int64)
        qty = raw["l_quantity"].astype(np.float64)
        tp = raw["o_totalprice"].astype(np.float64)
        od = raw["o_orderdate"]
        ck = raw["c_custkey"].astype(np.int64)
        cn = raw["c_name"]
        sort_idx = np.argsort(ok, kind="stable")
        ok_s = ok[sort_idx]
        qty_s = qty[sort_idx]
        tp_s = tp[sort_idx]
        od_s = od[sort_idx]
        ck_s = ck[sort_idx]
        cn_s = cn[sort_idx]
        unique_ok, starts, _ = np.unique(ok_s, return_index=True, return_counts=True)
        sum_qty = np.add.reduceat(qty_s, starts)
        result: Q18Result = []
        for i, okey in enumerate(unique_ok):
            result.append(
                (
                    str(cn_s[starts[i]]),
                    int(ck_s[starts[i]]),
                    int(okey),
                    _format_date(od_s[starts[i]]),
                    round(float(tp_s[starts[i]]), 2),
                    round(float(sum_qty[i]), 2),
                )
            )
        result.sort(key=lambda r: (-r[4], r[3]))
        return result

    # T in Python, J+G in SQL: inject qualifying keys; SQL filters + groups.
    if cfg.python_handles("T") and cfg.sql_handles("J") and cfg.sql_handles("G"):
        assert li_sums is not None
        qualifying_mask = li_sums.qty_sums > threshold
        qualifying_keys = li_sums.orderkeys[qualifying_mask]
        if len(qualifying_keys) == 0:
            return []
        rows = con.execute(build_sql_jg_only(qualifying_keys)).fetchall()
        return [
            (
                str(r[0]),
                int(r[1]),
                int(r[2]),
                _format_date(r[3]),
                round(float(r[4]), 2),
                round(float(r[5]), 2),
            )
            for r in rows
        ]

    # T+J in Python, G in SQL: Python thresholds + filters; inject keys for SQL GROUP BY.
    if cfg.python_handles("T") and cfg.python_handles("J") and cfg.sql_handles("G"):
        assert li_sums is not None
        qualifying_mask = li_sums.qty_sums > threshold
        qualifying_keys = li_sums.orderkeys[qualifying_mask]
        if len(qualifying_keys) == 0:
            return []
        rows = con.execute(build_sql_g_only(qualifying_keys)).fetchall()
        return [
            (
                str(r[0]),
                int(r[1]),
                int(r[2]),
                _format_date(r[3]),
                round(float(r[4]), 2),
                round(float(r[5]), 2),
            )
            for r in rows
        ]

    # All in Python: use precomputed sums and presorted orders - no SQL calls.
    assert li_sums is not None and orders is not None
    return _numpy_query_precomputed(li_sums, orders, threshold)


# Orchestrate fetches + N-query loop for one combo; return a timed SweepResult.
def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[int],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key}: {reason}")

    fetch_time = 0.0
    li_sums: LineitemSums | None = None
    orders: OrdersArrays | None = None
    key_sums_raw: dict | None = None

    # Bulk fetches (once, outside the sweep loop)
    if cfg.python_handles("T"):
        # Need lineitem sums for threshold evaluation (T, J, G, JG in SQL all use this)
        t0 = time.perf_counter()
        li_sums = _fetchnumpy_lineitem_sums(con)
        fetch_time += time.perf_counter() - t0

    if cfg.python_handles("J") and cfg.python_handles("G"):
        # Full orders bulk-fetch only needed when both J and G are in Python
        t0 = time.perf_counter()
        orders = _fetchnumpy_orders(con, cfg)
        fetch_time += time.perf_counter() - t0

    if cfg.sql_handles("T") and cfg.python_handles("J"):
        # T in SQL: fetch per-orderkey sums once; threshold applied per query in Python
        t0 = time.perf_counter()
        key_sums_raw = con.execute(build_sql_t_only()).fetchnumpy()
        key_sums_raw = {
            "l_orderkey": key_sums_raw["l_orderkey"].astype(np.int64),
            "sum_qty": key_sums_raw["sum_qty"].astype(np.float64),
        }
        fetch_time += time.perf_counter() - t0

    # Sweep
    results = []
    t1 = time.perf_counter()
    for threshold in params:
        results.append(
            run_one_query(con, cfg, li_sums, orders, key_sums_raw, threshold)
        )
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
def _rows_match(a: Q18Row, b: Q18Row, tol: float = 1.0) -> bool:
    return (
        a[0] == b[0]
        and a[1] == b[1]
        and a[2] == b[2]
        and a[3] == b[3]
        and abs(a[4] - b[4]) <= tol
        and abs(a[5] - b[5]) <= tol
    )


def validate(
    reference: list[Q18Result],
    candidate: list[Q18Result],
    params: list[int],
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
        for p, ref, cand, kind in mismatches[:3]:
            logger.warning(
                f"    threshold={p} ({kind}): ref_rows={len(ref)} got_rows={len(cand)}"
            )
            for rr, rc in zip(ref[:2], cand[:2]):
                logger.warning(f"      ref: {rr}")
                logger.warning(f"      got: {rc}")
        return False
    logger.info(f"  [{label}]  all results match reference")
    return True


# Plug Q18 into the shared benchmark harness.
class Q18Benchmark(QueryBenchmark):
    NAME = "Q18"
    ALL_STEPS = ("T", "J", "G")
    N_APPLICABLE = True
    N_HELP = f"Number of threshold values to sweep in [{THRESHOLD_LO},{THRESHOLD_HI}] (default: 40)"
    N_DEFAULT = 40

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_precompute_sums=args.opt_precompute_sums,
            opt_presort_orders=args.opt_presort_orders,
        )

    def generate_params(self, n: int) -> list[int]:
        return generate_thresholds(n)

    def single_params(self) -> list[int]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query (threshold={SINGLE_THRESHOLD})"

    def is_valid_combo(self, cfg) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg, params) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"precompute_sums={'ON' if args.opt_precompute_sums else 'OFF'}  "
            f"presort_orders={'ON' if args.opt_presort_orders else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_precompute_sums",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Precompute per-orderkey qty sums at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_presort_orders",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Presort orders by (totalprice DESC, orderdate) at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        return f"sweep  N={len(params)} thresholds in [{THRESHOLD_LO}, {THRESHOLD_HI}]"


# Run directly: python3 -m benchmarks.q18
def main() -> None:
    bench = Q18Benchmark()
    parser = make_base_parser(
        "Q18 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q18_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
