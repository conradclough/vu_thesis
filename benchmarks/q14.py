"""
Q14 - promotional revenue fraction, sweep over 1-month shipdate windows.

    SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                             THEN l_extendedprice * (1 - l_discount)
                             ELSE 0 END)
           / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
    FROM lineitem, part
    WHERE l_partkey = p_partkey
      AND l_shipdate >= DATE '[DATE]'
      AND l_shipdate < DATE '[DATE]' + INTERVAL '1' MONTH

The lineitem*part join is always in SQL. What varies is who handles the shipdate
filter, the promo split, and the ratio aggregation. Fetching the joined table once
and sweeping N month windows as cheap array slices is where Python wins - DuckDB
re-executes the join for every single month.

Steps:
  [S] shipdate : l_shipdate in [month_start, month_end)
  [P] promo    : p_type LIKE 'PROMO%' - splits rows into promo vs non-promo
  [A] agg      : 100 * SUM(promo_ep_disc) / SUM(all_ep_disc)

Important: P is never a WHERE filter. Filtering promo rows would destroy the
denominator (total revenue). So "P in SQL" means SQL emits two columns per row -
ep_disc_promo (zeroed for non-promo) and ep_disc_total - and Python (or SQL) sums them.

Validity: A in SQL requires P in SQL. This invalidates {A} and {S,A}, leaving 6 valid combos.

Opt flags (all ON by default):
  --opt_presort      sort by shipdate at fetch time means window is searchsorted slice
  --opt_promo_split  precompute separate promo_ep and total_ep arrays (P-in-Python path)

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

# S/P/A - the three steps and all valid subsets.

ALL_STEPS = ("S", "P", "A")  # shipdate filter, promo split, aggregation

# Generate N 1-month shipdate windows spanning the lineitem date range.

# TPC-H lineitem shipdates span 1992-01-02..1998-12-01.
# We sweep 1-month windows inside that range.
_SWEEP_START = np.datetime64("1993-01-01", "D")
_SWEEP_END = np.datetime64("1998-01-01", "D")  # leave 1-month tail

# Fixed single-query window (TPC-H Q14 spec: September 1995)
_SINGLE_START = np.datetime64("1995-09-01", "D")
_SINGLE_END = np.datetime64("1995-10-01", "D")


def _month_end(start: np.datetime64) -> np.datetime64:
    """Return the first day of the next calendar month after *start*."""
    d = start.astype("datetime64[D]").astype(object)  # -> datetime.date
    month = d.month + 1
    year = d.year + (month > 12)
    month = month if month <= 12 else month - 12
    return np.datetime64(datetime.date(year, month, 1), "D")


def generate_month_params(n: int) -> list[tuple[np.datetime64, np.datetime64]]:
    """
    Produce *n* evenly-spaced 1-month windows across the sweep range.
    Each window is (month_start, month_end) as numpy datetime64[D].
    """
    total_days = int((_SWEEP_END - _SWEEP_START) / np.timedelta64(1, "D"))
    offsets = np.linspace(0, total_days, n, endpoint=False).astype(int)
    params = []
    for off in offsets:
        # Snap to the 1st of whichever month this day falls in
        d = (
            (_SWEEP_START + np.timedelta64(int(off), "D"))
            .astype("datetime64[M]")
            .astype("datetime64[D]")
        )
        params.append((d, _month_end(d)))
    # Deduplicate while preserving order (multiple offsets may hit the same month)
    seen: set = set()
    unique = []
    for p in params:
        key = (str(p[0]), str(p[1]))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def single_query_params() -> list[tuple[np.datetime64, np.datetime64]]:
    return [(_SINGLE_START, _SINGLE_END)]


# Which steps go to SQL vs Python, and which numpy opts are active.
@dataclass
class BenchConfig:
    """Which steps go to SQL vs Python, and which NumPy tricks are active."""

    sql_steps: frozenset  # subset of {"S", "P", "A"}
    opt_presort: bool = True  # sort arrays by shipdate -> searchsorted
    opt_promo_split: bool = True  # split into promo/total arrays at fetch time

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


# A in SQL needs P in SQL to compute the promo/total split - A-without-P is skipped.
def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    """
    The only structural constraint: A in SQL requires P also in SQL.

    SQL can only compute the ratio SUM(promo)/SUM(total) if it performs the
    CASE split (P) itself.  If P is in Python, SQL sees a single undivided
    ep_disc column and cannot produce two separate aggregates in one pass.
    """
    if cfg.sql_handles("A") and cfg.python_handles("P"):
        return False, (
            "A in SQL requires P also in SQL: the ratio needs both promo_sum "
            "and total_sum, which requires the CASE split (P) to happen in SQL."
        )
    return True, ""


# Build the SELECT/WHERE/JOIN dynamically - what SQL sees changes per combo.
def _date_literal(d: np.datetime64) -> str:
    return f"DATE '{str(d)}'"


# Bulk-fetch SQL when S is in Python - no date filter embedded, fetch the full join.
def build_fetch_sql_p_in_python(cfg: BenchConfig) -> str:
    """
    Bulk fetch: S in Python, P in Python.
    Returns (ep_disc, l_shipdate, is_promo) for every row - no filters.
    """
    return (
        "SELECT\n"
        "  l_extendedprice * (1.0 - l_discount) AS ep_disc,\n"
        "  l_shipdate,\n"
        "  (p_type LIKE 'PROMO%') AS is_promo\n"
        "FROM lineitem\n"
        "JOIN part ON l_partkey = p_partkey"
    )


def build_fetch_sql_p_in_sql(cfg: BenchConfig) -> str:
    """
    Bulk fetch: S in Python, P in SQL.
    SQL pre-splits ep_disc into two columns (promo / total) at row level.
    Python then sweeps the shipdate window and sums each column - no CASE needed
    per query, no boolean masking, just two contiguous-memory reductions.

      ep_disc_promo[i] = ep_disc[i] if is_promo[i] else 0.0
      ep_disc_total[i] = ep_disc[i]  (always)
    """
    return (
        "SELECT\n"
        "  CASE WHEN p_type LIKE 'PROMO%'\n"
        "       THEN l_extendedprice * (1.0 - l_discount) ELSE 0.0 END AS ep_disc_promo,\n"
        "  l_extendedprice * (1.0 - l_discount)                        AS ep_disc_total,\n"
        "  l_shipdate\n"
        "FROM lineitem\n"
        "JOIN part ON l_partkey = p_partkey"
    )


# Per-query SQL when S is in SQL - date window embedded directly in WHERE.
def build_per_query_sql_p_in_python(
    month_start: np.datetime64,
    month_end: np.datetime64,
) -> str:
    """S in SQL, P in Python, A in Python: window-filtered rows + is_promo."""
    return (
        f"SELECT\n"
        f"  l_extendedprice * (1.0 - l_discount) AS ep_disc,\n"
        f"  (p_type LIKE 'PROMO%')               AS is_promo\n"
        f"FROM lineitem\n"
        f"JOIN part ON l_partkey = p_partkey\n"
        f"WHERE l_shipdate >= {_date_literal(month_start)}\n"
        f"  AND l_shipdate <  {_date_literal(month_end)}"
    )


def build_per_query_sql_p_in_sql_a_in_python(
    month_start: np.datetime64,
    month_end: np.datetime64,
) -> str:
    """S in SQL, P in SQL, A in Python: SQL returns (promo_sum, total_sum)."""
    return (
        f"SELECT\n"
        f"  SUM(CASE WHEN p_type LIKE 'PROMO%'\n"
        f"           THEN l_extendedprice * (1.0 - l_discount) ELSE 0.0 END) AS promo_sum,\n"
        f"  SUM(l_extendedprice * (1.0 - l_discount))                        AS total_sum\n"
        f"FROM lineitem\n"
        f"JOIN part ON l_partkey = p_partkey\n"
        f"WHERE l_shipdate >= {_date_literal(month_start)}\n"
        f"  AND l_shipdate <  {_date_literal(month_end)}"
    )


def build_per_query_sql_pure(
    month_start: np.datetime64,
    month_end: np.datetime64,
) -> str:
    """S in SQL, P in SQL, A in SQL: single promo_revenue float back."""
    return (
        f"SELECT\n"
        f"  100.0 * SUM(CASE WHEN p_type LIKE 'PROMO%'\n"
        f"                   THEN l_extendedprice * (1.0 - l_discount) ELSE 0.0 END)\n"
        f"  / NULLIF(SUM(l_extendedprice * (1.0 - l_discount)), 0) AS promo_revenue\n"
        f"FROM lineitem\n"
        f"JOIN part ON l_partkey = p_partkey\n"
        f"WHERE l_shipdate >= {_date_literal(month_start)}\n"
        f"  AND l_shipdate <  {_date_literal(month_end)}"
    )


# One-time bulk fetch when S is in Python - single round-trip amortised across all N queries.
@dataclass
class FetchedArrays:
    shipdate: np.ndarray  # int64 days-since-epoch, shape (N,)
    # P-in-Python layout
    ep_disc: np.ndarray | None  # ep*(1-disc), shape (N,)
    is_promo: np.ndarray | None  # bool, shape (N,)
    # P-in-SQL layout, or opt_promo_split post-processing of P-in-Python
    ep_disc_promo: np.ndarray | None  # ep*(1-disc) if promo else 0
    ep_disc_total: np.ndarray | None  # ep*(1-disc) always
    presorted: bool = False


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    # S in SQL means per-query SQL calls - no bulk fetch.
    if cfg.sql_handles("S"):
        return None, 0.0

    # P in SQL: SQL pre-splits ep_disc into ep_disc_promo (zeroed for non-promo)
    # and ep_disc_total. Python just sums two contiguous arrays per window.
    # P in Python: SQL returns (ep_disc, is_promo) per row; Python does the split.
    if cfg.python_handles("P"):
        sql = build_fetch_sql_p_in_python(cfg)
    else:
        sql = build_fetch_sql_p_in_sql(cfg)

    # One SQL round-trip for the full lineitem*part join, amortised across N months.
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    fetch_time = time.perf_counter() - t0

    shipdate_raw = raw["l_shipdate"]
    if shipdate_raw.dtype.kind == "M":
        shipdate = shipdate_raw.astype("datetime64[D]").view(np.int64)
    else:
        # DuckDB may return int32 days-since-epoch; widen to int64.
        shipdate = shipdate_raw.astype(np.int64)

    if cfg.python_handles("P"):
        ep_disc = raw["ep_disc"].astype(np.float64)
        is_promo = raw["is_promo"].astype(bool)
        ep_disc_promo_col = None
        ep_disc_total_col = None
    else:
        # SQL already emitted the two-column split - no CASE needed per query.
        ep_disc = None
        is_promo = None
        ep_disc_promo_col = raw["ep_disc_promo"].astype(np.float64)
        ep_disc_total_col = raw["ep_disc_total"].astype(np.float64)

    # Sort by shipdate once - per-query window is two searchsorted calls.
    # DuckDB applies the WHERE l_shipdate filter on every month; we pay it once.
    if cfg.opt_presort:
        sort_idx = np.argsort(shipdate, kind="stable")
        shipdate = shipdate[sort_idx]
        if ep_disc is not None:
            ep_disc = ep_disc[sort_idx]
        if is_promo is not None:
            is_promo = is_promo[sort_idx]
        if ep_disc_promo_col is not None:
            ep_disc_promo_col = ep_disc_promo_col[sort_idx]
        if ep_disc_total_col is not None:
            ep_disc_total_col = ep_disc_total_col[sort_idx]

    # Hoist the is_promo CASE out of the per-query loop: split ep_disc into
    # ep_disc_promo (zeroed for non-promo rows) and ep_disc_total once.
    # Each per-query window reduces to two np.sum calls on contiguous arrays.
    if cfg.opt_promo_split and cfg.python_handles("P") and is_promo is not None:
        ep_disc_promo_col = np.where(is_promo, ep_disc, 0.0)
        ep_disc_total_col = ep_disc  # total IS ep_disc - no copy needed

    return FetchedArrays(
        shipdate=shipdate,
        ep_disc=ep_disc,
        is_promo=is_promo,
        ep_disc_promo=ep_disc_promo_col,
        ep_disc_total=ep_disc_total_col,
        presorted=cfg.opt_presort,
    ), fetch_time


# N tight loops - no SQL calls from here. The bulk fetch bought us that.
def _shipdate_to_int64(d: np.datetime64) -> int:
    """Convert a datetime64[D] scalar to int64 days-since-epoch."""
    return int(d.astype("datetime64[D]").view(np.int64))


def _ratio(promo_sum: float, total_sum: float) -> float:
    if total_sum == 0.0:
        return 0.0
    return round(100.0 * promo_sum / total_sum, 6)


def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    month_start: np.datetime64,
    month_end: np.datetime64,
) -> float:
    """Execute one sweep iteration. Returns promo_revenue as float."""

    # S in SQL: issue a fresh SQL call per window.
    if cfg.sql_handles("S"):
        if cfg.sql_handles("P") and cfg.sql_handles("A"):
            # Pure SQL: single float back
            sql = build_per_query_sql_pure(month_start, month_end)
            row = con.execute(sql).fetchone()
            return round(float(row[0]) if (row and row[0] is not None) else 0.0, 6)

        if cfg.sql_handles("P") and cfg.python_handles("A"):
            # SQL computes (promo_sum, total_sum); Python divides
            sql = build_per_query_sql_p_in_sql_a_in_python(month_start, month_end)
            row = con.execute(sql).fetchone()
            if row is None or row[1] is None or float(row[1]) == 0.0:
                return 0.0
            return _ratio(float(row[0] or 0.0), float(row[1]))

        # S in SQL, P in Python, A in Python
        sql = build_per_query_sql_p_in_python(month_start, month_end)
        raw = con.execute(sql).fetchnumpy()
        ep_disc = raw["ep_disc"].astype(np.float64)
        is_promo = raw["is_promo"].astype(bool)
        return _ratio(ep_disc[is_promo].sum(), ep_disc.sum())

    # S in Python: arrays already bulk-fetched - apply window as array slice.
    assert arrays is not None

    start_i64 = _shipdate_to_int64(month_start)
    end_i64 = _shipdate_to_int64(month_end)

    if arrays.presorted:
        # O(log n) window extraction via binary search
        lo = np.searchsorted(arrays.shipdate, start_i64, side="left")
        hi = np.searchsorted(arrays.shipdate, end_i64, side="left")

        if arrays.ep_disc_promo is not None:
            # Best path: two pre-split contiguous arrays, pure SIMD reduction.
            # Applies to: P-in-SQL (SQL pre-split) or P-in-Python + opt_promo_split.
            promo_sum = arrays.ep_disc_promo[lo:hi].sum()
            total_sum = arrays.ep_disc_total[lo:hi].sum()
        else:
            # opt_promo_split OFF, P in Python: boolean-index within the window slice
            ep_sl = arrays.ep_disc[lo:hi]
            pr_sl = arrays.is_promo[lo:hi]
            promo_sum = ep_sl[pr_sl].sum()
            total_sum = ep_sl.sum()

    else:
        # No presort, full boolean mask over all rows
        sd = arrays.shipdate
        window_mask = (sd >= start_i64) & (sd < end_i64)

        if arrays.ep_disc_promo is not None:
            promo_sum = arrays.ep_disc_promo[window_mask].sum()
            total_sum = arrays.ep_disc_total[window_mask].sum()
        else:
            ep_w = arrays.ep_disc[window_mask]
            pr_w = arrays.is_promo[window_mask]
            promo_sum = ep_w[pr_w].sum()
            total_sum = ep_w.sum()

    return _ratio(float(promo_sum), float(total_sum))


# Orchestrate fetch + N-query loop for one combo; return a timed SweepResult.
def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[tuple[np.datetime64, np.datetime64]],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key or '(none)'}: {reason}")

    arrays, fetch_time = fetch_and_prepare(con, cfg)

    results = []
    t1 = time.perf_counter()
    for start, end in params:
        results.append(run_one_query(con, cfg, arrays, start, end))
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
    reference: list[float],
    candidate: list[float],
    params: list[tuple[np.datetime64, np.datetime64]],
    label: str,
    tol: float = 0.01,  # 0.01 percentage-point tolerance for floating-point drift
) -> bool:
    mismatches = [
        (params[i], reference[i], candidate[i])
        for i, (r, c) in enumerate(zip(reference, candidate))
        if abs(r - c) > tol
    ]
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, r, c in mismatches[:5]:
            logger.warning(f"    window=[{p[0]}, {p[1]}) ref={r:.6f} got={c:.6f}")
        return False
    logger.info(f"  [{label}] all results match reference")
    return True


# Plug Q14 into the shared benchmark harness.
class Q14Benchmark(QueryBenchmark):
    NAME = "Q14"
    ALL_STEPS = ("S", "P", "A")
    N_APPLICABLE = True
    N_HELP = "Number of 1-month sweep windows (default: 120)"
    N_DEFAULT = 120

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_presort=args.opt_presort,
            opt_promo_split=args.opt_promo_split,
        )

    def generate_params(self, n: int):
        return generate_month_params(n)

    def single_params(self):
        return single_query_params()

    def single_label(self) -> str:
        return f"single query (window=[{_SINGLE_START}, {_SINGLE_END}), N=1)"

    def is_valid_combo(self, cfg) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg, params) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"presort={'ON' if args.opt_presort else 'OFF'}  "
            f"promo_split={'ON' if args.opt_promo_split else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort arrays by shipdate to enable searchsorted (default: ON)",
        )
        parser.add_argument(
            "--opt_promo_split",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Split ep*(1-disc) into promo/total arrays at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        return f"sweep  N={len(params)} unique monthly windows (requested {args.n})"


# Run directly: python3 -m benchmarks.q14
def main() -> None:
    bench = Q14Benchmark()
    parser = make_base_parser(
        "Q14 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q14_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
