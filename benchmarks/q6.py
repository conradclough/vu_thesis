"""
Q6 - discount-window revenue sweep.

    SELECT SUM(l_extendedprice * l_discount) AS revenue
    FROM lineitem
    WHERE l_shipdate >= DATE '1994-01-01'
      AND l_shipdate < DATE '1995-01-01'
      AND l_discount BETWEEN [LO] AND [HI]
      AND l_quantity < 24

The sweep parameter is a discount range [lo, hi]; we test N different windows.
DuckDB re-executes the full WHERE clause for every window. The Python paths
bulk-fetch once and sweep N windows as cheap array operations.

Steps (each independently assigned to SQL or Python):
  [S] shipdate  : l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'
  [D] discount  : l_discount BETWEEN lo AND hi
  [Q] quantity  : l_quantity < 24
  [A] agg       : SUM(l_extendedprice * l_discount)

Validity: A in SQL returns a scalar - Python can't post-filter a scalar, so A
in SQL forces all filters (S, D, Q) into SQL too.

Opt flags (all ON by default):
  --opt_presort     sort by discount → per-query filter is searchsorted not a mask
  --opt_precompute  compute ep*disc at fetch time, not N times in the loop
  --opt_qty_premask zero out ep*disc where qty>=24 once at fetch time
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

# S/D/Q/A - the four steps and all their subset combinations.

ALL_STEPS = ("S", "D", "Q", "A")  # shipdate, discount, quantity, aggregation

# Fixed date bounds and thresholds from the TPC-H Q6 spec.

DATE_LO = "DATE '1994-01-01'"
DATE_HI = "DATE '1995-01-01'"
QTY_THRESH = 24
_D1 = np.datetime64("1994-01-01")
_D2 = np.datetime64("1995-01-01")

Q6_DISCOUNT_LO: float = 0.05
Q6_DISCOUNT_HI: float = 0.07


def generate_discount_params(n: int) -> list[tuple[float, float]]:
    lo_values = np.linspace(0.03, 0.07, n)
    return [(round(float(lo), 4), round(float(lo + 0.02), 4)) for lo in lo_values]


def single_query_params() -> list[tuple[float, float]]:
    return [(Q6_DISCOUNT_LO, Q6_DISCOUNT_HI)]


# Which steps go to SQL vs Python, and which numpy opts are active.
@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_presort: bool = True
    opt_precompute: bool = True
    opt_qty_premask: bool = True

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


# A in SQL returns a scalar - Python can't post-filter it, so A forces S+D+Q into SQL too.
def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("A") and cfg.sql_steps != frozenset(ALL_STEPS):
        python_filters = sorted(cfg.python_steps - {"A"})
        missing = ", ".join(python_filters) if python_filters else "(none)"
        return False, (
            f"A in SQL requires all predicates in SQL too (got {missing} in Python). "
            "SQL returns a scalar SUM - Python cannot post-filter an aggregate."
        )
    return True, ""


# Build the WHERE clause dynamically - what SQL sees changes per combo.
def build_fetch_sql(cfg: BenchConfig) -> str:
    assert cfg.python_handles("D"), "build_fetch_sql only called when D is in Python"

    where_clauses = []
    if cfg.sql_handles("S"):
        where_clauses += [f"l_shipdate >= {DATE_LO}", f"l_shipdate <  {DATE_HI}"]
    if cfg.sql_handles("Q"):
        where_clauses.append(f"l_quantity < {QTY_THRESH}")

    where_sql = ("\nWHERE " + "\n  AND ".join(where_clauses)) if where_clauses else ""

    needed = ["l_extendedprice", "l_discount"]
    if cfg.python_handles("S"):
        needed.append("l_shipdate")
    if cfg.python_handles("Q"):
        needed.append("l_quantity")

    return f"SELECT {', '.join(needed)}\nFROM lineitem{where_sql}"


def build_per_query_sql(
    cfg: BenchConfig, discount_lo: float, discount_hi: float
) -> str:
    assert cfg.sql_handles("D"), "build_per_query_sql only called when D is in SQL"

    where_clauses = [f"l_discount BETWEEN {discount_lo} AND {discount_hi}"]
    if cfg.sql_handles("S"):
        where_clauses += [f"l_shipdate >= {DATE_LO}", f"l_shipdate <  {DATE_HI}"]
    if cfg.sql_handles("Q"):
        where_clauses.append(f"l_quantity < {QTY_THRESH}")

    where_sql = "\nWHERE " + "\n  AND ".join(where_clauses)

    if cfg.sql_handles("A"):
        select = "SELECT SUM(l_extendedprice * l_discount) AS revenue"
    else:
        needed = ["l_extendedprice", "l_discount"]
        if cfg.python_handles("S"):
            needed.append("l_shipdate")
        if cfg.python_handles("Q"):
            needed.append("l_quantity")
        select = f"SELECT {', '.join(needed)}"

    return f"{select}\nFROM lineitem{where_sql}"


# One-time bulk fetch - single SQL round-trip whose cost is amortised across all N queries.
@dataclass
class FetchedArrays:
    discount: np.ndarray
    ep: np.ndarray  # raw l_extendedprice,  needed when ep_x_disc is None
    ep_x_disc: np.ndarray | None = None  # ep*disc when precompute=on
    shipdate: np.ndarray | None = None
    quantity: np.ndarray | None = None
    ep_x_disc_qty_masked: np.ndarray | None = None
    presorted: bool = False


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    if cfg.sql_handles("D"):
        return None, 0.0

    sql = build_fetch_sql(cfg)
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    fetch_time = time.perf_counter() - t0

    ep = raw["l_extendedprice"].astype(np.float64)
    disc = raw["l_discount"].astype(np.float64)
    shipdate = raw.get("l_shipdate")
    quantity = raw.get("l_quantity")

    # Sort by discount once so per-query D filter is searchsorted not a full mask scan.
    # DuckDB applies discount BETWEEN on every query; we pay this sort once.
    if cfg.opt_presort:
        sort_idx = np.argsort(disc, kind="stable")
        disc = disc[sort_idx]
        ep = ep[sort_idx]
        if shipdate is not None:
            shipdate = shipdate[sort_idx]
        if quantity is not None:
            quantity = quantity[sort_idx]

    # opt_precompute: materialise ep*disc once at fetch time so each query is a
    # pure reduction. When off, the per-query path multiplies on the window slice
    ep_x_disc = ep * disc if cfg.opt_precompute else None

    # opt_qty_premask: when Q is in Python, zero out non-qualifying rows once at
    # fetch time so the query loop just sums. Needs a product to mask, if
    # precompute=off we still build (ep*disc) here so the two flags can be tested
    # independently. The premask path then bypasses the per-query qty filter.
    ep_x_disc_qty_masked: np.ndarray | None = None
    if cfg.opt_qty_premask and cfg.python_handles("Q") and quantity is not None:
        product = ep_x_disc if ep_x_disc is not None else ep * disc
        ep_x_disc_qty_masked = product * (quantity < QTY_THRESH)

    return FetchedArrays(
        discount=disc,
        ep=ep,
        ep_x_disc=ep_x_disc,
        shipdate=shipdate,
        quantity=quantity,
        ep_x_disc_qty_masked=ep_x_disc_qty_masked,
        presorted=cfg.opt_presort,
    ), fetch_time


# N tight loops - no SQL calls from here. The bulk fetch bought us that.
def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    discount_lo: float,
    discount_hi: float,
) -> float:
    if cfg.sql_handles("D"):
        sql = build_per_query_sql(cfg, discount_lo, discount_hi)

        if cfg.sql_handles("A"):
            row = con.execute(sql).fetchone()
            return round(float(row[0]), 4) if row and row[0] is not None else 0.0

        raw = con.execute(sql).fetchnumpy()
        ep = raw["l_extendedprice"].astype(np.float64)
        disc = raw["l_discount"].astype(np.float64)
        ep_x_disc = ep * disc

        mask = np.ones(len(ep), dtype=bool)
        if cfg.python_handles("S"):
            sd = raw["l_shipdate"]
            mask &= (sd >= _D1) & (sd < _D2)
        if cfg.python_handles("Q"):
            mask &= raw["l_quantity"] < QTY_THRESH

        return round(float(ep_x_disc[mask].sum()), 4)

    assert arrays is not None

    # Pick the product source: pre-masked > precomputed > compute-on-slice.
    if cfg.python_handles("Q") and arrays.ep_x_disc_qty_masked is not None:
        working = arrays.ep_x_disc_qty_masked
        qty_already_masked = True
    elif arrays.ep_x_disc is not None:
        working = arrays.ep_x_disc
        qty_already_masked = False
    else:
        # precompute=off, no qty_premask: multiply on the window slice below.
        working = None
        qty_already_masked = False

    if arrays.presorted:
        lo_idx = np.searchsorted(arrays.discount, discount_lo, side="left")
        hi_idx = np.searchsorted(arrays.discount, discount_hi, side="right")
        if working is not None:
            ep_slice = working[lo_idx:hi_idx]
        else:
            ep_slice = arrays.ep[lo_idx:hi_idx] * arrays.discount[lo_idx:hi_idx]
        qty_slice = (
            arrays.quantity[lo_idx:hi_idx] if arrays.quantity is not None else None
        )
        sd_slice = (
            arrays.shipdate[lo_idx:hi_idx] if arrays.shipdate is not None else None
        )

        sub_mask = np.ones(hi_idx - lo_idx, dtype=bool)
        if cfg.python_handles("Q") and not qty_already_masked and qty_slice is not None:
            sub_mask &= qty_slice < QTY_THRESH
        if cfg.python_handles("S") and sd_slice is not None:
            sub_mask &= (sd_slice >= _D1) & (sd_slice < _D2)

        if not sub_mask.all():
            ep_slice = ep_slice[sub_mask]

        return round(float(ep_slice.sum()), 4)

    disc = arrays.discount
    mask = (disc >= discount_lo) & (disc <= discount_hi)
    if (
        cfg.python_handles("Q")
        and not qty_already_masked
        and arrays.quantity is not None
    ):
        mask &= arrays.quantity < QTY_THRESH
    if cfg.python_handles("S") and arrays.shipdate is not None:
        mask &= (arrays.shipdate >= _D1) & (arrays.shipdate < _D2)

    if working is not None:
        return round(float(working[mask].sum()), 4)
    return round(float((arrays.ep[mask] * arrays.discount[mask]).sum()), 4)


# Orchestrate fetch + N-query loop for one combo; return a timed SweepResult.
def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[tuple[float, float]],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key or '(none)'}: {reason}")

    arrays, fetch_time = fetch_and_prepare(con, cfg)

    results = []
    t1 = time.perf_counter()
    for lo, hi in params:
        results.append(run_one_query(con, cfg, arrays, lo, hi))
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
    params: list[tuple[float, float]],
    label: str,
    tol: float = 0.05,
) -> bool:
    mismatches = [
        (params[i], reference[i], candidate[i])
        for i, (r, c) in enumerate(zip(reference, candidate))
        if abs(r - c) > tol
    ]
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, r, c in mismatches[:5]:
            logger.warning(f"    discount=[{p[0]}, {p[1]}] ref={r} got={c}")
        return False
    logger.info(f"  [{label}] all results match reference")
    return True


# Plug Q6 into the shared benchmark harness.
class Q6Benchmark(QueryBenchmark):
    NAME = "Q6"
    ALL_STEPS = ("S", "D", "Q", "A")
    N_APPLICABLE = True
    N_HELP = "Number of discount-window sweep params (default: 200)"
    N_DEFAULT = 200

    def make_config(
        self, sql_steps: frozenset, args: argparse.Namespace
    ) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_presort=args.opt_presort,
            opt_precompute=args.opt_precompute,
            opt_qty_premask=args.opt_qty_premask,
        )

    def generate_params(self, n: int) -> list[tuple[float, float]]:
        return generate_discount_params(n)

    def single_params(self) -> list[tuple[float, float]]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query (discount=[{Q6_DISCOUNT_LO}, {Q6_DISCOUNT_HI}], N=1)"

    def is_valid_combo(self, cfg: BenchConfig) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg: BenchConfig, params) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args: argparse.Namespace) -> str:
        return (
            f"presort={'ON' if args.opt_presort else 'OFF'}  "
            f"precompute={'ON' if args.opt_precompute else 'OFF'}  "
            f"qty_premask={'ON' if args.opt_qty_premask else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort arrays by discount to enable searchsorted (default: ON)",
        )
        parser.add_argument(
            "--opt_precompute",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Precompute ep*discount at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_qty_premask",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Zero out ep*disc where qty>=24 at fetch time (default: ON)",
        )

    def log_mode(self, args: argparse.Namespace, params: list) -> str:
        return f"sweep  N={len(params)} parameter sets"


# Run directly: python3 -m benchmarks.q6
def main() -> None:
    bench = Q6Benchmark()
    parser = make_base_parser(
        "Q6 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q6_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)

    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
