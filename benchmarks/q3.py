"""
Q3 - top-10 unshipped orders by revenue, sweep over cutoff date * customer segment.

    SELECT l_orderkey,
           SUM(l_extendedprice * (1 - l_discount)) AS revenue,
           o_orderdate, o_shippriority
    FROM customer, orders, lineitem
    WHERE c_mktsegment = '[SEGMENT]'
      AND c_custkey = o_custkey
      AND l_orderkey = o_orderkey
      AND o_orderdate < DATE '[DATE]'
      AND l_shipdate > DATE '[DATE]'
    GROUP BY l_orderkey, o_orderdate, o_shippriority
    ORDER BY revenue DESC, o_orderdate
    LIMIT 10

The cutoff date splits orders (before) from lineitems (after); we test N dates
across [1995-03-01, 1995-03-31] * 5 segments = up to 5N params total.
DuckDB re-executes the 3-table join and GROUP BY for every (segment, date) pair.
Python bulk-fetches the join once and sweeps (segment, date) pairs as array slices.

Steps:
  [C] customer  : c_mktsegment = :segment
  [D] date      : o_orderdate < date AND l_shipdate > date (same cutoff, two tables)
  [G] group+agg : GROUP BY l_orderkey + SUM(ep*(1-disc)) + TOP-10 sort

Validity: G collapses rows - Python can't post-aggregate a GROUP BY result,
so G in SQL forces C and D into SQL too.

Opt flags (all ON by default):
  --opt_presort      sort by (seg_key, orderkey) at fetch time; per-query date
                     filter is a boolean mask within a contiguous segment slice
  --opt_precompute   compute ep*(1-disc) at fetch time - DuckDB recomputes it per query
  --opt_encode_seg   encode c_mktsegment to uint8 at fetch time - comparison is == not string eq

Usage:
  python3 -m benchmarks.benchmark_sweep q3 --sf 1 --n 20 --repeats 5
  python3 -m benchmarks.benchmark_sweep q3 --n 10 --no_opt_presort --no_opt_encode_seg
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import logging
import os
import time
from collections import defaultdict
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

# C/D/G - the three steps and all valid subsets.

ALL_STEPS = ("C", "D", "G")

# Fixed date range for the sweep from the TPC-H Q3 spec.

ALL_SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
_SWEEP_DATE_LO = np.datetime64("1995-03-01", "D")
_SWEEP_DATE_HI = np.datetime64("1995-03-31", "D")
SINGLE_SEGMENT = "BUILDING"
SINGLE_DATE = np.datetime64("1995-03-15", "D")
_SEG_TO_U8: dict[str, int] = {s: i for i, s in enumerate(ALL_SEGMENTS)}
LIMIT = 10

# One result row per qualifying order (top-10 by revenue).

Q3Row = tuple[int, float, str, int]
Q3Result = list[Q3Row]
SweepParam = tuple[str, np.datetime64]


# Generate the (segment, date) parameter grid.
def generate_sweep_params(n_dates: int) -> list[SweepParam]:
    span = int((_SWEEP_DATE_HI - _SWEEP_DATE_LO) / np.timedelta64(1, "D"))
    offsets = np.linspace(0, span, n_dates, dtype=int)
    dates = [_SWEEP_DATE_LO + np.timedelta64(int(o), "D") for o in offsets]
    return [(seg, d) for seg in ALL_SEGMENTS for d in dates]


def single_query_params() -> list[SweepParam]:
    return [(SINGLE_SEGMENT, SINGLE_DATE)]


def _date_str(d: np.datetime64) -> str:
    return str(d.astype("datetime64[D]"))


def _date_i64(d: np.datetime64) -> int:
    return int(d.astype("datetime64[D]").view(np.int64))


# Which steps go to SQL vs Python, and which numpy opts are active.
@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_precompute: bool = True
    opt_encode_seg: bool = True
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
        return step not in self.sql_steps


# G in SQL collapses rows - Python can't post-filter, so G forces C+D into SQL too.
def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("G") and cfg.python_steps - {"G"}:
        missing = sorted(cfg.python_steps - {"G"})
        return False, (
            f"G in SQL requires C and D also in SQL (missing: {missing}). "
            "SQL GROUP BY collapses rows; Python cannot post-filter."
        )
    return True, ""


# Build the SELECT/WHERE/JOIN dynamically - what SQL sees changes per combo.
_BASE_JOIN = (
    "FROM customer\n"
    "JOIN orders   ON c_custkey  = o_custkey\n"
    "JOIN lineitem ON l_orderkey = o_orderkey"
)


def _date_where(d: np.datetime64) -> str:
    ds = _date_str(d)
    return f"o_orderdate < DATE '{ds}' AND l_shipdate > DATE '{ds}'"


def _seg_where(seg: str) -> str:
    return f"c_mktsegment = '{seg}'"


def build_fetch_sql_full(cfg: BenchConfig) -> str:
    """Full bulk fetch (D in Python, C may be in SQL or Python)."""
    cols = [
        "l_orderkey",
        "l_extendedprice * (1.0 - l_discount) AS ep_disc",
        "o_orderdate",
        "l_shipdate",
        "o_shippriority",
        "r_name",  # always include region for segment filtering
    ]
    if cfg.python_handles("C"):
        cols.append(
            "CASE c_mktsegment "
            + " ".join(f"WHEN '{s}' THEN {i}" for s, i in _SEG_TO_U8.items())
            + " ELSE 255 END AS seg_key"
            if cfg.opt_encode_seg
            else "c_mktsegment"
        )
    # r_name isn't actually in Q3's join - remove it, use c_mktsegment directly
    cols = [c for c in cols if "r_name" not in c]
    return f"SELECT {', '.join(cols)}\n{_BASE_JOIN}"


def build_fetch_sql_segment(cfg: BenchConfig, seg: str) -> str:
    """Per-segment bulk fetch: C in SQL, D in Python."""
    assert cfg.sql_handles("C") and cfg.python_handles("D")
    cols = [
        "l_orderkey",
        "l_extendedprice * (1.0 - l_discount) AS ep_disc",
        "o_orderdate",
        "l_shipdate",
        "o_shippriority",
    ]
    return f"SELECT {', '.join(cols)}\n{_BASE_JOIN}\nWHERE {_seg_where(seg)}"


def build_per_query_sql(cfg: BenchConfig, seg: str, d: np.datetime64) -> str:
    """Per-query SQL: C and D both in SQL."""
    assert cfg.sql_handles("C") and cfg.sql_handles("D")
    where = f"{_seg_where(seg)}\n  AND {_date_where(d)}"
    if cfg.sql_handles("G"):
        return (
            "SELECT l_orderkey,\n"
            "  SUM(l_extendedprice * (1.0 - l_discount)) AS revenue,\n"
            "  o_orderdate, o_shippriority\n"
            f"{_BASE_JOIN}\n"
            f"WHERE {where}\n"
            "GROUP BY l_orderkey, o_orderdate, o_shippriority\n"
            "ORDER BY revenue DESC, o_orderdate\n"
            f"LIMIT {LIMIT}"
        )
    return (
        "SELECT l_orderkey,\n"
        "  l_extendedprice * (1.0 - l_discount) AS ep_disc,\n"
        "  o_orderdate, o_shippriority\n"
        f"{_BASE_JOIN}\n"
        f"WHERE {where}"
    )


def build_per_query_sql_d_only(cfg: BenchConfig, d: np.datetime64) -> str:
    """D in SQL, C in Python: embed date, return all segments."""
    assert cfg.sql_handles("D") and cfg.python_handles("C")
    return (
        "SELECT l_orderkey,\n"
        "  l_extendedprice * (1.0 - l_discount) AS ep_disc,\n"
        "  o_orderdate, l_shipdate,\n"
        "  o_shippriority, c_mktsegment\n"
        f"{_BASE_JOIN}\n"
        f"WHERE {_date_where(d)}"
    )


# Arrays holding the bulk-fetched join result, with optional precomputed fields.
@dataclass
class FetchedArrays:
    orderkey: np.ndarray
    ep_disc: np.ndarray
    orderdate: np.ndarray  # int64
    shipdate: np.ndarray  # int64
    shippriority: np.ndarray
    seg_key: np.ndarray | None  # uint8 (C in Python + opt_encode_seg)
    seg_raw: np.ndarray | None  # object strings
    presorted: bool = False
    seg_group_info: dict | None = None


def _to_i64(arr: np.ndarray) -> np.ndarray:
    return (
        arr.astype("datetime64[D]").view(np.int64)
        if arr.dtype.kind == "M"
        else arr.astype(np.int64)
    )


def _fmt_date(d_i64: int) -> str:
    return str(np.datetime64(int(d_i64), "D"))


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    segment: str | None = None,
) -> tuple[FetchedArrays | None, float]:
    # C+D in SQL means per-query SQL calls - no bulk fetch.
    if cfg.sql_handles("C") and cfg.sql_handles("D"):
        return None, 0.0

    # C in SQL, D in Python: fetch one segment's rows (smaller, no date filter yet).
    # C in Python: fetch the full 3-table join across all segments at once.
    if cfg.sql_handles("C"):
        assert segment is not None
        sql = build_fetch_sql_segment(cfg, segment)
    else:
        sql = build_fetch_sql_full(cfg)

    # One SQL round-trip, amortised across all N (segment, date) queries.
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    ft = time.perf_counter() - t0

    orderkey = raw["l_orderkey"].astype(np.int64)
    # SQL precomputed ep*(1-disc) - DuckDB recomputes this expression on every query.
    ep_disc = raw["ep_disc"].astype(np.float64)
    orderdate = _to_i64(raw["o_orderdate"])
    shipdate = _to_i64(raw["l_shipdate"])
    shippriority = raw["o_shippriority"].astype(np.int32)

    # SQL already encoded segment as uint8 - per-query filter is == not string eq.
    seg_key = seg_raw = None
    if cfg.python_handles("C"):
        if cfg.opt_encode_seg and "seg_key" in raw:
            seg_key = raw["seg_key"].astype(np.uint8)
        elif "c_mktsegment" in raw:
            seg_raw = raw["c_mktsegment"]

    presorted = False
    seg_group_info = None

    if cfg.opt_presort:
        # Sort by (seg_key, orderkey) so each segment's rows are a contiguous block.
        # Per-query date filter is a boolean mask within that block - no global scan.
        # DuckDB re-applies c_mktsegment = :seg on every query without this presorting.
        if seg_key is not None:
            sort_idx = np.lexsort((orderkey, seg_key))
            seg_key = seg_key[sort_idx]
        else:
            sort_idx = np.argsort(orderkey, kind="stable")

        orderkey = orderkey[sort_idx]
        ep_disc = ep_disc[sort_idx]
        orderdate = orderdate[sort_idx]
        shipdate = shipdate[sort_idx]
        shippriority = shippriority[sort_idx]
        if seg_raw is not None:
            seg_raw = seg_raw[sort_idx]
        presorted = True

        if seg_key is not None:
            # Build per-segment slice views once so per-query work is just a boolean
            # mask within the segment's contiguous block.
            unique_segs, s_starts, s_counts = np.unique(
                seg_key, return_index=True, return_counts=True
            )
            seg_group_info = {
                int(sk): {
                    "start": int(start),
                    "count": int(count),
                    "ok_slice": orderkey[start : start + count],
                    "od_slice": orderdate[start : start + count],
                    "sd_slice": shipdate[start : start + count],
                    "sp_slice": shippriority[start : start + count],
                    "ep_slice": ep_disc[start : start + count],
                }
                for sk, start, count in zip(unique_segs, s_starts, s_counts)
            }
        elif cfg.sql_handles("C"):
            # Single-segment fetch (C in SQL) - treat whole array as one group.
            seg_group_info = {
                0: {
                    "start": 0,
                    "count": len(orderkey),
                    "ok_slice": orderkey,
                    "od_slice": orderdate,
                    "sd_slice": shipdate,
                    "sp_slice": shippriority,
                    "ep_slice": ep_disc,
                }
            }
        # else: C in Python + encode_seg=off. seg_raw exists but we have no
        # per-segment grouping. Leave seg_group_info=None so _numpy_query falls
        # through to the seg_raw mask path (correct, just slower).

    return FetchedArrays(
        orderkey=orderkey,
        ep_disc=ep_disc,
        orderdate=orderdate,
        shipdate=shipdate,
        shippriority=shippriority,
        seg_key=seg_key,
        seg_raw=seg_raw,
        presorted=presorted,
        seg_group_info=seg_group_info,
    ), ft


# N tight loops - no SQL calls from here. The bulk fetch bought us that.
def _top10_from_arrays(
    ok: np.ndarray,
    ep: np.ndarray,
    od: np.ndarray,
    sp: np.ndarray,
) -> Q3Result:
    if len(ok) == 0:
        return []

    sort_idx = np.argsort(ok, kind="stable")
    ok_s = ok[sort_idx]
    ep_s = ep[sort_idx]
    od_s = od[sort_idx]
    sp_s = sp[sort_idx]

    unique_ok, starts, _ = np.unique(ok_s, return_index=True, return_counts=True)
    rev = np.add.reduceat(ep_s, starts)
    od_r = od_s[starts]
    sp_r = sp_s[starts]

    n = len(unique_ok)
    if n <= LIMIT:
        top_idx = np.arange(n)
    else:
        top_idx = np.argpartition(-rev, LIMIT)[:LIMIT]
    top_idx = top_idx[np.lexsort((od_r[top_idx], -rev[top_idx]))]

    return [
        (
            int(unique_ok[i]),
            round(float(rev[i]), 4),
            _fmt_date(int(od_r[i])),
            int(sp_r[i]),
        )
        for i in top_idx
    ]


def _numpy_query(
    arrays: FetchedArrays,
    segment: str,
    d: np.datetime64,
) -> Q3Result:
    d_i64 = _date_i64(d)
    seg_k = _SEG_TO_U8.get(segment, 255)

    if arrays.seg_group_info is not None:
        lookup_key = seg_k if arrays.seg_key is not None else 0
        gi = arrays.seg_group_info.get(lookup_key)
        if gi is None:
            return []
        mask = (gi["od_slice"] < d_i64) & (gi["sd_slice"] > d_i64)
        if not mask.any():
            return []
        return _top10_from_arrays(
            gi["ok_slice"][mask],
            gi["ep_slice"][mask],
            gi["od_slice"][mask],
            gi["sp_slice"][mask],
        )

    # Fallback: unsorted
    if arrays.seg_key is not None:
        seg_mask = arrays.seg_key == seg_k
    elif arrays.seg_raw is not None:
        seg_mask = np.frompyfunc(lambda s: s == segment, 1, 1)(arrays.seg_raw).astype(
            bool
        )
    else:
        seg_mask = np.ones(len(arrays.orderkey), dtype=bool)

    date_mask = (arrays.orderdate < d_i64) & (arrays.shipdate > d_i64)
    mask = seg_mask & date_mask
    if not mask.any():
        return []
    return _top10_from_arrays(
        arrays.orderkey[mask],
        arrays.ep_disc[mask],
        arrays.orderdate[mask],
        arrays.shippriority[mask],
    )


# Route each query to the right path depending on which steps SQL owns.
def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    segment: str,
    d: np.datetime64,
) -> Q3Result:

    if cfg.sql_handles("C") and cfg.sql_handles("D"):
        sql = build_per_query_sql(cfg, segment, d)
        rows = con.execute(sql).fetchall()
        if cfg.sql_handles("G"):
            return [
                (int(r[0]), round(float(r[1]), 4), str(r[2])[:10], int(r[3]))
                for r in rows
            ]
        raw = con.execute(sql).fetchnumpy()
        return _top10_from_arrays(
            raw["l_orderkey"].astype(np.int64),
            raw["ep_disc"].astype(np.float64),
            _to_i64(raw["o_orderdate"]),
            raw["o_shippriority"].astype(np.int32),
        )

    if cfg.sql_handles("D") and cfg.python_handles("C"):
        sql = build_per_query_sql_d_only(cfg, d)
        raw = con.execute(sql).fetchnumpy()
        seg_mask = np.frompyfunc(lambda s: s == segment, 1, 1)(
            raw["c_mktsegment"]
        ).astype(bool)
        if not seg_mask.any():
            return []
        return _top10_from_arrays(
            raw["l_orderkey"].astype(np.int64)[seg_mask],
            raw["ep_disc"].astype(np.float64)[seg_mask],
            _to_i64(raw["o_orderdate"])[seg_mask],
            raw["o_shippriority"].astype(np.int32)[seg_mask],
        )

    assert arrays is not None
    return _numpy_query(arrays, segment, d)


# Orchestrate fetch + N-query loop for one combo; return a timed SweepResult.
def run_sweep(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    params: list[SweepParam],
) -> SweepResult:
    valid, reason = is_valid_combo(cfg)
    if not valid:
        raise ValueError(f"Invalid combo sql={cfg.key}: {reason}")

    fetch_time = 0.0
    results: list[Q3Result] = []

    if cfg.sql_handles("C") and cfg.sql_handles("D"):
        t1 = time.perf_counter()
        for seg, d in params:
            results.append(run_one_query(con, cfg, None, seg, d))
        logic_time = time.perf_counter() - t1

    elif cfg.sql_handles("C") and cfg.python_handles("D"):
        by_seg: dict[str, list[np.datetime64]] = defaultdict(list)
        for seg, d in params:
            by_seg[seg].append(d)
        seg_arrays: dict[str, FetchedArrays] = {}
        for seg in by_seg:
            arr, ft = fetch_and_prepare(con, cfg, segment=seg)
            fetch_time += ft
            seg_arrays[seg] = arr
        t1 = time.perf_counter()
        for seg, d in params:
            results.append(_numpy_query(seg_arrays[seg], seg, d))
        logic_time = time.perf_counter() - t1

    else:
        arrays, fetch_time = fetch_and_prepare(con, cfg)
        t1 = time.perf_counter()
        for seg, d in params:
            results.append(run_one_query(con, cfg, arrays, seg, d))
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
def _rows_match(a: Q3Row, b: Q3Row, tol: float = 1.0) -> bool:
    return a[0] == b[0] and abs(a[1] - b[1]) <= tol and a[2] == b[2] and a[3] == b[3]


def validate(
    reference: list[Q3Result],
    candidate: list[Q3Result],
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
        for p, ref, cand, kind in mismatches[:3]:
            logger.warning(f"    seg={p[0]} date={_date_str(p[1])} ({kind})")
            logger.warning(f"      ref: {ref[:2]}")
            logger.warning(f"      got: {cand[:2]}")
        return False
    logger.info(f"  [{label}] all results match reference")
    return True


# Plug Q3 into the shared benchmark harness.
class Q3Benchmark(QueryBenchmark):
    NAME = "Q3"
    ALL_STEPS = ("C", "D", "G")
    N_APPLICABLE = True
    N_HELP = "Number of cutoff dates in [1995-03-01, 1995-03-31] (default: 10); total params = 5 segments * n"
    N_DEFAULT = 10

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_precompute=args.opt_precompute,
            opt_encode_seg=args.opt_encode_seg,
            opt_presort=args.opt_presort,
        )

    def generate_params(self, n: int) -> list[SweepParam]:
        return generate_sweep_params(n)

    def single_params(self) -> list[SweepParam]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query ({SINGLE_SEGMENT}, {_date_str(SINGLE_DATE)})"

    def is_valid_combo(self, cfg: BenchConfig) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg: BenchConfig, params) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"precompute={'ON' if args.opt_precompute else 'OFF'}  "
            f"encode_seg={'ON' if args.opt_encode_seg else 'OFF'}  "
            f"presort={'ON' if args.opt_presort else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_precompute",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Precompute ep*(1-disc) at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_seg",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Encode segment string to uint8 at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort by (seg_key, orderkey) at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        n_dates = args.n if args.n is not None else self.N_DEFAULT
        return f"sweep  N={len(params)} params ({len(ALL_SEGMENTS)} segments * {n_dates} dates)"


# Run directly: python3 -m benchmarks.q3
def main() -> None:
    bench = Q3Benchmark()
    parser = make_base_parser(
        "Q3 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q3_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
