"""
Q7 - shipping revenue between nation pairs, sweep over C(25,2)=300 nation combinations.

    SELECT supp_nation, cust_nation, l_year,
           SUM(l_extendedprice * (1 - l_discount)) AS revenue
    FROM supplier, lineitem, orders, customer,
         nation n1, nation n2
    WHERE s_suppkey   = l_suppkey
      AND o_orderkey  = l_orderkey
      AND c_custkey   = o_custkey
      AND s_nationkey = n1.n_nationkey
      AND c_nationkey = n2.n_nationkey
      AND ((n1.n_name = '[NATION1]' AND n2.n_name = '[NATION2]')
        OR (n1.n_name = '[NATION2]' AND n2.n_name = '[NATION1]'))
      AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
    GROUP BY supp_nation, cust_nation, l_year
    ORDER BY supp_nation, cust_nation, l_year

The 6-table join and fixed shipdate range (1995-1996) are always in SQL.
What varies is who handles the nation-pair filter and the year grouping.

Steps (each independently assigned to SQL or Python):
  [N] nation pair : (n1=:n1 AND n2=:n2) OR (n1=:n2 AND n2=:n1) - bidirectional filter
  [Y] year        : EXTRACT(YEAR FROM l_shipdate) as groupby key (1995 or 1996)
  [G] group+agg   : GROUP BY (supp_nation, cust_nation, year) + SUM(ep*(1-disc))

Validity: G collapses rows - Python can't post-aggregate, so G in SQL forces N and Y
into SQL too. N in SQL with Y in Python is valid but can't bulk-fetch (the nation pair
changes each iteration, so SQL must be called once per pair). Y in SQL with N in Python
is valid and uses a full bulk-fetch - SQL returns year as a column, Python filters rows.

Opt flags (all ON by default):
  --opt_presort  sort by (sn_key, cn_key, year_key) at fetch time - each direction group
                 is a contiguous slice, so year boundaries are searchsorted not masked
  --opt_cumsum   prefix sums per (sn_key, cn_key) group means O(1) per-query aggregation
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

# N/Y/G - the three steps and all valid subsets.

ALL_STEPS = ("N", "Y", "G")  # nation-pair filter, year groupby, aggregate

# Fixed shipdate range (1995–1996) and nation encoding from the TPC-H Q7 spec.
ALL_NATIONS = [
    "ALGERIA",
    "ARGENTINA",
    "BRAZIL",
    "CANADA",
    "EGYPT",
    "ETHIOPIA",
    "FRANCE",
    "GERMANY",
    "INDIA",
    "INDONESIA",
    "IRAN",
    "IRAQ",
    "JAPAN",
    "JORDAN",
    "KENYA",
    "MOROCCO",
    "MOZAMBIQUE",
    "PERU",
    "CHINA",
    "ROMANIA",
    "SAUDI ARABIA",
    "VIETNAM",
    "RUSSIA",
    "UNITED KINGDOM",
    "UNITED STATES",
]
_NAT_TO_U8: dict[str, int] = {n: i for i, n in enumerate(ALL_NATIONS)}
_U8_TO_NAT: dict[int, str] = {i: n for n, i in _NAT_TO_U8.items()}

# Q7 fixed shipdate range: 1995-01-01 to 1996-12-31 inclusive
ALL_YEARS = [1995, 1996]
_YR_TO_U8 = {1995: 0, 1996: 1}
_U8_TO_YR = {0: 1995, 1: 1996}

# TPC-H spec single query
SINGLE_N1 = "FRANCE"
SINGLE_N2 = "GERMANY"

# One result row per (supp_nation, cust_nation, year) triple.

# (supp_nation, cust_nation, year, revenue)
Q7Row = tuple[str, str, int, float]
Q7Result = list[Q7Row]
SweepParam = tuple[str, str]  # (nation1, nation2), always n1 < n2 alphabetically


# Sample N nation pairs from all C(25,2)=300 combinations.
def generate_sweep_params(n: int) -> list[SweepParam]:
    """Sample n pairs from all C(25,2)=300 nation pairs."""
    all_pairs = [(a, b) for a, b in itertools.combinations(sorted(ALL_NATIONS), 2)]
    if n >= len(all_pairs):
        return all_pairs

    # Evenly spaced sample
    indices = np.linspace(0, len(all_pairs) - 1, n, dtype=int)
    seen = set()
    result = []
    for i in indices:
        p = all_pairs[int(i)]
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def single_query_params() -> list[SweepParam]:
    n1, n2 = sorted([SINGLE_N1, SINGLE_N2])
    return [(n1, n2)]


# Which steps go to SQL vs Python, and which numpy opts are active.
@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_presort: bool = True  # sort by (sn_key, cn_key, yr_key)
    opt_cumsum: bool = True  # prefix sums per (sn,cn) group

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


# G in SQL collapses rows - Python can't post-aggregate, so G forces N+Y into SQL too.
def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("G") and cfg.python_steps - {"G"}:
        missing = sorted(cfg.python_steps - {"G"})
        return False, (
            f"G in SQL requires N and Y also in SQL (missing: {missing}). "
            "SQL GROUP BY collapses rows; Python cannot post-filter."
        )
    return True, ""


# Build the SELECT/WHERE/JOIN dynamically - what SQL sees changes per combo.
_BASE_JOIN = (
    "FROM supplier\n"
    "JOIN lineitem ON s_suppkey   = l_suppkey\n"
    "JOIN orders   ON o_orderkey  = l_orderkey\n"
    "JOIN customer ON c_custkey   = o_custkey\n"
    "JOIN nation n1 ON s_nationkey = n1.n_nationkey\n"
    "JOIN nation n2 ON c_nationkey = n2.n_nationkey"
)

_SHIPDATE_WHERE = "l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'"


def _nation_where(n1: str, n2: str) -> str:
    return (
        f"((n1.n_name = '{n1}' AND n2.n_name = '{n2}')\n"
        f"  OR (n1.n_name = '{n2}' AND n2.n_name = '{n1}'))"
    )


def _nation_key_sql(alias: str, col: str) -> str:
    """SQL CASE encoding nation name -> uint8."""
    cases = " ".join(f"WHEN '{n}' THEN {i}" for n, i in _NAT_TO_U8.items())
    return f"CASE {alias}.n_name {cases} ELSE 255 END AS {col}"


def build_fetch_sql_full(cfg: BenchConfig) -> str:
    """
    Full bulk fetch: N in Python (and Y may be in Python or SQL).
    Always applies the fixed shipdate filter.
    Encodes nation keys and year as uint8 for fast per-query groupby.
    """
    assert cfg.python_handles("N")

    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    sn_col = _nation_key_sql("n1", "sn_key")
    cn_col = _nation_key_sql("n2", "cn_key")
    yr_col = (
        "CASE EXTRACT(YEAR FROM l_shipdate) "
        "WHEN 1995 THEN 0 "
        "WHEN 1996 THEN 1 "
        "ELSE 255 END AS yr_key"
    )

    return (
        f"SELECT {ep_col}, {sn_col}, {cn_col}, {yr_col}\n"
        f"{_BASE_JOIN}\n"
        f"WHERE {_SHIPDATE_WHERE}"
    )


def build_per_query_sql_n_only(n1: str, n2: str) -> str:
    """N in SQL, Y+G in Python: filter to nation pair, return per-row data."""
    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    yr_col = (
        "CASE EXTRACT(YEAR FROM l_shipdate) "
        "WHEN 1995 THEN 0 "
        "WHEN 1996 THEN 1 "
        "ELSE 255 END AS yr_key"
    )
    sn_col = _nation_key_sql("n1", "sn_key")
    cn_col = _nation_key_sql("n2", "cn_key")

    return (
        f"SELECT {ep_col}, {sn_col}, {cn_col}, {yr_col}\n"
        f"{_BASE_JOIN}\n"
        f"WHERE {_SHIPDATE_WHERE}\n"
        f"  AND {_nation_where(n1, n2)}"
    )


def build_per_query_sql_ny_g(n1: str, n2: str) -> str:
    """N+Y in SQL, G in Python: filter + year column, return for Python groupby."""
    ep_col = "l_extendedprice * (1.0 - l_discount) AS ep_disc"
    yr_col = (
        "CASE EXTRACT(YEAR FROM l_shipdate) "
        "WHEN 1995 THEN 0 "
        "WHEN 1996 THEN 1 "
        "ELSE 255 END AS yr_key"
    )

    return (
        f"SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,\n"
        f"  {yr_col}, {ep_col}\n"
        f"{_BASE_JOIN}\n"
        f"WHERE {_SHIPDATE_WHERE}\n"
        f"  AND {_nation_where(n1, n2)}"
    )


def build_pure_sql(n1: str, n2: str) -> str:
    """N+Y+G all in SQL."""
    return (
        "SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,\n"
        "  EXTRACT(YEAR FROM l_shipdate) AS l_year,\n"
        "  SUM(l_extendedprice * (1.0 - l_discount)) AS revenue\n"
        f"{_BASE_JOIN}\n"
        f"WHERE {_SHIPDATE_WHERE}\n"
        f"  AND {_nation_where(n1, n2)}\n"
        "GROUP BY supp_nation, cust_nation, l_year\n"
        "ORDER BY supp_nation, cust_nation, l_year"
    )


# Arrays holding the bulk-fetched 6-table join result, with optional prefix sums.
@dataclass
class FetchedArrays:
    ep_disc: np.ndarray  # float64
    sn_key: np.ndarray  # uint8 supplier nation
    cn_key: np.ndarray  # uint8 customer nation
    yr_key: np.ndarray  # uint8: 0=1995, 1=1996
    presorted: bool = False
    # group_info[(sn_key, cn_key)] = {"start", "count", "yr_slice", "cs_ep"}
    group_info: dict | None = None


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    # N in SQL means per-query SQL calls - no bulk fetch possible.
    if cfg.sql_handles("N"):
        return None, 0.0

    # One SQL round-trip for the entire 6-table join (~2M rows at SF=1),
    # amortised across all nation-pair queries that follow.
    sql = build_fetch_sql_full(cfg)

    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    fetch_time = time.perf_counter() - t0

    ep_disc = raw["ep_disc"].astype(np.float64)
    # SQL already encoded nation and year as uint8 - per-query filter is == not string eq.
    sn_key = raw["sn_key"].astype(np.uint8)
    cn_key = raw["cn_key"].astype(np.uint8)
    yr_key = raw["yr_key"].astype(np.uint8)

    presorted = False
    group_info = None

    if cfg.opt_presort:
        # Sort by (sn, cn, yr) so each (direction, year) group is a contiguous slice.
        # DuckDB sorts implicitly on every GROUP BY; we pay this once.
        sort_idx = np.lexsort((yr_key, cn_key, sn_key))
        ep_disc = ep_disc[sort_idx]
        sn_key = sn_key[sort_idx]
        cn_key = cn_key[sort_idx]
        yr_key = yr_key[sort_idx]
        presorted = True

        if cfg.opt_cumsum:
            # Pack (sn_key, cn_key) into one uint16 key so np.unique gives one entry
            # per (direction) group; build a global prefix sum over ep_disc.
            # Per query: cs_ep[end] - cs_ep[start] is the group revenue in O(1).
            compound = (
                sn_key.astype(np.uint16) * 32 + cn_key.astype(np.uint16)
            ).astype(np.uint16)

            cs_ep = np.zeros(len(ep_disc) + 1, dtype=np.float64)
            cs_ep[1:] = np.cumsum(ep_disc)

            unique_comp, starts, counts = np.unique(
                compound,
                return_index=True,
                return_counts=True,
            )

            group_info = {}
            for comp, start, count in zip(unique_comp, starts, counts):
                sn = int(comp) // 32
                cn = int(comp) % 32
                group_info[(sn, cn)] = {
                    "start": int(start),
                    "count": int(count),
                    "yr_slice": yr_key[start : start + count],
                    "cs_ep": cs_ep,
                }

    arrays = FetchedArrays(
        ep_disc=ep_disc,
        sn_key=sn_key,
        cn_key=cn_key,
        yr_key=yr_key,
        presorted=presorted,
        group_info=group_info,
    )

    return arrays, fetch_time


# N tight loops - no SQL calls from here. The bulk fetch bought us that.
def _agg_direction(
    ep_disc: np.ndarray,
    yr_key: np.ndarray,
    sn_name: str,
    cn_name: str,
) -> list[Q7Row]:
    """Sum ep_disc by year for one (supp_nation, cust_nation) direction."""
    rows: list[Q7Row] = []

    for yr_u8, yr_val in _U8_TO_YR.items():
        mask = yr_key == yr_u8
        if mask.any():
            revenue = round(float(ep_disc[mask].sum()), 4)
            rows.append((sn_name, cn_name, yr_val, revenue))

    return rows


def _numpy_query(
    arrays: FetchedArrays,
    n1: str,
    n2: str,
) -> Q7Result:
    n1_key = _NAT_TO_U8.get(n1, 255)
    n2_key = _NAT_TO_U8.get(n2, 255)

    # Path 1: prefix-sum lookup
    if arrays.group_info is not None:
        result: Q7Result = []

        for (sn, cn), (sn_nm, cn_nm) in [
            ((n1_key, n2_key), (n1, n2)),
            ((n2_key, n1_key), (n2, n1)),
        ]:
            entry = arrays.group_info.get((sn, cn))
            if entry is None:
                continue

            start = entry["start"]
            yr_slice = entry["yr_slice"]
            cs = entry["cs_ep"]

            for yr_u8, yr_val in _U8_TO_YR.items():
                a = int(np.searchsorted(yr_slice, yr_u8, side="left"))
                b = int(np.searchsorted(yr_slice, yr_u8, side="right"))
                if a < b:
                    revenue = float(cs[start + b] - cs[start + a])
                    result.append((sn_nm, cn_nm, yr_val, round(revenue, 4)))

        return sorted(result, key=lambda r: (r[0], r[1], r[2]))

    # Path 2: presorted, no cumsum
    if arrays.presorted:
        result: Q7Result = []

        for sn, cn, sn_nm, cn_nm in [
            (n1_key, n2_key, n1, n2),
            (n2_key, n1_key, n2, n1),
        ]:
            dir_mask = (arrays.sn_key == sn) & (arrays.cn_key == cn)
            if not dir_mask.any():
                continue

            result.extend(
                _agg_direction(
                    arrays.ep_disc[dir_mask],
                    arrays.yr_key[dir_mask],
                    sn_nm,
                    cn_nm,
                )
            )

        return sorted(result, key=lambda r: (r[0], r[1], r[2]))

    # Path 3: unsorted fallback
    mask_fwd = (arrays.sn_key == n1_key) & (arrays.cn_key == n2_key)
    mask_rev = (arrays.sn_key == n2_key) & (arrays.cn_key == n1_key)

    result: Q7Result = []

    if mask_fwd.any():
        result.extend(
            _agg_direction(
                arrays.ep_disc[mask_fwd],
                arrays.yr_key[mask_fwd],
                n1,
                n2,
            )
        )

    if mask_rev.any():
        result.extend(
            _agg_direction(
                arrays.ep_disc[mask_rev],
                arrays.yr_key[mask_rev],
                n2,
                n1,
            )
        )

    return sorted(result, key=lambda r: (r[0], r[1], r[2]))


# Route each query to the right path depending on which steps SQL owns.
def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    n1: str,
    n2: str,
) -> Q7Result:
    n1, n2 = sorted([n1, n2])

    # Pure SQL
    if cfg.sql_handles("N") and cfg.sql_handles("Y") and cfg.sql_handles("G"):
        rows = con.execute(build_pure_sql(n1, n2)).fetchall()
        return [
            (
                str(r[0]),
                str(r[1]),
                int(r[2]),
                round(float(r[3]), 4),
            )
            for r in rows
        ]

    # N + Y in SQL, G in Python
    if cfg.sql_handles("N") and cfg.sql_handles("Y") and cfg.python_handles("G"):
        raw = con.execute(build_per_query_sql_ny_g(n1, n2)).fetchnumpy()

        ep = raw["ep_disc"].astype(np.float64)
        sn = raw["supp_nation"]
        cn = raw["cust_nation"]
        yr = raw["yr_key"].astype(np.uint8)

        result: Q7Result = []

        for sn_nm, cn_nm in [(n1, n2), (n2, n1)]:
            dir_mask = np.frompyfunc(lambda s: s == sn_nm, 1, 1)(sn).astype(
                bool
            ) & np.frompyfunc(lambda s: s == cn_nm, 1, 1)(cn).astype(bool)
            if not dir_mask.any():
                continue

            result.extend(
                _agg_direction(
                    ep[dir_mask],
                    yr[dir_mask],
                    sn_nm,
                    cn_nm,
                )
            )

        return sorted(result, key=lambda r: (r[0], r[1], r[2]))

    # N in SQL, Y + G in Python
    if cfg.sql_handles("N") and cfg.python_handles("Y"):
        raw = con.execute(build_per_query_sql_n_only(n1, n2)).fetchnumpy()

        ep = raw["ep_disc"].astype(np.float64)
        sn_key = raw["sn_key"].astype(np.uint8)
        cn_key = raw["cn_key"].astype(np.uint8)
        yr_key = raw["yr_key"].astype(np.uint8)

        n1_key = _NAT_TO_U8[n1]
        n2_key = _NAT_TO_U8[n2]

        result: Q7Result = []

        for sn, cn, sn_nm, cn_nm in [
            (n1_key, n2_key, n1, n2),
            (n2_key, n1_key, n2, n1),
        ]:
            mask = (sn_key == sn) & (cn_key == cn)
            if mask.any():
                result.extend(
                    _agg_direction(
                        ep[mask],
                        yr_key[mask],
                        sn_nm,
                        cn_nm,
                    )
                )

        return sorted(result, key=lambda r: (r[0], r[1], r[2]))

    # N in Python
    assert arrays is not None
    return _numpy_query(arrays, n1, n2)


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

    t0 = time.perf_counter()
    for n1, n2 in params:
        results.append(run_one_query(con, cfg, arrays, n1, n2))
    logic_time = time.perf_counter() - t0

    return SweepResult(
        key=cfg.key,
        sql_steps_str=cfg.key if cfg.sql_steps else "(none)",
        python_steps_str=(
            "".join(s for s in ALL_STEPS if cfg.python_handles(s)) or "(none)"
        ),
        values=results,
        fetch_time=fetch_time,
        logic_time=logic_time,
        total_time=fetch_time + logic_time,
    )


# Check Python results against the SQL reference
def validate(
    reference: list[Q7Result],
    candidate: list[Q7Result],
    params: list[SweepParam],
    label: str,
    tol: float = 1.0,
) -> bool:
    mismatches = []

    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        if len(ref) != len(cand):
            mismatches.append((params[i], ref, cand, "length"))
            continue

        for rr, rc in zip(ref, cand):
            if (
                rr[0] != rc[0]
                or rr[1] != rc[1]
                or rr[2] != rc[2]
                or abs(rr[3] - rc[3]) > tol
            ):
                mismatches.append((params[i], ref, cand, "value"))
                break

    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, ref, cand, kind in mismatches[:3]:
            logger.warning(f"    pair=({p[0]},{p[1]}) ({kind})")
            logger.warning(f"      ref: {ref}")
            logger.warning(f"      got: {cand}")
        return False

    logger.info(f"  [{label}] all results match reference")
    return True


# Plug Q7 into the shared benchmark harness.
class Q7Benchmark(QueryBenchmark):
    NAME = "Q7"
    ALL_STEPS = ("N", "Y", "G")
    N_APPLICABLE = True
    N_HELP = "Number of nation pairs to sample from C(25,2)=300 (default: 50)"
    N_DEFAULT = 50

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_presort=args.opt_presort,
            opt_cumsum=args.opt_cumsum,
        )

    def generate_params(self, n: int) -> list[SweepParam]:
        return generate_sweep_params(n)

    def single_params(self) -> list[SweepParam]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query ({SINGLE_N1}, {SINGLE_N2})"

    def is_valid_combo(
        self,
        cfg: BenchConfig,
    ) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(
        self,
        con,
        cfg: BenchConfig,
        params,
    ) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(
        self,
        reference,
        candidate,
        params,
        label,
    ) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"presort={'ON' if args.opt_presort else 'OFF'}  "
            f"cumsum={'ON' if args.opt_cumsum else 'OFF'}"
        )

    def add_query_args(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=("Sort by (sn_key, cn_key, yr_key) at fetch time (default: ON)"),
        )
        parser.add_argument(
            "--opt_cumsum",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Build prefix sums per (sn,cn) group "
                "for O(1) per-query work (default: ON)"
            ),
        )

    def log_mode(self, args, params: list) -> str:
        return f"sweep N={len(params)} nation pairs (from C(25,2)=300)"


# Run directly: python3 -m benchmarks.q7
def main() -> None:
    bench = Q7Benchmark()

    parser = make_base_parser(
        "Q7 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir,
        f"q7_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log",
    )

    setup_logging(log_filename)

    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(
        sf=args.sf,
        memory_limit=args.memory_limit,
    )

    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
