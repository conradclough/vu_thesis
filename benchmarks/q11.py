"""
Q11 - important stock identification: sweep over 25 nations * fraction thresholds.

    WITH vals AS (
      SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
      FROM partsupp
      JOIN supplier ON ps_suppkey = s_suppkey
      JOIN nation   ON s_nationkey = n_nationkey
      WHERE n_name = '[NATION]'
      GROUP BY ps_partkey
    )
    SELECT ps_partkey, value FROM vals
    WHERE value > (SELECT SUM(value) FROM vals) * [FRACTION]
    ORDER BY value DESC

The 3-table join re-executes and re-groups per (nation, fraction) query pair.
Python bulk-fetches all nations at once, precomputes per-nation part sums and totals,
then applies the HAVING threshold as a cheap array comparison - O(n_parts_per_nation).

Steps (each independently assigned to SQL or Python):
  [N] nation   : n_name = '[NATION]'
  [G] group+sum: GROUP BY ps_partkey + SUM(ps_supplycost * ps_availqty)
  [F] fraction : HAVING value > total_nation_value * fraction

Validity:
  G in SQL without N: SQL groups across all nations; the per-nation total denominator
    for HAVING is wrong. G in SQL forces N in SQL.
  F in SQL without G: HAVING presupposes a GROUP BY result. F in SQL forces G (and N).

Opt flags (all ON by default):
  --opt_precompute    compute ps_supplycost * ps_availqty in SQL at fetch time
  --opt_encode_nation map n_name to uint8 in SQL at fetch time
  --opt_presort       sort by (nation_key, ps_partkey) once; per-nation slice is
                      contiguous for searchsorted, partkeys within it are sorted
  --opt_cumsum        precompute per-nation (partkey_arr, part_sum_arr, total) at
                      fetch time; per-query cost is one array comparison
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

ALL_STEPS = ("N", "G", "F")

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

SINGLE_NATION = "GERMANY"
SINGLE_FRACTION = 0.0001
FRAC_LO = 0.00001
FRAC_HI = 0.001

Q11Row = tuple[int, float]
Q11Result = list[Q11Row]
SweepParam = tuple[str, float]


def generate_params(n: int) -> list[SweepParam]:
    nations = [ALL_NATIONS[i % len(ALL_NATIONS)] for i in range(n)]
    fractions = list(np.linspace(FRAC_LO, FRAC_HI, n))
    return list(zip(nations, fractions))


def single_query_params() -> list[SweepParam]:
    return [(SINGLE_NATION, SINGLE_FRACTION)]


@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_precompute: bool = True
    opt_encode_nation: bool = True
    opt_presort: bool = True
    opt_cumsum: bool = True

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


def is_valid_combo(cfg: BenchConfig) -> tuple[bool, str]:
    if cfg.sql_handles("G") and cfg.python_handles("N"):
        return False, (
            "G in SQL requires N in SQL: GROUP BY collapses nation-specific data; "
            "Python cannot apply a per-nation threshold on a cross-nation aggregate."
        )
    if cfg.sql_handles("F") and cfg.python_handles("G"):
        return False, (
            "F in SQL requires G in SQL: HAVING presupposes a GROUP BY result. "
            "SQL cannot apply the fraction threshold without the GROUP BY."
        )
    return True, ""


_BASE_JOIN = (
    "FROM partsupp\n"
    "JOIN supplier ON ps_suppkey = s_suppkey\n"
    "JOIN nation   ON s_nationkey = n_nationkey"
)


def _nation_key_sql() -> str:
    cases = " ".join(f"WHEN '{n}' THEN {i}" for n, i in _NAT_TO_U8.items())
    return f"CASE n_name {cases} ELSE 255 END AS n_key"


def build_bulk_fetch_sql(cfg: BenchConfig) -> str:
    assert cfg.python_handles("N")
    val_col = (
        "ps_supplycost * ps_availqty AS value"
        if cfg.opt_precompute
        else "ps_supplycost, ps_availqty"
    )
    nat_col = _nation_key_sql() if cfg.opt_encode_nation else "n_name"
    return f"SELECT ps_partkey, {val_col}, {nat_col}\n{_BASE_JOIN}"


def build_per_query_sql_n(nation: str, cfg: BenchConfig) -> str:
    assert cfg.sql_handles("N")
    val_col = (
        "ps_supplycost * ps_availqty AS value"
        if cfg.opt_precompute
        else "ps_supplycost, ps_availqty"
    )
    return f"SELECT ps_partkey, {val_col}\n{_BASE_JOIN}\nWHERE n_name = '{nation}'"


def build_per_query_sql_ng(nation: str) -> str:
    return (
        "SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value\n"
        f"{_BASE_JOIN}\n"
        f"WHERE n_name = '{nation}'\n"
        "GROUP BY ps_partkey\n"
        "ORDER BY value DESC"
    )


def build_per_query_sql_ngf(nation: str, fraction: float) -> str:
    return (
        "WITH vals AS (\n"
        "  SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value\n"
        f"  {_BASE_JOIN}\n"
        f"  WHERE n_name = '{nation}'\n"
        "  GROUP BY ps_partkey\n"
        ")\n"
        "SELECT ps_partkey, value FROM vals\n"
        f"WHERE value > (SELECT SUM(value) FROM vals) * {fraction}\n"
        "ORDER BY value DESC"
    )


@dataclass
class FetchedArrays:
    value: np.ndarray
    partkey: np.ndarray
    nation_key: np.ndarray | None
    nation_raw: np.ndarray | None
    presorted: bool
    nation_data: dict | None


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    if cfg.sql_handles("N"):
        return None, 0.0

    sql = build_bulk_fetch_sql(cfg)
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    fetch_time = time.perf_counter() - t0

    if cfg.opt_precompute:
        value = raw["value"].astype(np.float64)
    else:
        value = raw["ps_supplycost"].astype(np.float64) * raw["ps_availqty"].astype(
            np.float64
        )

    partkey = raw["ps_partkey"].astype(np.int32)

    if cfg.opt_encode_nation:
        nation_key = raw["n_key"].astype(np.uint8)
        nation_raw = None
    else:
        nation_key = None
        nation_raw = raw["n_name"]

    presorted = False
    nation_data = None

    if cfg.opt_presort and nation_key is not None:
        sort_idx = np.lexsort((partkey, nation_key))
        value = value[sort_idx]
        partkey = partkey[sort_idx]
        nation_key = nation_key[sort_idx]
        presorted = True

        if cfg.opt_cumsum:
            # Per (nation, partkey) group: compute sum of value.
            # Pack into int64 compound key so np.unique gives one entry per group.
            compound = nation_key.astype(np.int64) * (1 << 24) + partkey.astype(
                np.int64
            )
            unique_comp, g_starts, _ = np.unique(
                compound, return_index=True, return_counts=True
            )
            group_sums = np.add.reduceat(value, g_starts)

            group_nat = (unique_comp >> 24).astype(np.uint8)
            group_pk = (unique_comp & 0xFFFFFF).astype(np.int32)

            unique_nats, ns, nc = np.unique(
                group_nat, return_index=True, return_counts=True
            )
            nation_data = {}
            for n_key_val, n_start, n_count in zip(unique_nats, ns, nc):
                pk_arr = group_pk[n_start : n_start + n_count]
                sum_arr = group_sums[n_start : n_start + n_count]
                total = float(sum_arr.sum())
                desc_order = np.argsort(sum_arr)[::-1]
                nation_data[int(n_key_val)] = {
                    "partkey": pk_arr[desc_order],
                    "part_sum": sum_arr[desc_order],
                    "total": total,
                }

    return FetchedArrays(
        value=value,
        partkey=partkey,
        nation_key=nation_key,
        nation_raw=nation_raw,
        presorted=presorted,
        nation_data=nation_data,
    ), fetch_time


def _numpy_query(
    arrays: FetchedArrays,
    nation: str,
    fraction: float,
) -> Q11Result:
    # PATH 1: precomputed per-nation part sums and total
    if arrays.nation_data is not None:
        n_key = _NAT_TO_U8.get(nation, 255)
        entry = arrays.nation_data.get(n_key)
        if entry is None:
            return []
        pk_arr = entry["partkey"]
        sum_arr = entry["part_sum"]
        total = entry["total"]
        threshold = total * fraction
        mask = sum_arr > threshold
        return [(int(pk), float(v)) for pk, v in zip(pk_arr[mask], sum_arr[mask])]

    # PATH 2: presorted by (nation_key, partkey) - searchsorted to find nation slice
    if arrays.presorted and arrays.nation_key is not None:
        n_key = _NAT_TO_U8.get(nation, 255)
        lo = int(np.searchsorted(arrays.nation_key, n_key, side="left"))
        hi = int(np.searchsorted(arrays.nation_key, n_key, side="right"))
        if lo >= hi:
            return []
        pk_slice = arrays.partkey[lo:hi]
        val_slice = arrays.value[lo:hi]
    elif arrays.nation_key is not None:
        n_key = _NAT_TO_U8.get(nation, 255)
        mask = arrays.nation_key == n_key
        if not mask.any():
            return []
        pk_slice = arrays.partkey[mask]
        val_slice = arrays.value[mask]
    else:
        mask = np.frompyfunc(lambda s: s == nation, 1, 1)(arrays.nation_raw).astype(
            bool
        )
        if not mask.any():
            return []
        pk_slice = arrays.partkey[mask]
        val_slice = arrays.value[mask]

    # Group by partkey and sum via bincount (works on unsorted input)
    unique_pk, inverse = np.unique(pk_slice, return_inverse=True)
    part_sums = np.bincount(inverse, weights=val_slice)
    total = float(part_sums.sum())
    threshold = total * fraction
    keep = part_sums > threshold
    result = [(int(pk), float(v)) for pk, v in zip(unique_pk[keep], part_sums[keep])]
    return sorted(result, key=lambda r: r[1], reverse=True)


def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    nation: str,
    fraction: float,
) -> Q11Result:
    if cfg.sql_handles("N") and cfg.sql_handles("G") and cfg.sql_handles("F"):
        rows = con.execute(build_per_query_sql_ngf(nation, fraction)).fetchall()
        return [(int(r[0]), round(float(r[1]), 4)) for r in rows]

    if cfg.sql_handles("N") and cfg.sql_handles("G"):
        rows = con.execute(build_per_query_sql_ng(nation)).fetchall()
        if not rows:
            return []
        sum_arr = np.array([r[1] for r in rows], dtype=np.float64)
        total = float(sum_arr.sum())
        threshold = total * fraction
        result = [
            (int(r[0]), round(float(r[1]), 4)) for r in rows if float(r[1]) > threshold
        ]
        return sorted(result, key=lambda r: r[1], reverse=True)

    if cfg.sql_handles("N"):
        raw = con.execute(build_per_query_sql_n(nation, cfg)).fetchnumpy()
        if cfg.opt_precompute:
            val = raw["value"].astype(np.float64)
        else:
            val = raw["ps_supplycost"].astype(np.float64) * raw["ps_availqty"].astype(
                np.float64
            )
        pk = raw["ps_partkey"].astype(np.int32)
        unique_pk, inverse = np.unique(pk, return_inverse=True)
        part_sums = np.bincount(inverse, weights=val)
        total = float(part_sums.sum())
        threshold = total * fraction
        keep = part_sums > threshold
        result = [
            (int(p), round(float(v), 4))
            for p, v in zip(unique_pk[keep], part_sums[keep])
        ]
        return sorted(result, key=lambda r: r[1], reverse=True)

    assert arrays is not None
    return _numpy_query(arrays, nation, fraction)


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
    for nation, fraction in params:
        results.append(run_one_query(con, cfg, arrays, nation, fraction))
    logic_time = time.perf_counter() - t0

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


def validate(
    reference: list[Q11Result],
    candidate: list[Q11Result],
    params: list[SweepParam],
    label: str,
    tol: float = 1.0,
) -> bool:
    mismatches = []
    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        ref_s = sorted(ref, key=lambda r: r[0])
        cand_s = sorted(cand, key=lambda r: r[0])
        if len(ref_s) != len(cand_s):
            mismatches.append((params[i], ref, cand, "length"))
            continue
        for rr, rc in zip(ref_s, cand_s):
            if rr[0] != rc[0] or abs(rr[1] - rc[1]) > tol:
                mismatches.append((params[i], ref, cand, "value"))
                break
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, ref, cand, kind in mismatches[:3]:
            logger.warning(f"    nation={p[0]} fraction={p[1]:.6f} ({kind})")
            logger.warning(f"      ref: {ref[:3]}")
            logger.warning(f"      got: {cand[:3]}")
        return False
    logger.info(f"  [{label}] all results match reference")
    return True


class Q11Benchmark(QueryBenchmark):
    NAME = "Q11"
    ALL_STEPS = ("N", "G", "F")
    N_APPLICABLE = True
    N_HELP = "Number of (nation, fraction) sweep pairs (default: 100)"
    N_DEFAULT = 100

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_precompute=args.opt_precompute,
            opt_encode_nation=args.opt_encode_nation,
            opt_presort=args.opt_presort,
            opt_cumsum=args.opt_cumsum,
        )

    def generate_params(self, n: int) -> list[SweepParam]:
        return generate_params(n)

    def single_params(self) -> list[SweepParam]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query (nation={SINGLE_NATION}, fraction={SINGLE_FRACTION})"

    def is_valid_combo(self, cfg) -> tuple[bool, str]:
        return is_valid_combo(cfg)

    def run_sweep(self, con, cfg, params) -> SweepResult:
        return run_sweep(con, cfg, params)

    def validate(self, reference, candidate, params, label) -> bool:
        return validate(reference, candidate, params, label)

    def opt_flags_str(self, args) -> str:
        return (
            f"precompute={'ON' if args.opt_precompute else 'OFF'}  "
            f"encode_nation={'ON' if args.opt_encode_nation else 'OFF'}  "
            f"presort={'ON' if args.opt_presort else 'OFF'}  "
            f"cumsum={'ON' if args.opt_cumsum else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_precompute",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Compute ps_supplycost*ps_availqty in SQL at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_nation",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Encode n_name → uint8 in SQL at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Sort by (nation_key, ps_partkey) at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_cumsum",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Precompute per-nation part sums and totals at fetch time (default: ON)",
        )

    def log_mode(self, args, params: list) -> str:
        return f"sweep  N={len(params)} (nation, fraction) pairs"


def main() -> None:
    bench = Q11Benchmark()
    parser = make_base_parser(
        "Q11 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q11_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)
    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
