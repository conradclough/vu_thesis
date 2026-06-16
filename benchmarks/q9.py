"""
Q9 - product type profit per nation/year, sweep over the 92 TPC-H color keywords.

    SELECT n_name AS nation,
           EXTRACT(YEAR FROM o_orderdate) AS o_year,
           SUM(l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity) AS sum_profit
    FROM part
    JOIN lineitem ON p_partkey = l_partkey
    JOIN partsupp ON ps_partkey = l_partkey AND ps_suppkey = l_suppkey
    JOIN orders   ON o_orderkey = l_orderkey
    JOIN supplier ON s_suppkey  = l_suppkey
    JOIN nation   ON n_nationkey = s_nationkey
    WHERE p_name LIKE '%[COLOR]%'
    GROUP BY n_name, EXTRACT(YEAR FROM o_orderdate)
    ORDER BY nation, o_year DESC

The 6-table join is expensive; DuckDB re-executes it with the LIKE filter per color.
Python bulk-fetches the full join once (no color filter) plus the part table once,
pre-encodes color membership per row at fetch time, and sweeps N color queries cheaply.

Steps (each independently assigned to SQL or Python):
  [C] color   : p_name LIKE '%color%'
  [G] group+agg: GROUP BY (nation, year) + SUM(amount)

Validity: G in SQL collapses rows; Python cannot post-filter an aggregate. So G in SQL
forces C in SQL too (they always move together). Valid combos: (none), C, CG.

Opt flags (all ON by default):
  --opt_precompute    compute amount = ep*(1-disc) - ps_supplycost*qty in SQL at fetch time
  --opt_encode_nation map n_name -> uint8 and o_year means uint8 in SQL at fetch time
  --opt_encode_color  precompute per-color row membership (part table parse once at fetch
                      time); avoids O(n) numpy string scan per query
  --opt_presort       within each color's matching rows, sort by (nation_key, year_key)
                      so group boundaries can be found with searchsorted
  --opt_cumsum        build prefix sums per (color, nation, year) group for O(G) per-query
                      aggregation where G = number of (nation,year) groups, about 175
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

ALL_STEPS = ("C", "G")

# 92-word TPC-H color syllable list (used to construct p_name as 5 random words)
TPC_H_COLORS: list[str] = [
    "almond",
    "antique",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanched",
    "blue",
    "blush",
    "brown",
    "burlywood",
    "burnished",
    "chartreuse",
    "chiffon",
    "chocolate",
    "coral",
    "cornflower",
    "cornsilk",
    "cream",
    "cyan",
    "dark",
    "deep",
    "dim",
    "dodger",
    "drab",
    "firebrick",
    "floral",
    "forest",
    "frosted",
    "gainsboro",
    "ghost",
    "goldenrod",
    "green",
    "grey",
    "honeydew",
    "hot",
    "indian",
    "ivory",
    "khaki",
    "lace",
    "lavender",
    "lawn",
    "lemon",
    "light",
    "lime",
    "linen",
    "magenta",
    "maroon",
    "medium",
    "metallic",
    "midnight",
    "mint",
    "misty",
    "moccasin",
    "navajo",
    "navy",
    "olive",
    "orange",
    "orchid",
    "pale",
    "papaya",
    "peach",
    "peru",
    "pink",
    "plum",
    "powder",
    "puff",
    "purple",
    "red",
    "rose",
    "rosy",
    "royal",
    "saddle",
    "salmon",
    "sandy",
    "seashell",
    "sienna",
    "sky",
    "slate",
    "smoke",
    "snow",
    "spring",
    "steel",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "yellow",
]
_COLOR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TPC_H_COLORS)}
N_COLORS = len(TPC_H_COLORS)

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

# TPC-H orders span 1992-01-01 to 1998-12-31
ALL_YEARS = list(range(1992, 1999))
_YR_TO_U8: dict[int, int] = {y: i for i, y in enumerate(ALL_YEARS)}
_U8_TO_YR: dict[int, int] = {i: y for y, i in _YR_TO_U8.items()}

SINGLE_COLOR = "green"

Q9Row = tuple[str, int, float]
Q9Result = list[Q9Row]
SweepParam = str  # color keyword


def generate_color_params(n: int) -> list[SweepParam]:
    """Sample n colors from the 92-word list, cycling if n > 92."""
    indices = np.linspace(0, N_COLORS - 1, min(n, N_COLORS), dtype=int)
    base = [TPC_H_COLORS[int(i)] for i in indices]
    if n <= N_COLORS:
        return base
    # Repeat to reach n
    result = []
    for i in range(n):
        result.append(base[i % len(base)])
    return result


def single_query_params() -> list[SweepParam]:
    return [SINGLE_COLOR]


@dataclass
class BenchConfig:
    sql_steps: frozenset
    opt_precompute: bool = True
    opt_encode_nation: bool = True
    opt_encode_color: bool = True
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
    if cfg.sql_handles("G") and cfg.python_handles("C"):
        return False, (
            "G in SQL requires C in SQL: GROUP BY collapses rows; "
            "Python cannot post-filter an aggregate by color."
        )
    return True, ""


_BASE_JOIN = (
    "FROM part\n"
    "JOIN lineitem ON p_partkey = l_partkey\n"
    "JOIN partsupp ON ps_partkey = l_partkey AND ps_suppkey = l_suppkey\n"
    "JOIN orders   ON o_orderkey = l_orderkey\n"
    "JOIN supplier ON s_suppkey  = l_suppkey\n"
    "JOIN nation   ON n_nationkey = s_nationkey"
)


def _nation_key_sql() -> str:
    cases = " ".join(f"WHEN '{n}' THEN {i}" for n, i in _NAT_TO_U8.items())
    return f"CASE n_name {cases} ELSE 255 END AS n_key"


def _year_key_sql() -> str:
    cases = " ".join(f"WHEN {y} THEN {i}" for y, i in _YR_TO_U8.items())
    return f"CASE EXTRACT(YEAR FROM o_orderdate) {cases} ELSE 255 END AS yr_key"


def build_bulk_fetch_sql(cfg: BenchConfig) -> str:
    """Bulk fetch without color filter (C in Python). Includes l_partkey for color lookup."""
    assert cfg.python_handles("C")
    amt = (
        "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity AS amount"
        if cfg.opt_precompute
        else "l_extendedprice, l_discount, ps_supplycost, l_quantity"
    )
    nat = _nation_key_sql() if cfg.opt_encode_nation else "n_name"
    yr = (
        _year_key_sql()
        if cfg.opt_encode_nation
        else "EXTRACT(YEAR FROM o_orderdate) AS o_year"
    )
    return f"SELECT l_partkey, {amt}, {nat}, {yr}\n{_BASE_JOIN}"


def build_per_query_sql_c(color: str, cfg: BenchConfig) -> str:
    """C in SQL, G in Python: SQL filters by color, returns per-row data."""
    assert cfg.sql_handles("C") and cfg.python_handles("G")
    amt = (
        "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity AS amount"
        if cfg.opt_precompute
        else "l_extendedprice, l_discount, ps_supplycost, l_quantity"
    )
    nat = _nation_key_sql() if cfg.opt_encode_nation else "n_name"
    yr = (
        _year_key_sql()
        if cfg.opt_encode_nation
        else "EXTRACT(YEAR FROM o_orderdate) AS o_year"
    )
    return f"SELECT {amt}, {nat}, {yr}\n{_BASE_JOIN}\nWHERE p_name LIKE '%{color}%'"


def build_per_query_sql_cg(color: str) -> str:
    """CG (all SQL): full Q9 with color filter + GROUP BY."""
    return (
        "SELECT n_name AS nation,\n"
        "  EXTRACT(YEAR FROM o_orderdate) AS o_year,\n"
        "  SUM(l_extendedprice*(1.0-l_discount) - ps_supplycost*l_quantity) AS sum_profit\n"
        f"{_BASE_JOIN}\n"
        f"WHERE p_name LIKE '%{color}%'\n"
        "GROUP BY n_name, EXTRACT(YEAR FROM o_orderdate)\n"
        "ORDER BY nation, o_year DESC"
    )


@dataclass
class FetchedArrays:
    # Main bulk arrays
    amount: np.ndarray  # float64
    partkey: np.ndarray  # int32
    nation_key: np.ndarray | None  # uint8
    year_key: np.ndarray | None  # uint8
    nation_raw: np.ndarray | None  # object strings
    year_raw: np.ndarray | None  # int64

    # Color membership: part_has_color[partkey_idx, color_idx] = True if p_name contains color
    # Shape: (max_partkey + 1, N_COLORS), dtype bool
    part_has_color: np.ndarray | None

    # Precomputed per-color structures (opt_encode_color=ON with opt_presort+opt_cumsum)
    # color_data[color_idx] = {
    #   'nat_key': uint8 array (one per group, sorted by (nat,yr)),
    #   'yr_key': uint8 array (one per group),
    #   'g_starts': int array (start of group in this color's sorted amount slice),
    #   'g_counts': int array (group sizes),
    #   'cs': float64 prefix-sum array (len = n_matching_rows + 1),
    # }  OR None if no rows match
    color_data: dict | None


def _build_amount(raw: dict, cfg: BenchConfig) -> np.ndarray:
    if cfg.opt_precompute:
        return raw["amount"].astype(np.float64)
    ep = raw["l_extendedprice"].astype(np.float64)
    disc = raw["l_discount"].astype(np.float64)
    sc = raw["ps_supplycost"].astype(np.float64)
    qty = raw["l_quantity"].astype(np.float64)
    return ep * (1.0 - disc) - sc * qty


def fetch_and_prepare(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
) -> tuple[FetchedArrays | None, float]:
    if cfg.sql_handles("C"):
        return None, 0.0

    sql = build_bulk_fetch_sql(cfg)
    t0 = time.perf_counter()
    raw = con.execute(sql).fetchnumpy()
    fetch_time = time.perf_counter() - t0

    amount = _build_amount(raw, cfg)
    partkey = raw["l_partkey"].astype(np.int32)

    if cfg.opt_encode_nation:
        nation_key = raw["n_key"].astype(np.uint8)
        year_key = raw["yr_key"].astype(np.uint8)
        nation_raw = year_raw = None
    else:
        nation_key = year_key = None
        nation_raw = raw["n_name"]
        year_raw = raw["o_year"].astype(np.int64)

    # Fetch part table to get p_name per partkey (200K rows - much smaller than bulk)
    part_has_color: np.ndarray | None = None
    color_data: dict | None = None

    if cfg.opt_encode_color and nation_key is not None:
        t_part = time.perf_counter()
        part_raw = con.execute(
            "SELECT p_partkey, p_name FROM part ORDER BY p_partkey"
        ).fetchnumpy()
        fetch_time += time.perf_counter() - t_part

        part_pkeys = part_raw["p_partkey"].astype(np.int32)
        part_names = part_raw["p_name"]
        max_pk = int(part_pkeys.max())

        # Build (max_pk+1) * N_COLORS bool matrix in one pass over parts
        phc = np.zeros((max_pk + 1, N_COLORS), dtype=bool)
        for pk, name in zip(part_pkeys, part_names):
            for word in str(name).split():
                c_idx = _COLOR_TO_IDX.get(word)
                if c_idx is not None:
                    phc[int(pk), c_idx] = True
        part_has_color = phc

        if cfg.opt_presort:
            # Pre-build per-color sorted arrays and optionally prefix sums
            compound_all = (
                nation_key.astype(np.uint16) * 16 + year_key.astype(np.uint16)
            ).astype(np.uint16)

            color_data = {}
            for c_idx in range(N_COLORS):
                # Fast fancy-index: which rows have this colour?
                row_mask = phc[partkey, c_idx]
                if not row_mask.any():
                    color_data[c_idx] = None
                    continue

                amt_c = amount[row_mask]
                comp_c = compound_all[row_mask]

                # Sort by (nat_key, yr_key)
                sort_idx = np.argsort(comp_c, kind="stable")
                amt_c = amt_c[sort_idx]
                comp_c = comp_c[sort_idx]

                if cfg.opt_cumsum:
                    cs = np.zeros(len(amt_c) + 1, dtype=np.float64)
                    cs[1:] = np.cumsum(amt_c)
                    unique_comp, g_starts, g_counts = np.unique(
                        comp_c, return_index=True, return_counts=True
                    )
                    nat_k = (unique_comp >> 4).astype(np.uint8)
                    yr_k = (unique_comp & 0xF).astype(np.uint8)
                    color_data[c_idx] = {
                        "nat_key": nat_k,
                        "yr_key": yr_k,
                        "g_starts": g_starts,
                        "g_counts": g_counts,
                        "cs": cs,
                    }
                else:
                    # Presorted but no cumsum: store sorted arrays for per-query groupby
                    nat_k_full = (comp_c >> 4).astype(np.uint8)
                    yr_k_full = (comp_c & 0xF).astype(np.uint8)
                    color_data[c_idx] = {
                        "nat_key_full": nat_k_full,
                        "yr_key_full": yr_k_full,
                        "amt": amt_c,
                        "comp": comp_c,
                    }
    else:
        # Fallback: fetch part table for per-query lookup. part_has_color is needed
        # by _agg_from_raw regardless of how nation is encoded,  the partkey ->
        # color membership mapping is the same in either case. Gating this on
        # nation_key would leave part_has_color=None when encode_nation is off,
        # which makes every per-query call return [] silently.
        t_part = time.perf_counter()
        part_raw = con.execute(
            "SELECT p_partkey, p_name FROM part ORDER BY p_partkey"
        ).fetchnumpy()
        fetch_time += time.perf_counter() - t_part
        part_pkeys = part_raw["p_partkey"].astype(np.int32)
        part_names = part_raw["p_name"]
        max_pk = int(part_pkeys.max())
        phc = np.zeros((max_pk + 1, N_COLORS), dtype=bool)
        for pk, name in zip(part_pkeys, part_names):
            for word in str(name).split():
                c_idx = _COLOR_TO_IDX.get(word)
                if c_idx is not None:
                    phc[int(pk), c_idx] = True
        part_has_color = phc

    return FetchedArrays(
        amount=amount,
        partkey=partkey,
        nation_key=nation_key,
        year_key=year_key,
        nation_raw=nation_raw,
        year_raw=year_raw,
        part_has_color=part_has_color,
        color_data=color_data,
    ), fetch_time


def _agg_from_raw(
    arrays: FetchedArrays,
    color: str,
) -> Q9Result:
    """Fallback path: apply color filter and group-by without precomputed structures."""
    c_idx = _COLOR_TO_IDX.get(color, -1)

    if arrays.part_has_color is not None and c_idx >= 0:
        row_mask = arrays.part_has_color[arrays.partkey, c_idx]
    elif arrays.nation_raw is not None:
        # No part_has_color available - skip (shouldn't happen)
        return []
    else:
        return []

    if not row_mask.any():
        return []

    amt_m = arrays.amount[row_mask]

    if arrays.nation_key is not None and arrays.year_key is not None:
        nat_m = arrays.nation_key[row_mask]
        yr_m = arrays.year_key[row_mask]
        compound = nat_m.astype(np.uint16) * 16 + yr_m.astype(np.uint16)
        unique_comp, inverse = np.unique(compound, return_inverse=True)
        group_sums = np.bincount(inverse, weights=amt_m)
        result: Q9Result = []
        for comp, gs in zip(unique_comp, group_sums):
            nat_k = int(comp) >> 4
            yr_k = int(comp) & 0xF
            nation = _U8_TO_NAT.get(nat_k, "?")
            year = _U8_TO_YR.get(yr_k, 0)
            result.append((nation, year, round(float(gs), 4)))
    else:
        nm_m = arrays.nation_raw[row_mask]
        yr_m = arrays.year_raw[row_mask]
        unique_nm = sorted(set(str(s) for s in nm_m))
        result = []
        for nm in unique_nm:
            nm_mask = np.frompyfunc(lambda s: s == nm, 1, 1)(nm_m).astype(bool)
            for yr in np.unique(yr_m[nm_mask]):
                yr_mask = nm_mask & (yr_m == yr)
                result.append((nm, int(yr), round(float(amt_m[yr_mask].sum()), 4)))

    return sorted(result, key=lambda r: (r[0], -r[1]))


def _numpy_query(arrays: FetchedArrays, color: str) -> Q9Result:
    c_idx = _COLOR_TO_IDX.get(color, -1)

    # PATH 1: precomputed per-colour prefix sums
    if arrays.color_data is not None and c_idx >= 0:
        entry = arrays.color_data.get(c_idx)
        if entry is None:
            return []
        if "cs" in entry:
            cs = entry["cs"]
            nat_k_arr = entry["nat_key"]
            yr_k_arr = entry["yr_key"]
            g_starts = entry["g_starts"]
            g_counts = entry["g_counts"]
            result: Q9Result = []
            for nat_k, yr_k, gs, gc in zip(nat_k_arr, yr_k_arr, g_starts, g_counts):
                s = float(cs[gs + gc] - cs[gs])
                result.append(
                    (
                        _U8_TO_NAT.get(int(nat_k), "?"),
                        _U8_TO_YR.get(int(yr_k), 0),
                        round(s, 4),
                    )
                )
            return sorted(result, key=lambda r: (r[0], -r[1]))

        # PATH 2: presorted, no cumsum
        if "comp" in entry:
            comp_c = entry["comp"]
            amt_c = entry["amt"]
            unique_comp, g_starts, _ = np.unique(
                comp_c, return_index=True, return_counts=True
            )
            group_sums = np.add.reduceat(amt_c, g_starts)
            result = []
            for comp, gs in zip(unique_comp, group_sums):
                nat_k = int(comp) >> 4
                yr_k = int(comp) & 0xF
                result.append(
                    (
                        _U8_TO_NAT.get(nat_k, "?"),
                        _U8_TO_YR.get(yr_k, 0),
                        round(float(gs), 4),
                    )
                )
            return sorted(result, key=lambda r: (r[0], -r[1]))

    # PATH 3: fallback - mask + groupby
    return _agg_from_raw(arrays, color)


def _groupby_from_raw_sql(raw: dict, cfg: BenchConfig) -> Q9Result:
    """Group per-row SQL result (C combo: C in SQL, G in Python)."""
    amt = _build_amount(raw, cfg)

    if cfg.opt_encode_nation:
        nat_key = raw["n_key"].astype(np.uint8)
        yr_key = raw["yr_key"].astype(np.uint8)
        compound = nat_key.astype(np.uint16) * 16 + yr_key.astype(np.uint16)
        unique_comp, inverse = np.unique(compound, return_inverse=True)
        group_sums = np.bincount(inverse, weights=amt)
        result: Q9Result = []
        for comp, gs in zip(unique_comp, group_sums):
            nat_k = int(comp) >> 4
            yr_k = int(comp) & 0xF
            result.append(
                (
                    _U8_TO_NAT.get(nat_k, "?"),
                    _U8_TO_YR.get(yr_k, 0),
                    round(float(gs), 4),
                )
            )
    else:
        nm = raw["n_name"]
        yr = raw["o_year"].astype(np.int64)
        compound = np.zeros(len(amt), dtype=np.int64)
        # Encode nation string inline
        for i, s in enumerate(nm):
            n_k = _NAT_TO_U8.get(str(s), 255)
            compound[i] = n_k * 10000 + int(yr[i])
        unique_comp, inverse = np.unique(compound, return_inverse=True)
        group_sums = np.bincount(inverse, weights=amt)
        result = []
        for comp, gs in zip(unique_comp, group_sums):
            nat_k = int(comp) // 10000
            year = int(comp) % 10000
            result.append((_U8_TO_NAT.get(nat_k, "?"), year, round(float(gs), 4)))

    return sorted(result, key=lambda r: (r[0], -r[1]))


def run_one_query(
    con: duckdb.DuckDBPyConnection,
    cfg: BenchConfig,
    arrays: FetchedArrays | None,
    color: str,
) -> Q9Result:
    # CG: pure SQL
    if cfg.sql_handles("C") and cfg.sql_handles("G"):
        rows = con.execute(build_per_query_sql_cg(color)).fetchall()
        return [(str(r[0]), int(r[1]), round(float(r[2]), 4)) for r in rows]

    # C in SQL, G in Python
    if cfg.sql_handles("C"):
        raw = con.execute(build_per_query_sql_c(color, cfg)).fetchnumpy()
        if len(raw.get("amount" if cfg.opt_precompute else "l_extendedprice", [])) == 0:
            return []
        return _groupby_from_raw_sql(raw, cfg)

    # All in Python
    assert arrays is not None
    return _numpy_query(arrays, color)


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
    for color in params:
        results.append(run_one_query(con, cfg, arrays, color))
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
    reference: list[Q9Result],
    candidate: list[Q9Result],
    params: list[SweepParam],
    label: str,
    tol: float = 1.0,
) -> bool:
    mismatches = []
    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        ref_s = sorted(ref, key=lambda r: (r[0], r[1]))
        cand_s = sorted(cand, key=lambda r: (r[0], r[1]))
        if len(ref_s) != len(cand_s):
            mismatches.append((params[i], ref, cand, "length"))
            continue
        for rr, rc in zip(ref_s, cand_s):
            if rr[0] != rc[0] or rr[1] != rc[1] or abs(rr[2] - rc[2]) > tol:
                mismatches.append((params[i], ref, cand, "value"))
                break
    if mismatches:
        logger.warning(f"  [{label}] {len(mismatches)} mismatches:")
        for p, ref, cand, kind in mismatches[:3]:
            logger.warning(f"    color={p} ({kind})")
            logger.warning(f"      ref:  {ref[:3]}")
            logger.warning(f"      got:  {cand[:3]}")
        return False
    logger.info(f"  [{label}]  all results match reference")
    return True


class Q9Benchmark(QueryBenchmark):
    NAME = "Q9"
    ALL_STEPS = ("C", "G")
    N_APPLICABLE = True
    N_HELP = "Number of color keywords to sweep (default: 50; max unique: 92)"
    N_DEFAULT = 50

    def make_config(self, sql_steps: frozenset, args) -> BenchConfig:
        return BenchConfig(
            sql_steps=sql_steps,
            opt_precompute=args.opt_precompute,
            opt_encode_nation=args.opt_encode_nation,
            opt_encode_color=args.opt_encode_color,
            opt_presort=args.opt_presort,
            opt_cumsum=args.opt_cumsum,
        )

    def generate_params(self, n: int) -> list[SweepParam]:
        return generate_color_params(n)

    def single_params(self) -> list[SweepParam]:
        return single_query_params()

    def single_label(self) -> str:
        return f"single query (color='{SINGLE_COLOR}')"

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
            f"encode_color={'ON' if args.opt_encode_color else 'OFF'}  "
            f"presort={'ON' if args.opt_presort else 'OFF'}  "
            f"cumsum={'ON' if args.opt_cumsum else 'OFF'}"
        )

    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--opt_precompute",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Compute amount=ep*(1-disc)-supplycost*qty in SQL at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_nation",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Encode n_name and year as uint8 in SQL at fetch time (default: ON)",
        )
        parser.add_argument(
            "--opt_encode_color",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Parse part table once at fetch time to precompute per-color "
                "row membership; avoids O(n) string scan per query (default: ON)"
            ),
        )
        parser.add_argument(
            "--opt_presort",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Within each color's matching rows, sort by (nation_key, year_key) "
                "for contiguous group boundaries (default: ON)"
            ),
        )
        parser.add_argument(
            "--opt_cumsum",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Build prefix sums per (color, nation, year) group at fetch time "
                "for O(G) per-query aggregation (default: ON)"
            ),
        )

    def log_mode(self, args, params: list) -> str:
        return f"sweep  N={len(params)} color keywords"


def main() -> None:
    bench = Q9Benchmark()
    parser = make_base_parser(
        "Q9 predicate mix-and-match: benchmark every SQL/Python split"
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
        log_dir, f"q9_predmix_sf{args.sf}_{mode_tag}_{timestamp}.log"
    )
    setup_logging(log_filename)
    logger.info(f"Setting up database (SF={args.sf})")
    con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
    bench.run(con, args, log_filename)


if __name__ == "__main__":
    main()
