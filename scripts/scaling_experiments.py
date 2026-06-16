"""
Three data-collection experiments for thesis analysis:

  n_scaling: vary batch size N at fixed SF, all valid combos.
                 Reveals the crossover point where a combo starts beating SQL.

  sf_scaling: vary SF at fixed N, all valid combos.
                 Checks whether wins hold at larger data volumes.

  flag_ablation: toggle each numpy opt flag OFF one at a time (plus all-ON and
                 all-OFF baselines) at fixed SF and N.
                 Attributes speedup to individual optimisations.

Output: a single timestamped CSV in logs/ for figure generation.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import os
import sys
from benchmarks.benchmark_sweep import (
    QueryBenchmark,
    QUERY_REGISTRY,
    _safe_add_args,
    compute_stats,
    make_base_parser,
)
from scripts.setup_db import setup_db

# Best Python combo per query, from experiment data
# Used by flag_ablation and sf_scaling to skip all but the best combos
BEST_COMBOS: dict[str, str] = {
    "Q1": "GD",
    "Q3": "(none)",
    "Q4": "E",
    "Q6": "SQ",
    "Q7": "(none)",
    "Q9": "(none)",
    "Q10": "RA",
    "Q11": "(none)",
    "Q14": "(none)",
    "Q18": "(none)",
}

CSV_FIELDS = [
    "experiment",
    "query",
    "sf",
    "n_requested",
    "n_actual",
    "combo",
    "flags",
    "fetch_median",
    "fetch_std",
    "logic_median",
    "logic_std",
    "total_median",
    "total_std",
    "total_cv",
    "n_repeats",
    "sql_baseline_median",
    "speedup",
    "valid",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_bench(name: str) -> QueryBenchmark:
    mod_path, cls_name = QUERY_REGISTRY[name.lower()]
    return getattr(importlib.import_module(mod_path), cls_name)()


def bench_args(
    bench: QueryBenchmark, sf: float, n: int, **overrides
) -> argparse.Namespace:
    """Parse through the bench's own argparser to get correct flag defaults, then apply overrides."""
    parser = make_base_parser("")
    with _safe_add_args(parser):
        bench.add_query_args(parser)
    args = parser.parse_args([])
    args.sf = sf
    args.n = n
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def opt_flags(bench: QueryBenchmark) -> list[str]:
    """Return dest names of all --opt_* BooleanOptionalAction flags for this bench."""
    parser = make_base_parser("")
    with _safe_add_args(parser):
        bench.add_query_args(parser)
    return [
        a.dest
        for a in parser._actions
        if isinstance(a, argparse.BooleanOptionalAction) and a.dest.startswith("opt_")
    ]


def restrict_to_best(bench: QueryBenchmark, cfgs: list) -> list:
    """Keep only the hardcoded best combo + the all-SQL baseline for this bench.

    Falls back to the full cfg list (with a warning) if the bench isn't in
    BEST_COMBOS (do NOT do this, it slows down by magnitudes)
    """
    sql_steps = frozenset(bench.ALL_STEPS)
    best = BEST_COMBOS.get(bench.NAME)
    if best is None:
        print(
            f"  [warn] no entry in BEST_COMBOS for {bench.NAME}; "
            "running every valid combo"
        )
        return cfgs
    restricted = [c for c in cfgs if c.key == best or c.sql_steps == sql_steps]
    # Sanity check: the named best combo must actually exist in cfgs
    if not any(c.key == best for c in cfgs):
        raise RuntimeError(
            f"BEST_COMBOS[{bench.NAME!r}] = {best!r} but no valid cfg has that "
            "key. Find best combo again with n_scaling experiment."
        )
    return restricted


def valid_configs(bench: QueryBenchmark, args: argparse.Namespace) -> list:
    """Return all valid BenchConfig objects for this bench."""
    cfgs = []
    for key in bench.all_combo_keys():
        cfg = bench.make_config(frozenset(key), args)
        if bench.is_valid_combo(cfg)[0]:
            cfgs.append(cfg)
    return cfgs


def time_combos(
    con,
    bench: QueryBenchmark,
    cfgs: list,
    params: list,
    repeats: int,
    sf: float,
    n_requested: int,
    experiment: str,
    flags_str: str,
) -> tuple[list[dict], float]:
    """Time all cfgs; return (rows, sql_baseline_median).

    Each non-SQL combo's result is validated against the SQL baseline. A
    combo whose flag combination silently returns wrong output is invalidated.
    """
    sql_steps = frozenset(bench.ALL_STEPS)

    # SQL baseline first (always timed, even if in cfgs). Reuse the last run's
    # result list as the reference for validating every other combo.
    sql_cfg = next(c for c in cfgs if c.sql_steps == sql_steps)
    sql_runs = [bench.run_sweep(con, sql_cfg, params) for _ in range(repeats)]
    sql_stats = compute_stats(sql_runs)
    sql_baseline = sql_stats.total_median
    sql_reference = sql_runs[-1].values

    rows = []
    for cfg in cfgs:
        if cfg.sql_steps == sql_steps:
            stats = sql_stats
            valid = True
        else:
            runs = [bench.run_sweep(con, cfg, params) for _ in range(repeats)]
            stats = compute_stats(runs)
            label = f"{cfg.key or '(none)'} | {flags_str}"
            try:
                valid = bool(
                    bench.validate(sql_reference, runs[-1].values, params, label)
                )
            except Exception as e:
                valid = False
                print(f"    [VALIDATION ERROR] {label}: {e}")
            if not valid:
                print(
                    f"    [INVALID] {bench.NAME} {label} — "
                    "results differ from SQL; speedup is not meaningful"
                )
        speedup = sql_baseline / stats.total_median if stats.total_median > 0 else 0.0
        rows.append(
            {
                "experiment": experiment,
                "query": bench.NAME,
                "sf": sf,
                "n_requested": n_requested,
                "n_actual": len(params),
                "combo": cfg.key or "(none)",
                "flags": flags_str,
                "fetch_median": round(stats.fetch_median, 5),
                "fetch_std": round(stats.fetch_std, 5),
                "logic_median": round(stats.logic_median, 5),
                "logic_std": round(stats.logic_std, 5),
                "total_median": round(stats.total_median, 5),
                "total_std": round(stats.total_std, 5),
                "total_cv": round(stats.total_cv, 4),
                "n_repeats": stats.n_repeats,
                "sql_baseline_median": round(sql_baseline, 5),
                "speedup": round(speedup, 4),
                "valid": valid,
            }
        )
    return rows, sql_baseline


# ---------------------------------------------------------------------------
# Experiment 1: N scaling
# ---------------------------------------------------------------------------


def run_n_scaling(
    query_names: list[str],
    sf: float,
    n_values: list[int],
    repeats: int,
    memory_limit: str = "4GB",
) -> list[dict]:
    rows = []
    for qname in query_names:
        bench = load_bench(qname)
        if not bench.N_APPLICABLE:
            print(f"  {bench.NAME}: N not applicable, skipping n_scaling")
            continue

        print(f"\n[n_scaling] {bench.NAME}  SF={sf}  N={n_values}  repeats={repeats}")
        flags_str = ",".join(f"{f}=on" for f in opt_flags(bench))
        args = bench_args(bench, sf, max(n_values))

        con = setup_db(sf=sf, memory_limit=memory_limit)
        try:
            for n in n_values:
                args.n = n
                params = bench.generate_params(n)
                cfgs = valid_configs(bench, args)
                batch_rows, _ = time_combos(
                    con, bench, cfgs, params, repeats, sf, n, "n_scaling", flags_str
                )
                rows.extend(batch_rows)
                print(f"  n={n:>4} -> {len(params):>4} params, {len(cfgs)} combos")
        finally:
            con.close()
    return rows


# ---------------------------------------------------------------------------
# Experiment 2: SF scaling
# ---------------------------------------------------------------------------


def run_sf_scaling(
    query_names: list[str],
    sf_values: list[float],
    n: int,
    repeats: int,
    only_best: bool = True,
    memory_limit: str = "4GB",
) -> list[dict]:
    """
    SF scaling for each query.

    If only_best is True (default): restrict to the combo listed in
    BEST_COMBOS plus the all-SQL baseline.
    """
    rows = []
    for qname in query_names:
        bench = load_bench(qname)
        best = BEST_COMBOS.get(bench.NAME, "(unknown)")
        print(
            f"\n[sf_scaling] {bench.NAME}  N={n}  SF={sf_values}  repeats={repeats}  "
            f"best_combo={best if only_best else 'ALL'}"
        )
        flags_str = ",".join(f"{f}=on" for f in opt_flags(bench))

        for sf in sf_values:
            args = bench_args(bench, sf, n)
            params = (
                bench.generate_params(n)
                if bench.N_APPLICABLE
                else bench.single_params()
            )
            con = setup_db(sf=sf, memory_limit=memory_limit)
            try:
                cfgs = valid_configs(bench, args)
                if only_best:
                    cfgs = restrict_to_best(bench, cfgs)
                batch_rows, _ = time_combos(
                    con, bench, cfgs, params, repeats, sf, n, "sf_scaling", flags_str
                )
                rows.extend(batch_rows)
                print(f"  sf={sf}  {len(params)} params, {len(cfgs)} combos")
            finally:
                con.close()
    return rows


# ---------------------------------------------------------------------------
# Experiment 3: Flag ablation
# ---------------------------------------------------------------------------


def run_flag_ablation(
    query_names: list[str],
    sf: float,
    n: int,
    repeats: int,
    only_best: bool = True,
    memory_limit: str = "4GB",
) -> list[dict]:
    """
    Flag ablation for each query.

    If only_best is True (default): restrict each flag variant to the combo
    listed in BEST_COMBOS (derived from prior n_scaling data) plus the all-SQL
    baseline.
    """
    rows = []
    for qname in query_names:
        bench = load_bench(qname)
        flags = opt_flags(bench)
        if not flags:
            print(f"  {bench.NAME}: no opt flags, skipping flag_ablation")
            continue

        best = BEST_COMBOS.get(bench.NAME, "(unknown)")
        print(
            f"\n[flag_ablation] {bench.NAME}  SF={sf}  N={n}  flags={flags}  "
            f"repeats={repeats}  best_combo={best if only_best else 'ALL'}"
        )

        args = bench_args(bench, sf, n)
        params = (
            bench.generate_params(n)
            if bench.N_APPLICABLE
            else bench.generate_params(bench.N_DEFAULT)
        )

        # Variants: all-ON, each individually OFF, all-OFF.
        variants: list[dict[str, bool]] = []
        variants.append({f: True for f in flags})  # all ON
        for flag in flags:
            v = {f: True for f in flags}
            v[flag] = False
            variants.append(v)  # one OFF
        variants.append({f: False for f in flags})  # all OFF

        con = setup_db(sf=sf, memory_limit=memory_limit)
        try:
            for flag_combo in variants:
                flags_str = ",".join(
                    f"{f}={'on' if v else 'off'}" for f, v in flag_combo.items()
                )
                for f, v in flag_combo.items():
                    setattr(args, f, v)
                cfgs = valid_configs(bench, args)
                if only_best:
                    cfgs = restrict_to_best(bench, cfgs)
                batch_rows, _ = time_combos(
                    con, bench, cfgs, params, repeats, sf, n, "flag_ablation", flags_str
                )
                rows.extend(batch_rows)
                print(f"  {flags_str}")
        finally:
            con.close()
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="N-scaling, SF-scaling, and flag-ablation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "queries",
        nargs="*",
        default=list(QUERY_REGISTRY),
        help=f"Queries to run (default: all). Known: {', '.join(QUERY_REGISTRY)}",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["n_scaling", "sf_scaling", "flag_ablation"],
        default=["n_scaling", "sf_scaling", "flag_ablation"],
    )
    parser.add_argument(
        "--sf",
        type=float,
        default=1.0,
        help="Scale factor for n_scaling and flag_ablation",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Batch size N for sf_scaling and flag_ablation",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1024],
        help="N values for n_scaling",
    )
    parser.add_argument(
        "--sf_values",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
        help="SF values for sf_scaling",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Timing repeats per data point"
    )
    parser.add_argument(
        "--all_combos",
        action="store_true",
        help=(
            "In flag_ablation and sf_scaling, run every valid combo (slow). "
            "Default is to run only the BEST_COMBOS-listed combo plus the "
            "all-SQL baseline, since plot_flag_ablation and plot_sf_scaling "
            "only use the best combo per query."
        ),
    )
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="4GB",
        help="DuckDB memory limit (default: 4GB). Increase for SF > 4.",
    )
    args = parser.parse_args(argv)

    unknown = [q for q in args.queries if q.lower() not in QUERY_REGISTRY]
    if unknown:
        print(f"Unknown queries: {unknown}")
        print(f"Known: {', '.join(QUERY_REGISTRY)}")
        sys.exit(1)

    all_rows: list[dict] = []

    if "n_scaling" in args.experiments:
        print("\n=== Experiment 1: N Scaling ===")
        all_rows.extend(
            run_n_scaling(
                args.queries, args.sf, args.n_values, args.repeats, args.memory_limit
            )
        )

    if "sf_scaling" in args.experiments:
        print("\n=== Experiment 2: SF Scaling ===")
        all_rows.extend(
            run_sf_scaling(
                args.queries,
                args.sf_values,
                args.n,
                args.repeats,
                only_best=not args.all_combos,
                memory_limit=args.memory_limit,
            )
        )

    if "flag_ablation" in args.experiments:
        print("\n=== Experiment 3: Flag Ablation ===")
        all_rows.extend(
            run_flag_ablation(
                args.queries,
                args.sf,
                args.n,
                args.repeats,
                only_best=not args.all_combos,
                memory_limit=args.memory_limit,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"scaling_experiments_{timestamp}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
