"""
Benchmarking infrastructure for parameter sweep experiments.

Each query module implements `QueryBenchmark`, and this module handles all timing, reporting, logging, and argument parsing boilerplate.

Notes:
    Multiple queries can be sequentially ran at once, default is all.

    When a speedup is 1.0x +-0.05 and variation is >10% it's counted as noise rather than a definitive win.

    cProfile can be used with --profile, but doesn't capture Numpy C internals. py-spy does:
        `py-spy record -o logs/flamegraph.svg -- python3 -m benchmarks.q6_predicate_sweep`
"""

from __future__ import annotations

import argparse
import cProfile
import datetime
import importlib
import logging
import os
import pstats
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from scripts.setup_db import setup_db
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def _safe_add_args(parser: argparse.ArgumentParser):
    """Temporarily patch parser.add_argument to silently skip duplicate flags."""
    orig = parser.add_argument

    def _safe(*a, **kw):
        try:
            orig(*a, **kw)
        except argparse.ArgumentError:
            pass

    parser.add_argument = _safe
    try:
        yield
    finally:
        parser.add_argument = orig


# ---------------------------------------------------------------------------
# Noise and variation thresholds
# ---------------------------------------------------------------------------

NOISE_RATIO_BAND: float = 0.05  # +-5% SQL
MIN_REPEATS_FOR_NOISE: int = 3
CV_THRESHOLD: float = 0.10  # stddev/mean, 10%, means unstable timing


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Timing result for one combo over one full sweep."""

    key: str
    sql_steps_str: str
    python_steps_str: str
    values: list  # query results (type depends on query)
    fetch_time: float  # seconds spent in bulk SQL fetch(es)
    logic_time: float  # seconds spent in per-query logic
    total_time: float  # fetch_time + logic_time


@dataclass
class RepeatStats:
    """Total stats across multiple repeats of the same combo."""

    fetch_median: float
    fetch_std: float
    logic_median: float
    logic_std: float
    total_median: float
    total_std: float
    total_cv: float  # coefficient of variation = stddev/mean
    n_repeats: int


def compute_stats(repeats: list[SweepResult]) -> RepeatStats:
    fetch_arr = np.array([r.fetch_time for r in repeats])
    logic_arr = np.array([r.logic_time for r in repeats])
    total_arr = np.array([r.total_time for r in repeats])

    t_med = float(np.median(total_arr))
    t_std = float(np.std(total_arr, ddof=1)) if len(total_arr) > 1 else 0.0
    cv = t_std / t_med if t_med > 0 else 0.0

    return RepeatStats(
        fetch_median=float(np.median(fetch_arr)),
        fetch_std=float(np.std(fetch_arr, ddof=1)) if len(fetch_arr) > 1 else 0.0,
        logic_median=float(np.median(logic_arr)),
        logic_std=float(np.std(logic_arr, ddof=1)) if len(logic_arr) > 1 else 0.0,
        total_median=t_med,
        total_std=t_std,
        total_cv=cv,
        n_repeats=len(repeats),
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_result(
    stats: RepeatStats,
    sql_total_median: float,
    noise_band: float = NOISE_RATIO_BAND,
    cv_threshold: float = CV_THRESHOLD,
) -> tuple[str, str]:
    """
    Returns (winner_label, detail_note).
    winner_label: one of "PYTHON wins", "  SQL wins  ", "   NOISE    "
    detail_note:  empty string or a brief explanation appended to the row
    """
    if sql_total_median <= 0:
        return "  SQL wins  ", ""

    ratio = stats.total_median / sql_total_median
    is_within_noise_band = abs(ratio - 1.0) < noise_band
    is_high_cv = (
        stats.n_repeats >= MIN_REPEATS_FOR_NOISE and stats.total_cv > cv_threshold
    )

    if is_within_noise_band and is_high_cv:
        return "~ NOISE ~    ", f"cv={stats.total_cv:.0%}"

    if stats.total_median < sql_total_median:
        if is_within_noise_band:
            return (
                "   NOISE   ",
                f"delta={abs(1 - ratio):.1%}<{noise_band:.0%} threshold",
            )
        return "PYTHON wins", ""

    return "  SQL wins  ", ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_filename: str) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def log_results_table(
    query_name: str,
    all_stats: dict[str, tuple[Any, RepeatStats]],  # key -> (cfg, stats)
    sql_total_median: float,
    n: int,
    mode_label: str,
    all_steps: tuple[str, ...],
    noise_band: float = NOISE_RATIO_BAND,
    cv_threshold: float = CV_THRESHOLD,
) -> None:
    width = 104
    logger.info(f"\n{'=' * width}")
    logger.info(f"{query_name} PREDICATE MIX RESULTS  ({mode_label})")
    logger.info(f"{'=' * width}")
    logger.info(
        f"  {'SQL steps':<12} {'Python steps':<12} "
        f"{'Fetch':>8} {'+-':>6} {'Logic':>8} {'+-':>6} {'Total':>8} {'+-':>6} "
        f"{'vs SQL':>7}  {'Result':<16} {'ms/query':>10}"
    )
    logger.info(f"  {'-' * (width - 2)}")

    sorted_items = sorted(all_stats.items(), key=lambda kv: kv[1][1].total_median)

    for _, (cfg, stats) in sorted_items:
        sql_str = cfg.key if cfg.sql_steps else "(none)"
        py_str = "".join(s for s in all_steps if cfg.python_handles(s)) or "(none)"
        ratio = stats.total_median / sql_total_median if sql_total_median > 0 else 0.0
        winner, note = classify_result(
            stats, sql_total_median, noise_band, cv_threshold
        )
        ms_per = stats.logic_median / n * 1000 if n > 0 else 0.0
        result_str = f"{winner} {note}".rstrip()

        logger.info(
            f"  {sql_str:<12} {py_str:<12} "
            f"{stats.fetch_median:>8.3f} {stats.fetch_std:>6.3f} "
            f"{stats.logic_median:>8.3f} {stats.logic_std:>6.3f} "
            f"{stats.total_median:>8.3f} {stats.total_std:>6.3f} "
            f"{ratio:>7.2f}x  {result_str:<16} {ms_per:>10.3f}ms"
        )

    sql_key = next(
        k for k, (cfg, _) in all_stats.items() if cfg.sql_steps == frozenset(all_steps)
    )
    sql_std = all_stats[sql_key][1].total_std
    sql_cv = all_stats[sql_key][1].total_cv
    logger.info(
        f"\n  Pure SQL baseline: {sql_total_median:.4f}s "
        f"(\u00b1{sql_std:.4f}s, cv={sql_cv:.1%})"
    )

    if n > 0:
        logger.info(f"  ({sql_total_median / n * 1000:.3f}ms/query over {n} queries)")
    logger.info(
        f"\n  Noise threshold: ±{noise_band:.0%} of baseline AND cv>{cv_threshold:.0%}  "
        f"-> result flagged as NOISE rather than a meaningful win/loss."
    )
    logger.info(
        "  Flamegraph (includes numpy C internals): "
        "py-spy record -o logs/flamegraph.svg -- python3 -m benchmarks.<query>"
    )


# ---------------------------------------------------------------------------
# Query Benchmarks
# ---------------------------------------------------------------------------


class QueryBenchmark(ABC):
    """
    Baseclass for per-query benchmark.
    Subclasses have the query-specific logic, and this provides the shared run/report/validate loop.
    """

    NAME: str = "Q?"
    ALL_STEPS: tuple[str, ...] = ()
    N_APPLICABLE: bool = True
    N_HELP: str = "Number of sweep parameter variations (default: 200)"
    N_DEFAULT: int = 200

    @abstractmethod
    def make_config(self, sql_steps: frozenset, args: argparse.Namespace):
        """Construct a query-specific BenchConfig from parsed args."""

    @abstractmethod
    def generate_params(self, n: int) -> list:
        """Return a list of N sweep parameter values/tuples."""

    @abstractmethod
    def single_params(self) -> list:
        """Return a list with exactly one (fixed) parameter set."""

    @abstractmethod
    def single_label(self) -> str:
        """Description of the single-query parameters."""

    @abstractmethod
    def is_valid_combo(self, cfg) -> tuple[bool, str]:
        """Return (valid, reason) for a given BenchConfig."""

    @abstractmethod
    def run_sweep(self, con, cfg, params) -> SweepResult:
        """Run one combo over all params. Return a SweepResult."""

    @abstractmethod
    def validate(
        self, reference: list, candidate: list, params: list, label: str
    ) -> bool:
        """Return True if candidate matches reference within tolerance."""

    @abstractmethod
    def opt_flags_str(self, args: argparse.Namespace) -> str:
        """Summary of numpy optimisation flags for logging."""

    @abstractmethod
    def add_query_args(self, parser: argparse.ArgumentParser) -> None:
        """Register query-specific arguments on the shared parser."""

    @abstractmethod
    def log_mode(self, args: argparse.Namespace, params: list) -> str:
        """Return a mode label string for the results table header."""

    def all_combo_keys(self) -> list[str]:
        """Return every powerset-combo key for this query's steps, sorted by (length, key)."""
        return sorted(
            [
                "".join(s for s in self.ALL_STEPS if s in subset)
                for subset in (
                    frozenset(c)
                    for length in range(len(self.ALL_STEPS) + 1)
                    for c in combinations(self.ALL_STEPS, length)
                )
            ],
            key=lambda k: (len(k), k),
        )

    def parse_combo_keys(self, raw: list[str]) -> list[str]:
        """Validate and normalise user-supplied combo keys."""
        normalised = []
        for r in raw:
            r = r.upper().strip()
            invalid = set(r) - set(self.ALL_STEPS)
            if invalid:
                raise ValueError(
                    f"Unknown step(s) {invalid} in combo '{r}'. "
                    f"Valid: {set(self.ALL_STEPS)}"
                )
            key = "".join(s for s in self.ALL_STEPS if s in set(r))
            normalised.append(key)
        return normalised

    def run(self, con, args: argparse.Namespace, log_filename: str) -> None:
        """
        Full benchmark run: resolve combos, validate, time repeats, report.
        """

        all_combo_keys = self.all_combo_keys()

        # Resolve combos
        if args.combos is None:
            combo_keys = all_combo_keys
        else:
            raw = [("" if c.lower() in ("none", "") else c) for c in args.combos]
            combo_keys = self.parse_combo_keys(raw)

        configs = []
        skipped = []
        for key in combo_keys:
            cfg = self.make_config(frozenset(key), args)
            valid, reason = self.is_valid_combo(cfg)
            if valid:
                configs.append(cfg)
            else:
                skipped.append((key or "(none)", reason))

        # Sweep params
        n = args.n if self.N_APPLICABLE else 1
        if args.single:
            params = self.single_params()
            mode_label = self.single_label()
        else:
            params = self.generate_params(n)
            mode_label = self.log_mode(args, params)

        # Header logging
        logger.info(f"Mode: {mode_label}")
        logger.info(f"NumPy opts: {self.opt_flags_str(args)}")
        if not self.N_APPLICABLE and not args.single:
            logger.info(
                f"  [--n ignored for {self.NAME}: sweep params are fixed "
                f"({len(params)} total)]"
            )

        if skipped:
            logger.info(f"\nSkipping {len(skipped)} invalid combo(s):")
            for key, reason in skipped:
                logger.info(f"  sql={key}: {reason}")

        logger.info(
            f"\nRunning {len(configs)} combo(s): "
            + ", ".join(f"sql={cfg.key or '(none)'}" for cfg in configs)
        )

        # SQL reference
        sql_cfg = self.make_config(frozenset(self.ALL_STEPS), args)
        logger.info("\nBuilding SQL reference...")
        ref_result = self.run_sweep(con, sql_cfg, params)
        reference_values = ref_result.values

        # Validation
        logger.info("\nValidating all combos against SQL reference...")
        for cfg in configs:
            if cfg.sql_steps == frozenset(self.ALL_STEPS):
                continue
            result = self.run_sweep(con, cfg, params)
            self.validate(
                reference_values, result.values, params, f"sql={cfg.key or '(none)'}"
            )

        # ------------------------------------------------------------------
        # Repeats
        # ------------------------------------------------------------------
        logger.info(
            f"\nRunning {args.repeats} repeat(s) of each combo (N={len(params)})..."
        )

        def _do_timing():
            nonlocal sql_stats, all_stats

            sql_timing: list[SweepResult] = []
            for _ in range(args.repeats):
                sql_timing.append(self.run_sweep(con, sql_cfg, params))
            sql_stats = compute_stats(sql_timing)

            all_stats[sql_cfg.key] = (sql_cfg, sql_stats)

            for cfg in configs:
                if cfg.sql_steps == frozenset(self.ALL_STEPS):
                    continue
                label = cfg.key or "(none)"
                repeat_results = []
                for rep in range(args.repeats):
                    r = self.run_sweep(con, cfg, params)
                    repeat_results.append(r)
                    logger.info(
                        f"  sql={label:<8} rep {rep + 1}/{args.repeats}: "
                        f"fetch={r.fetch_time:.3f}s  logic={r.logic_time:.3f}s  "
                        f"total={r.total_time:.3f}s"
                    )
                stats = compute_stats(repeat_results)
                all_stats[label] = (cfg, stats)

        sql_stats: RepeatStats | None = None
        all_stats: dict[str, tuple[Any, RepeatStats]] = {}

        if getattr(args, "profile", False):
            prof_path = log_filename.replace(".log", ".prof")
            profiler = cProfile.Profile()
            profiler.enable()
            _do_timing()
            profiler.disable()
            profiler.dump_stats(prof_path)
            with open(os.devnull, "w") as devnull:
                pstats.Stats(profiler, stream=devnull).sort_stats("cumulative")
            logger.info(f"\ncProfile data saved to: {prof_path}")
            logger.info("  View with: python3 -m snakeviz " + prof_path)
            logger.info(
                "  For numpy C internals use py-spy instead:\n"
                f"    py-spy record -o logs/flamegraph.svg -- "
                f"python3 -m benchmarks.{self.NAME.lower()}"
            )
        else:
            _do_timing()

        # Report
        log_results_table(
            query_name=self.NAME,
            all_stats=all_stats,
            sql_total_median=sql_stats.total_median,
            n=len(params),
            mode_label=mode_label,
            all_steps=self.ALL_STEPS,
            noise_band=getattr(args, "noise_band", NOISE_RATIO_BAND),
            cv_threshold=getattr(args, "cv_threshold", CV_THRESHOLD),
        )
        logger.info(f"\nLog saved to: {log_filename}")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------


def make_base_parser(description: str) -> argparse.ArgumentParser:
    """Build an ArgumentParser with all shared flags pre-registered."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sf", type=int, default=1)
    parser.add_argument("--memory_limit", type=str, default="4GB")
    parser.add_argument(
        "--num_params",
        "--n",
        dest="n",
        type=int,
        default=None,  # None = use N_DEFAULT from QueryBenchmark
        metavar="N",
        help="Number of sweep parameter variations. Ignored when --single is set, "
        "and ignored with a logged note when not applicable to the query.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timing repeats for stable medians (default: 5)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        default=False,
        help="Run a single fixed query instead of a parameter sweep.",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        default=None,
        metavar="COMBO",
        help="SQL-step combos to run. Default: all valid. Use 'none' for pure-Python.",
    )
    parser.add_argument(
        "--noise_band",
        type=float,
        default=NOISE_RATIO_BAND,
        help=f"Speedup ratio band around 1.0 considered noise (default: {NOISE_RATIO_BAND})",
    )
    parser.add_argument(
        "--cv_threshold",
        type=float,
        default=CV_THRESHOLD,
        help=f"Coefficient-of-variation threshold for noise detection (default: {CV_THRESHOLD})",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Write a cProfile .prof file for the timed repeats (view with snakeviz).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point for multi-query
# ---------------------------------------------------------------------------


def run_multi(benchmark_classes, argv=None) -> None:
    """
    Run multiple QueryBenchmark instances sequentially.
    Each query loads the DB, runs to completion, then releases it.
    One DB loaded at a time.
    """

    parser = make_base_parser(
        "Run multiple Q predicate-sweep benchmarks: "
        + ", ".join(b.NAME for b in benchmark_classes)
    )

    # Add query-specific args from all benchmarks, skipping duplicates across queries.
    with _safe_add_args(parser):
        for cls in benchmark_classes:
            cls().add_query_args(parser)

    args = parser.parse_args(argv)
    if args.n is None:
        args.n = max(b.N_DEFAULT for b in benchmark_classes)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    queries_str = "".join(b.NAME.lower() for b in benchmark_classes)
    mode_tag = "single" if args.single else f"n{args.n}"
    log_filename = os.path.join(
        log_dir,
        f"multi_{queries_str}_sf{args.sf}_{mode_tag}_{timestamp}.log",
    )
    setup_logging(log_filename)

    for cls in benchmark_classes:
        bench = cls()

        logger.info(f"\n{'#' * 60}")
        logger.info(f"# {bench.NAME} predicate sweep")
        logger.info(f"{'#' * 60}")
        logger.info(f"Setting up database (SF={args.sf})")

        con = setup_db(sf=args.sf, memory_limit=args.memory_limit)
        try:
            bench.run(con, args, log_filename)
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Registry of all known query modules: name -> (module, class)
QUERY_REGISTRY = {
    "q1": ("benchmarks.q1", "Q1Benchmark"),
    "q3": ("benchmarks.q3", "Q3Benchmark"),
    "q4": ("benchmarks.q4", "Q4Benchmark"),
    "q6": ("benchmarks.q6", "Q6Benchmark"),
    "q7": ("benchmarks.q7", "Q7Benchmark"),
    "q9": ("benchmarks.q9", "Q9Benchmark"),
    "q10": ("benchmarks.q10", "Q10Benchmark"),
    "q11": ("benchmarks.q11", "Q11Benchmark"),
    "q14": ("benchmarks.q14", "Q14Benchmark"),
    "q18": ("benchmarks.q18", "Q18Benchmark"),
}


def main(argv=None) -> None:

    if argv is None:
        argv = sys.argv[1:]

    known = sorted(QUERY_REGISTRY)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("queries", nargs="*")
    pre_args, remaining = pre.parse_known_args(argv)

    query_names = pre_args.queries
    if not query_names:
        print(
            "Usage: python3 -m benchmarks.benchmark_sweep <query> [<query> ...] [options]"
        )
        print(f"Known queries: {', '.join(known)}")
        sys.exit(1)

    unknown = [q for q in query_names if q.lower() not in QUERY_REGISTRY]
    if unknown:
        print(f"Unknown query name(s): {', '.join(unknown)}")
        print(f"Known queries: {', '.join(known)}")
        sys.exit(1)

    benchmark_classes = []
    for name in query_names:
        mod_path, cls_name = QUERY_REGISTRY[name.lower()]
        mod = importlib.import_module(mod_path)
        benchmark_classes.append(getattr(mod, cls_name))

    run_multi(benchmark_classes, argv=remaining)


if __name__ == "__main__":
    main()
