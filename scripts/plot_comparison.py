"""
Generate figures from a compare_engine_*.csv (DuckDB vs PostgreSQL vs MySQL
vs SQLite on TPC-H Q1/Q3/Q4/Q5/Q6/Q7/Q10/Q12/Q14/Q18).

Figures generated:
  compare_engine_time.pdf          median query time per engine, grouped by query, log scale
  compare_engine_speedup_{e}.pdf   speedup of DuckDB vs each other engine, per query
  compare_engine_heatmap.pdf       query, engine heatmap of speedup vs the fastest engine
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

# colourblind friendly, printable palette: https://personal.sron.nl/~pault
_PALETTE = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#BBBBBB",
]

ENGINE_ORDER = ["duckdb", "postgres", "mysql", "sqlite"]
ENGINE_LABEL = {
    "duckdb": "DuckDB",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "sqlite": "SQLite",
}
ENGINE_COLOR = dict(zip(ENGINE_ORDER, _PALETTE))
QUERY_ORDER = ["q1", "q3", "q4", "q5", "q6", "q7", "q10", "q12", "q14", "q18"]


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------


def find_latest_csv(log_dir: str = "logs") -> str | None:
    files = glob.glob(os.path.join(log_dir, "compare_engine_sf*_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


def _order(values, ref_order) -> list:
    present = set(values)
    return [v for v in ref_order if v in present] + sorted(present - set(ref_order))


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: grouped bar chart of median query time per engine
# ---------------------------------------------------------------------------


def plot_time_comparison(df: pd.DataFrame, out_dir: str) -> None:
    queries = _order(df["query"].unique(), QUERY_ORDER)
    engines = _order(df["engine"].unique(), ENGINE_ORDER)

    x = np.arange(len(queries))
    n_engines = len(engines)
    width = 0.8 / n_engines

    fig, ax = plt.subplots(figsize=(max(8, len(queries) * 1.1), 5))
    for i, engine in enumerate(engines):
        edata = df[df["engine"] == engine].set_index("query")
        medians = [
            edata.loc[q, "median_s"] if q in edata.index else np.nan for q in queries
        ]
        stds = [edata.loc[q, "std_s"] if q in edata.index else 0.0 for q in queries]
        offset = (i - (n_engines - 1) / 2) * width
        ax.bar(
            x + offset,
            medians,
            width=width * 0.92,
            yerr=stds,
            color=ENGINE_COLOR.get(engine, "#888888"),
            label=ENGINE_LABEL.get(engine, engine),
            capsize=2,
            error_kw={"elinewidth": 1.0},
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([q.upper() for q in queries])
    ax.set_xlabel("TPC-H query")
    ax.set_ylabel("Median query time (s, log scale)")
    ax.set_title("Query execution time by engine")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=n_engines, framealpha=0.8
    )
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.subplots_adjust(bottom=0.28)
    _save(fig, out_dir, "compare_engine_time.pdf")


# ---------------------------------------------------------------------------
# Figure 2: DuckDB speedup vs each other engine, per query
# ---------------------------------------------------------------------------


def plot_speedup_vs_duckdb(df: pd.DataFrame, out_dir: str) -> None:
    if "duckdb" not in df["engine"].unique():
        print("  [skip] speedup_vs_duckdb: no duckdb rows in this CSV")
        return

    duck = df[df["engine"] == "duckdb"].set_index("query")["median_s"]
    others = [e for e in _order(df["engine"].unique(), ENGINE_ORDER) if e != "duckdb"]

    for engine in others:
        edata = df[df["engine"] == engine].set_index("query")
        queries = _order(edata.index.unique(), QUERY_ORDER)
        queries = [q for q in queries if q in duck.index]
        if not queries:
            continue
        speedup = [edata.loc[q, "median_s"] / duck.loc[q] for q in queries]

        fig, ax = plt.subplots(figsize=(max(6, len(queries) * 0.9), 4))
        colors = [_PALETTE[2] if s >= 1.0 else _PALETTE[6] for s in speedup]
        bars = ax.bar(queries, speedup, color=colors, edgecolor="white")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_xticks(range(len(queries)))
        ax.set_xticklabels([q.upper() for q in queries])
        ax.set_ylabel(f"{ENGINE_LABEL[engine]} time / DuckDB time")
        ax.set_title(f"DuckDB speed-up vs {ENGINE_LABEL[engine]} (>1 = DuckDB faster)")
        for bar, s in zip(bars, speedup):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{s:.1f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        _save(fig, out_dir, f"compare_engine_speedup_{engine}.pdf")


# ---------------------------------------------------------------------------
# Figure 3: query x engine heatmap, speedup relative to the fastest engine
# ---------------------------------------------------------------------------


def plot_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    queries = _order(df["query"].unique(), QUERY_ORDER)
    engines = _order(df["engine"].unique(), ENGINE_ORDER)

    pivot = df.pivot_table(
        index="query", columns="engine", values="median_s", aggfunc="median"
    )
    pivot = pivot.reindex(index=queries, columns=engines)
    fastest = pivot.min(axis=1)
    relative = pivot.div(fastest, axis=0)  # 1.0 = fastest engine for that query

    fig, ax = plt.subplots(
        figsize=(max(5, len(engines) * 1.3), max(4, len(queries) * 0.5))
    )
    im = ax.imshow(relative.values, cmap="RdYlGn_r", aspect="auto", vmin=1.0)
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels([ENGINE_LABEL.get(e, e) for e in engines])
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels([q.upper() for q in queries])
    for i in range(len(queries)):
        for j in range(len(engines)):
            v = relative.values[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.1f}x",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                    if v > relative.values[~np.isnan(relative.values)].mean()
                    else "black",
                )
    ax.set_title("Slowdown relative to the fastest engine per query")
    fig.colorbar(im, ax=ax, label="x slower than fastest")
    fig.tight_layout()
    _save(fig, out_dir, "compare_engine_heatmap.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot compare_engine.py TPC-H results (DuckDB vs Postgres vs MySQL vs SQLite).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        default=None,
        help="CSV to load. Defaults to latest compare_engine_*.csv in --log_dir.",
    )
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args(argv)

    csv_path = args.file or find_latest_csv(args.log_dir)
    if not csv_path or not os.path.exists(csv_path):
        print("No CSV found. Pass --file or run benchmarks/compare_engine.py first.")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(
        f"  {len(df)} rows  |  engines: {df['engine'].unique().tolist()}  |  queries: {df['query'].unique().tolist()}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing figures to: {args.output_dir}/\n")

    plot_time_comparison(df, args.output_dir)
    plot_speedup_vs_duckdb(df, args.output_dir)
    plot_heatmap(df, args.output_dir)


if __name__ == "__main__":
    main()
