"""
Generate figures from a scaling_experiments CSV

Figures generated:
  n_scaling_speedup_{q}.pdf, speedup vs N, all combos, error bars
  n_scaling_decomp_{q}.pdf, fetch / logic / SQL time vs N, error bars
  sf_scaling_{q}.pdf, speedup vs SF, all combos, error bars
  flag_ablation.pdf, speedup under each flag knockout, error bars
  crossover_summary.pdf, best speedup + crossover N per query
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

# colourblind friendly, printable palette:
# https://personal.sron.nl/~pault
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


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------


def find_latest_csv(log_dir: str = "logs") -> str | None:
    files = glob.glob(os.path.join(log_dir, "scaling_experiments_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


def flags_label(flags_str: str) -> str:
    """'opt_presort=on,opt_precompute=off' -> '−precompute'."""
    if not flags_str or "=off" not in str(flags_str):
        return "all ON"
    parts = dict(p.split("=") for p in str(flags_str).split(",") if "=" in p)
    off = [k.replace("opt_", "") for k, v in parts.items() if v == "off"]
    return "all OFF" if len(off) == len(parts) else "−" + "/".join(off)


def is_all_on(flags_str) -> bool:
    return "=off" not in str(flags_str)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose Python output matched the SQL baseline."""
    if "valid" not in df.columns:
        return pd.Series(True, index=df.index)
    s = df["valid"]
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(("true", "1"))


def combo_colors(combos: list[str]) -> dict[str, str]:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(combos)}


def sort_combos(combos: list[str]) -> list[str]:
    """SQL-heavy first, pure Python last."""
    return sorted(
        combos, key=lambda c: (c == "(none)", -len(c) if c != "(none)" else 0)
    )


def is_sql_baseline(series: pd.Series) -> bool:
    return bool((series == 1.0).all())


def legend_label(combo: str, sql_key: str) -> str:
    """Map raw combo keys to legend text."""
    if combo == sql_key:
        return "All SQL"
    if combo == "(none)":
        return "All Python"
    return combo


def with_speedup_err(data: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    """
    Add a 'speedup_std' column using error propagation.

    speedup = sql_median / combo_median
    sigma(speedup) approx speedup * sqrt(cv_combo^2 + cv_sql^2)

    cv_sql is looked up from the SQL-baseline row (speedup == 1.0) for each
    unique combination of join_cols.
    """
    sql_cv = (
        data[data["speedup"] == 1.0][join_cols + ["total_cv"]]
        .rename(columns={"total_cv": "sql_cv"})
        .drop_duplicates(join_cols)
    )
    out = data.merge(sql_cv, on=join_cols, how="left")
    out["speedup_std"] = out["speedup"] * np.sqrt(
        out["total_cv"] ** 2 + out["sql_cv"].fillna(0) ** 2
    )
    return out


def set_log_ticks(ax, values, axis: str = "x") -> None:
    """
    On a log-scale axis, place ticks at every tested value but only label
    exact powers of two. Non-power-of-2 values (e.g. saturation points like
    N=300 or N=92) get a tick mark so the axis extends correctly, but no label.
    """

    def _is_pow2(v: float) -> bool:
        n = int(v)
        return n >= 1 and (n & (n - 1)) == 0 and float(n) == v

    all_vals = sorted(set(values))
    ticks = [v for v in all_vals if _is_pow2(v)]
    labels = [str(int(v)) for v in ticks]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(mticker.FixedLocator(ticks))
    target.set_major_formatter(mticker.FixedFormatter(labels))
    target.set_minor_locator(mticker.NullLocator())


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: N-scaling speedup  (one file per query)
# ---------------------------------------------------------------------------


def plot_n_scaling_speedup(df: pd.DataFrame, out_dir: str) -> None:
    raw = df[(df["experiment"] == "n_scaling") & df["flags"].map(is_all_on)]
    if raw.empty:
        print("  [skip] n_scaling_speedup: no data")
        return

    data = with_speedup_err(raw, ["query", "n_requested"])

    for query in sorted(data["query"].unique()):
        qdata = data[data["query"] == query]
        max_n = _SF_MAX_N.get(query, float("inf"))
        qdata = qdata[qdata["n_requested"] <= max_n]
        combos = sort_combos(qdata["combo"].unique().tolist())
        colors = combo_colors(combos)
        n_vals = sorted(qdata["n_requested"].unique())
        sql_key = next(
            (
                c
                for c in combos
                if is_sql_baseline(qdata[qdata["combo"] == c]["speedup"])
            ),
            "",
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        for combo in combos:
            cdata = qdata[qdata["combo"] == combo].sort_values("n_requested")
            baseline = is_sql_baseline(cdata["speedup"])
            col = colors[combo]
            ax.errorbar(
                cdata["n_requested"],
                cdata["speedup"],
                yerr=cdata["speedup_std"],
                color=col,
                marker="o",
                markersize=4,
                linewidth=1.8,
                linestyle="--" if baseline else "-",
                label=legend_label(combo, sql_key),
                alpha=0.9,
                capsize=3,
                elinewidth=1.0,
            )

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        set_log_ticks(ax, n_vals, "x")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("N (sweep size)")
        ax.set_ylabel("Speed-up  (SQL : combo)")
        ax.set_title(f"{query}: Speed-up vs SQL (>1 = Python wins)")
        n_combos = len(combos)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.36),
            ncol=(n_combos + 2) // 3,  # ceil(n/3), always 3 rows
            framealpha=0.8,
        )
        fig.subplots_adjust(bottom=0.44)
        _save(fig, out_dir, f"n_scaling_speedup_{query.lower()}.pdf")


# ---------------------------------------------------------------------------
# Figure 2: N-scaling time decomposition  (one file per query)
# ---------------------------------------------------------------------------


def plot_n_scaling_decomp(df: pd.DataFrame, out_dir: str) -> None:
    data = df[(df["experiment"] == "n_scaling") & df["flags"].map(is_all_on)]
    if data.empty:
        print("  [skip] n_scaling_decomp: no data")
        return

    for query in sorted(data["query"].unique()):
        qdata = data[data["query"] == query]
        max_n = _SF_MAX_N.get(query, float("inf"))
        qdata = qdata[qdata["n_requested"] <= max_n]
        py = qdata[qdata["speedup"] != 1.0]
        if py.empty:
            print(f"  [skip] n_scaling_decomp {query}: no Python combos")
            continue

        max_n = qdata["n_requested"].max()
        best_combo = (
            py[py["n_requested"] == max_n]
            .sort_values("speedup", ascending=False)
            .iloc[0]["combo"]
        )
        best = qdata[qdata["combo"] == best_combo].sort_values("n_requested")
        n_vals = sorted(qdata["n_requested"].unique())

        fig, ax = plt.subplots(figsize=(6, 4))

        # fetch (one-time cost, should be flat)
        ax.errorbar(
            best["n_requested"],
            best["fetch_median"],
            yerr=best["fetch_std"],
            color=_PALETTE[1],  # cyan
            marker="s",
            markersize=4,
            linewidth=1.8,
            label="Python fetch (one-time)",
            capsize=3,
            elinewidth=1.0,
        )

        # logic (N * marginal)
        ax.errorbar(
            best["n_requested"],
            best["logic_median"],
            yerr=best["logic_std"],
            color=_PALETTE[6],  # rose
            marker="^",
            markersize=4,
            linewidth=1.8,
            label="Python logic",
            capsize=3,
            elinewidth=1.0,
        )

        # Python total
        ax.errorbar(
            best["n_requested"],
            best["total_median"],
            yerr=best["total_std"],
            color=_PALETTE[0],  # indigo
            marker="o",
            markersize=4,
            linewidth=1.8,
            linestyle="--",
            alpha=0.8,
            label="Python total",
            capsize=3,
            elinewidth=1.0,
        )

        ax.set_xscale("log")
        set_log_ticks(ax, n_vals, "x")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("N (sweep size)")
        ax.set_ylabel("Time (s)")
        combo_label = "All Python" if best_combo == "(none)" else best_combo
        ax.set_title(f"{query}: Time Decomposition  [best combo: {combo_label}]")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.36),
            ncol=1,  # 3 entries, 1 col = 3 rows, matching speedup legend height
            framealpha=0.8,
        )
        fig.subplots_adjust(bottom=0.44)
        _save(fig, out_dir, f"n_scaling_decomp_{query.lower()}.pdf")


# ---------------------------------------------------------------------------
# Figure 3: SF scaling  (one file per query)
# ---------------------------------------------------------------------------


# Upper bound on unique parameter values for queries with bounded sweep spaces.
# Derived by running generate_params(n) at large n and counting deduplicated results.
# n_scaling plots are clipped to these limits so the x-axis doesn't extend past the
# param space size (e.g. Q14 has 62 distinct 1-month windows, Q10 has 17 distinct date pairs).
_SF_MAX_N = {
    "Q1": 61,
    "Q3": 31,
    "Q4": 60,
    "Q7": 300,
    "Q9": 92,
    "Q10": 17,
    "Q14": 62,
    "Q18": 201,
}


def plot_sf_scaling(df: pd.DataFrame, out_dir: str) -> None:
    raw = df[(df["experiment"] == "sf_scaling") & df["flags"].map(is_all_on)]
    if raw.empty:
        print("  [skip] sf_scaling: no data")
        return

    data = with_speedup_err(raw, ["query", "sf", "n_requested"])

    for query in sorted(data["query"].unique()):
        qdata = data[data["query"] == query]
        max_n = _SF_MAX_N.get(query, float("inf"))
        n_vals = sorted(n for n in qdata["n_requested"].unique() if n <= max_n)
        sf_vals = sorted(qdata["sf"].unique())
        max_sf = max(sf_vals)

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, n in enumerate(n_vals):
            ndata = qdata[qdata["n_requested"] == n]
            py = ndata[ndata["speedup"] != 1.0]
            if py.empty:
                continue
            best_combo = (
                py[py["sf"] == max_sf]
                .sort_values("speedup", ascending=False)
                .iloc[0]["combo"]
            )
            best = ndata[ndata["combo"] == best_combo].sort_values("sf")
            combo_label = "All Python" if best_combo == "(none)" else best_combo
            ax.errorbar(
                best["sf"],
                best["speedup"],
                yerr=best["speedup_std"],
                color=_PALETTE[i % len(_PALETTE)],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=f"N={n}  (best: {combo_label})",
                capsize=3,
                elinewidth=1.0,
            )

        if not n_vals:
            plt.close(fig)
            print(f"  [skip] sf_scaling_{query.lower()}: no valid N data")
            continue

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        set_log_ticks(ax, sf_vals, "x")
        ax.set_xlabel("Scale factor (SF)")
        ax.set_ylabel("Speed-up  (SQL : best combo)")
        ax.set_title(f"{query}: Speed-up vs Scale Factor (>1 = Python wins)")
        n_lines = len(n_vals)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.36),
            ncol=(n_lines + 2) // 3,
            framealpha=0.8,
        )
        fig.subplots_adjust(bottom=0.44)
        _save(fig, out_dir, f"sf_scaling_{query.lower()}.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Flag ablation
# ---------------------------------------------------------------------------


def plot_flag_ablation(df: pd.DataFrame, out_dir: str) -> None:
    raw = df[df["experiment"] == "flag_ablation"].copy()
    if raw.empty:
        print("  [skip] flag_ablation: no data")
        return

    data = with_speedup_err(raw, ["query", "flags"])
    data["flags_label"] = data["flags"].map(flags_label)

    for query in sorted(data["query"].unique()):
        qdata = data[data["query"] == query]

        all_on_py = qdata[
            (qdata["flags_label"] == "all ON") & (qdata["speedup"] != 1.0)
        ]
        if all_on_py.empty:
            print(f"  [skip] flag_ablation {query}: no Python wins")
            continue

        best_combo = all_on_py.sort_values("speedup", ascending=False).iloc[0]["combo"]
        cdata = qdata[qdata["combo"] == best_combo]

        def variant_key(v):
            return (0, v) if v == "all ON" else (2, v) if v == "all OFF" else (1, v)

        variants = sorted(cdata["flags_label"].unique(), key=variant_key)
        speedups = [
            cdata[cdata["flags_label"] == v]["speedup"].mean() for v in variants
        ]
        errs = [
            cdata[cdata["flags_label"] == v]["speedup_std"].mean() for v in variants
        ]
        colors = [
            _PALETTE[2]
            if v == "all ON"
            else _PALETTE[6]
            if v == "all OFF"
            else _PALETTE[5]
            for v in variants
        ]

        fig, ax = plt.subplots(figsize=(max(4, len(variants) * 1.1), 4))
        bars = ax.bar(
            range(len(variants)),
            speedups,
            yerr=errs,
            color=colors,
            edgecolor="white",
            width=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Speed-up")
        combo_label = "All Python" if best_combo == "(none)" else best_combo
        ax.set_title(f"{query}: Flag Ablation  [best combo: {combo_label}]")

        for bar, spd, err in zip(bars, speedups, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.02,
                f"{spd:.2f}*",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        fig.tight_layout()
        _save(fig, out_dir, f"flag_ablation_{query.lower()}.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Crossover summary
# ---------------------------------------------------------------------------


def plot_crossover_summary(df: pd.DataFrame, out_dir: str) -> None:
    raw = df[
        (df["experiment"] == "n_scaling")
        & df["flags"].map(is_all_on)
        & (df["speedup"] != 1.0)
    ]
    if raw.empty:
        print("  [skip] crossover_summary: no n_scaling data")
        return

    data = with_speedup_err(raw, ["query", "n_requested"])

    records = []
    for (query, combo), grp in data.groupby(["query", "combo"]):
        wins = grp[grp["speedup"] > 1.0].sort_values("n_requested")
        crossover = float(wins.iloc[0]["n_requested"]) if not wins.empty else None
        max_n = grp["n_requested"].max()
        best_row = grp[grp["n_requested"] == max_n].iloc[0]
        records.append(
            {
                "query": query,
                "combo": combo,
                "crossover_n": crossover,
                "max_speedup": float(best_row["speedup"]),
                "max_speedup_std": float(best_row["speedup_std"]),
            }
        )
    summary = pd.DataFrame(records)

    best_per_query = (
        summary.sort_values("max_speedup", ascending=False)
        .groupby("query", sort=False)
        .first()
        .reset_index()
        .sort_values("query")
    )

    nq = len(best_per_query)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3, nq * 0.55 + 1.5)))

    queries = best_per_query["query"].tolist()
    speedups = best_per_query["max_speedup"].tolist()
    spd_errs = best_per_query["max_speedup_std"].tolist()
    crossovers = best_per_query["crossover_n"].tolist()
    combos = best_per_query["combo"].tolist()
    y = np.arange(nq)

    # Left: max speedup with error bars
    bar_colors = [_PALETTE[2] if s > 1.0 else _PALETTE[6] for s in speedups]
    bars = ax1.barh(
        y,
        speedups,
        xerr=spd_errs,
        color=bar_colors,
        edgecolor="white",
        height=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    ax1.axvline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [f"{q}  [{legend_label(c, '')}]" for q, c in zip(queries, combos)], fontsize=8
    )
    ax1.set_xlabel("Speed-up  (SQL : best Python combo)")
    ax1.set_title(f"Best speedup at N={int(data['n_requested'].max())}")
    x_max = max(s + e for s, e in zip(speedups, spd_errs))
    for bar, spd, err in zip(bars, speedups, spd_errs):
        ax1.text(
            spd + err + x_max * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{spd:.2f}*",
            va="center",
            fontsize=7.5,
        )
    ax1.set_xlim(0, x_max * 1.2)

    # Right: crossover N
    never_mask = [c is None for c in crossovers]
    plot_vals = [c if c is not None else 1 for c in crossovers]
    bar_colors2 = [_PALETTE[6] if nm else _PALETTE[2] for nm in never_mask]
    bars2 = ax2.barh(y, plot_vals, color=bar_colors2, edgecolor="white", height=0.6)
    ax2.set_xscale("log")
    # Let matplotlib choose sparse auto log ticks (1, 10, 100, 1000).
    # Custom ticks at every crossover value crowd the axis when values are close
    # on a log scale (e.g. 40 and 45). Exact values are shown as bar labels anyway.
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f"{q}  [{legend_label(c, '')}]" for q, c in zip(queries, combos)], fontsize=8
    )
    ax2.set_xlabel("Minimum N to beat SQL")
    ax2.set_title("Crossover sweep size")
    for bar, cv, nm in zip(bars2, crossovers, never_mask):
        label = "never" if nm else str(int(cv))
        ax2.text(
            bar.get_width() * 1.18,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=7.5,
            color=_PALETTE[6] if nm else "black",
        )
    # Pad the right edge so bar labels don't clip
    ax2.set_xlim(right=max(plot_vals) * 4)

    fig.suptitle("Per-query summary: best Python combo", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "crossover_summary.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot scaling experiment figures from a scaling_experiments CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        default=None,
        help="CSV to load. Defaults to latest scaling_experiments_*.csv in --log_dir.",
    )
    parser.add_argument(
        "--log_dir", default="logs", help="Directory to search for the latest CSV."
    )
    parser.add_argument(
        "--output_dir", default="figures", help="Directory to write figures into."
    )
    parser.add_argument(
        "--sf_files",
        nargs="+",
        default=None,
        metavar="CSV",
        help=(
            "One CSV per N value for SF-scaling plots. "
            "When given, SF scaling data is taken from these files instead of --file."
        ),
    )
    parser.add_argument(
        "--n_files",
        nargs="+",
        default=None,
        metavar="CSV",
        help=(
            "Replacement N-scaling CSVs for queries with bounded parameter spaces "
            "(Q1, Q3, Q9, Q18). Rows for those queries are replaced with this data."
        ),
    )
    parser.add_argument(
        "--include_invalid",
        action="store_true",
        help=(
            "Keep rows whose Python output disagreed with the SQL baseline "
            "(valid=False). Off by default since invalid speedups are usually "
            "empty-result artefacts and would mislead the figures."
        ),
    )
    args = parser.parse_args(argv)

    csv_path = args.file or find_latest_csv(args.log_dir)
    if not csv_path or not os.path.exists(csv_path):
        print("No CSV found. Pass --file or run scaling_experiments.py first.")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows  |  experiments: {df['experiment'].unique().tolist()}")

    valid = _valid_mask(df)
    n_invalid = int((~valid).sum())
    if n_invalid:
        breakdown = df[~valid].groupby(["query", "experiment"]).size().to_dict()
        if args.include_invalid:
            print(f"  {n_invalid} rows have valid=False, kept (--include_invalid)")
        else:
            df = df[valid].copy()
            print(
                f"  {n_invalid} rows have valid=False, dropped "
                "(re-add with --include_invalid)"
            )
        for (q, e), n in sorted(breakdown.items()):
            print(f"    {q} / {e}: {n}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing figures to: {args.output_dir}/\n")

    if args.sf_files:
        sf_frames = []
        for path in args.sf_files:
            print(f"Loading SF file: {path}")
            sf_frames.append(pd.read_csv(path))
        sf_df = pd.concat(sf_frames, ignore_index=True)
        if not args.include_invalid:
            sf_df = sf_df[_valid_mask(sf_df)].copy()
    else:
        sf_df = df

    if args.n_files:
        n_frames = []
        for path in args.n_files:
            print(f"Loading N file: {path}")
            n_frames.append(pd.read_csv(path))
        n_extra = pd.concat(n_frames, ignore_index=True)
        if not args.include_invalid:
            n_extra = n_extra[_valid_mask(n_extra)].copy()
        affected = set(n_extra["query"].unique())
        n_df = pd.concat([df[~df["query"].isin(affected)], n_extra], ignore_index=True)
    else:
        n_df = df

    plot_n_scaling_speedup(n_df, args.output_dir)
    plot_n_scaling_decomp(n_df, args.output_dir)
    plot_sf_scaling(sf_df, args.output_dir)
    plot_flag_ablation(df, args.output_dir)
    plot_crossover_summary(n_df, args.output_dir)


if __name__ == "__main__":
    main()
