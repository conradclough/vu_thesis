# Performance Characterization of SQL-Python Query Partitioning for Parametric Analytical Workloads  

BSc Computer Science thesis for Vrije Universiteit Amsterdam

## Abstract 

Modern businesses generate and ingest extreme amounts of data regularly, and it's the job of data analysts to pare this down to make it digestible for people. To this end, these analysts work with large relational datasets using SQL-based analytical databases. DuckDB makes query execution very fast, so SQL is a good choice for exploration.  

Analysis often requires a sweep of many related queries, such as over a date window, a range of thresholds, or varying a parameter for "what-if" scenarios. Here, analyst workflows and DuckDB's execution pipeline are misaligned.

DuckDB treats each query as independent. When the same dataset is scanned, joined, and aggregated many times with only one parameter changed, the database repeats the entire execution for each call. This is a bottleneck for analysts, who often script parameter sweeps in notebooks and pipelines rather than issuing one-off queries. SQL has no mechanism to recognise or use query similarity. 

We decompose each query into logical steps. We ask: which steps are sweep-invariant and which are sweep-variant? We also investigate whether pre-aggregation steps in SQL can further improve asymptotic per-query cost.
We implement this framework for ten TPC-H queries, benchmarking all valid combinations of SQL/Python for each logical step. This is compared against a pure DuckDB baseline.

---

## Layout

```
benchmarks/
  benchmark_sweep.py              shared harness: timing, reporting, argument parsing
  q{1,3,4,6,7,9,10,11,14,18}.py   per-query benchmark modules

scripts/
  scaling_experiments.py          run the three experiments, make CSV
  plot_experiments.py             generate figures from a CSV
  setup_db.py                     create an in-memory TPC-H DuckDB instance

logs/
  final_nscaling.csv              archived n_scaling results  (516 rows)
  final_sf.csv                    archived sf_scaling results (380 rows)
  final_flag.csv                  archived flag_ablation results (98 rows)
```

---

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if needed 
pipx install uv

# Create the venv and install all dependencies
uv sync

# Activate venv (all commands below assume an active venv)
source .venv/bin/activate
```

All commands must be run from the **project root** and use `python3 -m` (not `python3 scripts/...`).

---

# Regenerate figures from the included data

The `logs/final_*.csv` files are the experimental results referenced in the thesis. 
To reproduce all thesis figures from them:

```bash
python3 -m scripts.plot_experiments \
    --file      logs/final_flag.csv \
    --sf_files  logs/final_sf.csv \
    --n_files   logs/final_nscaling.csv \
    --output_dir figures
```

Each CSV gives different plots: 
- `--file` (`final_flag.csv`): `flag_ablation` figures; also the default
  source for any experiment not overridden below
- `--sf_files` (`final_sf.csv`):  `sf_scaling` figures; the SF data was
  collected at fixed per-query crossover N values, not the default N=200
- `--n_files` (`final_nscaling.csv`): `n_scaling` and `crossover_summary`
  figures; includes measurements at exact N values for queries with bounded parameter spaces (Q1, Q3, Q9, Q18)

Figures are written to `figures/` as PDFs:

| File | Description |
|---|---|
| `n_scaling_speedup_{q}.pdf` | Speedup vs N for every valid combo |
| `n_scaling_decomp_{q}.pdf` | Fetch / logic time decomposition for the best combo |
| `sf_scaling_{q}.pdf` | Speedup vs scale factor at crossover N |
| `flag_ablation_{q}.pdf` | Speedup with each optimisation flag toggled off |
| `crossover_summary.pdf` | Best speedup and crossover N for all queries |

---

## Running experiments

The three experiments are independent and can be run together or separately.
A CSV is produced, which is then used by `plot_experiments.py`.

### Full run (all queries, all experiments)

```bash
python3 -m scripts.scaling_experiments \
    --sf 1 \
    --n 200 \
    --n_values 1 2 4 8 16 32 64 128 256 512 1024 \
    --sf_values 1 2 4 8 16 \
    --repeats 5 \
    --output_dir logs
```

Runtime at SF=1 with 5 repeats takes multiple hours. 
The output is `logs/scaling_experiments_<timestamp>.csv`.

### How the included data was produced

Queries with a finite parameter space saturate at some maximum N_sat (see the parameter space bounds table below). Queries with an unbounded or very large parameter space are evaluated up to N_max = 4096 (or 512 for SF scaling).

For bounded queries, N_max = N_sat.

Three experiments were run:

**N-scaling**: sweep size N in {2^k : 2^k <= N_max} at SF=1; all valid combos; 5 repeats each.
```bash
python3 -m scripts.scaling_experiments \
    --experiments n_scaling \
    --sf 1 --repeats 5 \
    --n_values 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
```

**Flag ablation**: all optimisation flags enabled, then each disabled individually, then all disabled; at SF=1, N=N_max, best combo per query only; 5 repeats each.
```bash
python3 -m scripts.scaling_experiments \
    --experiments flag_ablation \
    --sf 1 --n 4096 --repeats 5
```

(Bounded queries saturate below 4096; `n_actual` in the CSV reflects the true
unique count.)

**SF scaling**: SF in {1, 2, 4, 8, 16} at N in {8\*4^k : 8\*4^k <= N_max} = {8, 32, 128, 512}; best combo per query only; 5 repeats each.

```bash
for n in 8 32 128 512; do
    python3 -m scripts.scaling_experiments \
        --experiments sf_scaling \
        --sf_values 1 2 4 8 16 --n $n --repeats 5
done
```

The resulting CSVs were merged and renamed to `final_nscaling.csv`,
`final_sf.csv`, and `final_flag.csv`.

---

## Entry point

### `python3 -m scripts.scaling_experiments`

Runs experiments and writes a CSV.

| Argument | Default | Description |
|---|---|---|
| `queries` | all | Space-separated query names: `q1 q3 q4 q6 q7 q9 q10 q11 q14 q18` |
| `--experiments` | all three | `n_scaling`, `sf_scaling`, `flag_ablation` (space-separated) |
| `--sf` | `1.0` | Scale factor for `n_scaling` and `flag_ablation` |
| `--n` | `200` | Batch size N for `sf_scaling` and `flag_ablation` |
| `--n_values` | `1 4 16 64 256 1024` | N values swept in `n_scaling` |
| `--sf_values` | `1.0 2.0 4.0 8.0 16.0` | SF values swept in `sf_scaling` |
| `--repeats` | `5` | Timing repeats per data point (median is reported) |
| `--memory_limit` | `4GB` | DuckDB memory limit; increase for SF > 4 |
| `--all_combos` | off | Run every valid combo in `sf_scaling`/`flag_ablation` (slow; default runs only the best combo per query plus the SQL baseline) |
| `--output_dir` | `logs` | Directory for the output CSV |

### `python3 -m scripts.plot_experiments`

Generates figures from a CSV made by `scaling_experiments`.

| Argument | Default | Description |
|---|---|---|
| `--file` | latest `scaling_experiments_*.csv` in `--log_dir` | Primary CSV |
| `--log_dir` | `logs` | Searched for the latest CSV when `--file` is omitted |
| `--output_dir` | `figures` | Directory to write PDFs into |
| `--sf_files` | - | One or more CSVs whose `sf_scaling` rows override the primary file |
| `--n_files` | - | One or more CSVs whose `n_scaling` rows replace the primary file for bounded queries |
| `--include_invalid` | off | Keep rows where Python output differed from SQL (normally dropped) |

### `python3 -m benchmarks.benchmark_sweep <query> [<query> ...]`

Runner for one or more queries: prints a full results table and writes a log file. Useful for checking a single query without running the full experiment pipeline.

```bash
# Run Q6 with default settings
python3 -m benchmarks.benchmark_sweep q6

# Run Q1 and Q14 at SF=4, 10 repeats, single fixed parameter
python3 -m benchmarks.benchmark_sweep q1 q14 --sf 4 --repeats 10 --single
```

Each query module can also be run directly:

```bash
python3 -m benchmarks.q6 --sf 1 --n 200
```

---

## Included data

All three CSVs share the same columns:

| Column | Description |
|---|---|
| `experiment` | `n_scaling`, `sf_scaling`, or `flag_ablation` |
| `query` | `Q1` ... `Q18` |
| `sf` | TPC-H scale factor |
| `n_requested` | N passed to `generate_params` |
| `n_actual` | Unique parameter values after deduplication (<= `n_requested` for bounded queries) |
| `combo` | SQL-assigned steps as a string (e.g. `GD`); `(none)` = pure Python; the all-SQL key = SQL baseline |
| `flags` | Comma-separated opt flag states, e.g. `opt_presort=on,opt_precompute=off` |
| `fetch_median` | Median time (s) for the one-time bulk SQL fetch |
| `fetch_std` | Sample std dev of fetch time |
| `logic_median` | Median time (s) for the N-query Python loop |
| `logic_std` | Sample std dev of logic time |
| `total_median` | `fetch_median + logic_median` |
| `total_std` | Propagated std dev |
| `total_cv` | Coefficient of variation (std/mean) |
| `n_repeats` | Number of timed repeats |
| `sql_baseline_median` | Total median of the all-SQL combo for the same (query, sf, n, flags) |
| `speedup` | `sql_baseline_median / total_median`; > 1 means the combo beats pure SQL |
| `valid` | `True` if Python output matched the SQL reference within tolerance |

---

## Understanding combos

Each query is decomposed into **steps**, operations such as a date filter, an aggregation, or a join. Any subset of steps can be assigned to SQL (run inside DuckDB) while the complement runs in Python/NumPy. A **combo** is identified by the string of SQL-assigned step letters, sorted in canonical order.

**Example Q6** (`steps = S D Q A`):

| Combo key | SQL handles | Python handles |
|---|---|---|
| `SDQA` | all | nothing, pure-SQL baseline |
| `SQ` | shipdate filter, quantity filter | discount filter, aggregation |
| `D` | discount filter | shipdate, quantity, aggregation |
| `(none)` | nothing | all Python |

The pure-SQL combo always has `speedup = 1.0` (it is the baseline).  A combo with `speedup > 1` runs faster than pure SQL for that N.

### Opt flags

Several queries have `--opt_*` flags that toggle individual NumPy optimisations (e.g. pre-sorting arrays to enable binary search, precomputing products at fetch time). The `flag_ablation` experiment measures their contribution by toggling each flag off one at a time.  All flags are `on` by default; the `flags` column in the CSV records the exact state used for each row.

---

## Per-query reference

### Step decompositions

The sweep parameter for each query is noted in parentheses.

| Query | Steps |
|---|---|
| Q1  | **S** (date filter), **G** (group encode), **D** (daily pre-agg), **A** (range-agg) |
| Q3  | **C** (segment), **D** (date), **G** (group+agg) |
| Q4  | **E** (late-key precompute), **J** (semi-join), **D** (date), **G** (group+count) |
| Q6  | **S** (shipdate), **Q** (quantity), **D** (discount), **A** (sum) |
| Q7  | **N** (nation pair), **Y** (year), **G** (group+agg) |
| Q9  | **C** (color), **G** (group+agg) |
| Q10 | **R** (returnflag), **D** (date), **A** (pre-agg), **G** (group+top-20) |
| Q11 | **N** (nation), **G** (group+sum), **F** (fraction threshold) |
| Q14 | **S** (shipdate), **P** (promo split), **A** (agg) |
| Q18 | **T** (threshold), **J** (semi-join), **G** (group+sum) |

The combo key in the CSV and figures lists the SQL-assigned step letters; the complement runs in Python. See the module docstring in `benchmarks/q{N}.py` for the full per-step definition.

### Parameter space bounds

Queries whose sweep parameter is drawn from a finite domain saturate at some maximum N; requesting more unique parameter values produces duplicates that are deduplicated before timing. This is why `_SF_MAX_N` in `plot_experiments.py` clips the x-axis of n-scaling plots.


| Query | Max N | Reason |
|---|---|---|
| Q1  | 61  | Shipdate offset drawn from a 61-day window fixed in the TPC-H spec |
| Q3  | 31  | 31 integer day-offsets * 5 segment values = 155 distinct queries |
| Q4  | 60  | 58 distinct month-starts * 4 window widths = 232 distinct queries |
| Q6  | -   | Discount is a continuous real value; parameter space is unbounded |
| Q7  | 300 | All unordered nation pairs: $\binom{25}{2} = 300$ combinations |
| Q9  | 92  | 92 distinct colour keywords defined in the TPC-H spec |
| Q10 | 17  | 16 distinct month-starts * 3 window widths = 48 distinct queries |
| Q11 | -   | Fraction threshold is a continuous real value; parameter space is unbounded |
| Q14 | 62  | 60 distinct calendar months in the sweep window |
| Q18 | 201 | Integer quantity threshold in [250, 450]: 201 distinct values |

For Q3, Q4, and Q10 the n-scaling figures show the number of dates requested on the x-axis; the total distinct queries at saturation is higher because each date crosses with window widths or segment values.
