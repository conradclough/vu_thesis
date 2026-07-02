"""
Cross-engine TPC-H comparison: DuckDB vs PostgreSQL vs MySQL vs SQLite.

Runs TPC-H queries Q1, Q3, Q4, Q5, Q6, Q7, Q10, Q12, Q14, Q18 against all
four engines at a chosen scale factor and times each one.

Steps:
  1. Generate TPC-H tables at --sf via DuckDB's dbgen (scripts.setup_db).
  2. Export every table to CSV once.
  3. For each non-DuckDB engine: create the standard TPC-H schema, bulk-load
     the CSVs client-side (executemany, not timed), then run the query set.
  4. DuckDB queries its dbgen tables directly (no export/load round-trip).
  5. Each query is run --repeats times per engine.
  6. Results go to a CSV in logs/.
  7. plot_comparison.py can be used to generate figures.

Usage:
  python3 -m benchmarks.compare_engine --sf 1
  python3 -m benchmarks.compare_engine --sf 1 --engines duckdb sqlite
  python3 -m benchmarks.compare_engine --sf 1 --skip_load --engines postgres
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import duckdb

from scripts.setup_db import setup_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Docker containers are made for Postgres, MySQL. Teardown isn't done
# automatically, `docker rm -f tpch-pg tpch-mysql` or run with
# --docker_teardown flag.
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_names(name: str) -> list[str]:
    """Run `docker ps -a --filter name=^NAME$` and return matching names."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return [n for n in result.stdout.split() if n]


def _docker_running_names(name: str) -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return [n for n in result.stdout.split() if n]


def _wait_for_connection(
    connect_fn, timeout: float, interval: float, label: str
) -> None:
    t0 = time.time()
    last_err: Exception | None = None
    while time.time() - t0 < timeout:
        try:
            conn = connect_fn()
            conn.close()
            logger.info(f"  [docker] {label} is accepting connections")
            return
        except Exception as e:
            last_err = e
            time.sleep(interval)
    raise RuntimeError(
        f"[docker] {label} did not become ready within {timeout:.0f}s: {last_err}"
    )


def _pg_mem_settings(mem_gb: float) -> tuple[int, int]:
    """(shared_buffers_mb, effective_cache_size_mb) for a total memory cap.

    Normal Postgres tuning: shared_buffers ~25% of budget,
    effective_cache_size ~75%.
    """
    shared_mb = max(128, int(mem_gb * 1024 * 0.25))
    cache_mb = max(256, int(mem_gb * 1024 * 0.75))
    return shared_mb, cache_mb


def ensure_docker_postgres(args: argparse.Namespace) -> str:
    """Start (or reuse) a Postgres container. Returns its name."""
    name = args.docker_pg_name
    if name in _docker_running_names(name):
        logger.info(f"[docker] {name} already running")
    elif name in _docker_names(name):
        logger.info(f"[docker] starting existing container {name}")
        subprocess.run(["docker", "start", name], check=True)
    else:
        shared_mb, cache_mb = _pg_mem_settings(args.docker_mem_gb)
        logger.info(
            f"[docker] creating container {name} ({args.docker_pg_image}, "
            f"shared_buffers={shared_mb}MB, effective_cache_size={cache_mb}MB, "
            f"cap={args.docker_mem_gb}GB)"
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-e",
                f"POSTGRES_USER={args.pg_user}",
                "-e",
                f"POSTGRES_PASSWORD={args.pg_password}",
                "-e",
                f"POSTGRES_DB={args.pg_dbname}",
                "-p",
                f"{args.pg_port}:5432",
                "--shm-size",
                f"{shared_mb}m",  # shared_buffers lives in /dev/shm
                "--memory",
                f"{args.docker_mem_gb}g",  # hard cap on the container
                args.docker_pg_image,
                "-c",
                f"shared_buffers={shared_mb}MB",
                "-c",
                f"effective_cache_size={cache_mb}MB",
            ],
            check=True,
        )

    def _connect():
        import psycopg2

        return psycopg2.connect(
            host=args.pg_host,
            port=args.pg_port,
            user=args.pg_user,
            password=args.pg_password,
            dbname=args.pg_dbname,
            connect_timeout=2,
        )

    _wait_for_connection(_connect, timeout=60, interval=1.5, label=f"postgres ({name})")
    return name


def ensure_docker_mysql(args: argparse.Namespace) -> str:
    """Start (or reuse) a MySQL container. Returns its name."""
    name = args.docker_mysql_name
    if name in _docker_running_names(name):
        logger.info(f"[docker] {name} already running")
    elif name in _docker_names(name):
        logger.info(f"[docker] starting existing container {name}")
        subprocess.run(["docker", "start", name], check=True)
    else:
        pool_mb = int(args.docker_mem_gb * 1024)
        logger.info(
            f"[docker] creating container {name} ({args.docker_mysql_image}, "
            f"innodb_buffer_pool_size={pool_mb}M, cap={args.docker_mem_gb}GB)"
        )
        env = ["-e", f"MYSQL_DATABASE={args.mysql_dbname}"]
        if args.mysql_password:
            env += ["-e", f"MYSQL_ROOT_PASSWORD={args.mysql_password}"]
        else:
            env += ["-e", "MYSQL_ALLOW_EMPTY_PASSWORD=yes"]
        if args.mysql_user != "root":
            env += [
                "-e",
                f"MYSQL_USER={args.mysql_user}",
                "-e",
                f"MYSQL_PASSWORD={args.mysql_password}",
            ]
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                *env,
                "-p",
                f"{args.mysql_port}:3306",
                "--memory",
                f"{args.docker_mem_gb}g",  # hard cap on the container
                args.docker_mysql_image,
                f"--innodb-buffer-pool-size={pool_mb}M",
            ],
            check=True,
        )

    def _connect():
        import pymysql

        return pymysql.connect(
            host=args.mysql_host,
            port=args.mysql_port,
            user=args.mysql_user,
            password=args.mysql_password,
            database=args.mysql_dbname,
            connect_timeout=2,
        )

    # MySQL's entrypoint does a slow init (create db/users, maybe a
    # restart partway through), so give it a longer timeout than postgres.
    _wait_for_connection(_connect, timeout=120, interval=2.0, label=f"mysql ({name})")
    return name


def ensure_docker_engines(engines: list[str], args: argparse.Namespace) -> list[str]:
    """Start containers for any of postgres/mysql present in `engines`.

    Returns the list of container names that were touched, for --docker_teardown.
    """
    if not _docker_available():
        raise RuntimeError("--docker passed but Docker CLI not found.")
    started = []
    if "postgres" in engines:
        started.append(ensure_docker_postgres(args))
    if "mysql" in engines:
        started.append(ensure_docker_mysql(args))
    return started


def docker_teardown(names: list[str]) -> None:
    for name in names:
        logger.info(f"[docker] removing container {name}")
        subprocess.run(["docker", "rm", "-f", name], check=False)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TABLE_ORDER: list[str] = [
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
]

SCHEMA_DDL: dict[str, str] = {
    "region": """
        CREATE TABLE region (
            r_regionkey INTEGER NOT NULL,
            r_name CHAR(25) NOT NULL,
            r_comment VARCHAR(152),
            PRIMARY KEY (r_regionkey)
        )""",
    "nation": """
        CREATE TABLE nation (
            n_nationkey INTEGER NOT NULL,
            n_name CHAR(25) NOT NULL,
            n_regionkey INTEGER NOT NULL,
            n_comment VARCHAR(152),
            PRIMARY KEY (n_nationkey)
        )""",
    "supplier": """
        CREATE TABLE supplier (
            s_suppkey INTEGER NOT NULL,
            s_name CHAR(25) NOT NULL,
            s_address VARCHAR(40) NOT NULL,
            s_nationkey INTEGER NOT NULL,
            s_phone CHAR(15) NOT NULL,
            s_acctbal DECIMAL(15,2) NOT NULL,
            s_comment VARCHAR(101) NOT NULL,
            PRIMARY KEY (s_suppkey)
        )""",
    "customer": """
        CREATE TABLE customer (
            c_custkey INTEGER NOT NULL,
            c_name VARCHAR(25) NOT NULL,
            c_address VARCHAR(40) NOT NULL,
            c_nationkey INTEGER NOT NULL,
            c_phone CHAR(15) NOT NULL,
            c_acctbal DECIMAL(15,2) NOT NULL,
            c_mktsegment CHAR(10) NOT NULL,
            c_comment VARCHAR(117) NOT NULL,
            PRIMARY KEY (c_custkey)
        )""",
    "part": """
        CREATE TABLE part (
            p_partkey INTEGER NOT NULL,
            p_name VARCHAR(55) NOT NULL,
            p_mfgr CHAR(25) NOT NULL,
            p_brand CHAR(10) NOT NULL,
            p_type VARCHAR(25) NOT NULL,
            p_size INTEGER NOT NULL,
            p_container CHAR(10) NOT NULL,
            p_retailprice DECIMAL(15,2) NOT NULL,
            p_comment VARCHAR(23) NOT NULL,
            PRIMARY KEY (p_partkey)
        )""",
    "partsupp": """
        CREATE TABLE partsupp (
            ps_partkey INTEGER NOT NULL,
            ps_suppkey INTEGER NOT NULL,
            ps_availqty INTEGER NOT NULL,
            ps_supplycost DECIMAL(15,2) NOT NULL,
            ps_comment VARCHAR(199) NOT NULL,
            PRIMARY KEY (ps_partkey, ps_suppkey)
        )""",
    "orders": """
        CREATE TABLE orders (
            o_orderkey INTEGER NOT NULL,
            o_custkey INTEGER NOT NULL,
            o_orderstatus CHAR(1) NOT NULL,
            o_totalprice DECIMAL(15,2) NOT NULL,
            o_orderdate DATE NOT NULL,
            o_orderpriority CHAR(15) NOT NULL,
            o_clerk CHAR(15) NOT NULL,
            o_shippriority INTEGER NOT NULL,
            o_comment VARCHAR(79) NOT NULL,
            PRIMARY KEY (o_orderkey)
        )""",
    "lineitem": """
        CREATE TABLE lineitem (
            l_orderkey INTEGER NOT NULL,
            l_partkey INTEGER NOT NULL,
            l_suppkey INTEGER NOT NULL,
            l_linenumber INTEGER NOT NULL,
            l_quantity DECIMAL(15,2) NOT NULL,
            l_extendedprice DECIMAL(15,2) NOT NULL,
            l_discount DECIMAL(15,2) NOT NULL,
            l_tax DECIMAL(15,2) NOT NULL,
            l_returnflag CHAR(1) NOT NULL,
            l_linestatus CHAR(1) NOT NULL,
            l_shipdate DATE NOT NULL,
            l_commitdate DATE NOT NULL,
            l_receiptdate DATE NOT NULL,
            l_shipinstruct CHAR(25) NOT NULL,
            l_shipmode CHAR(10) NOT NULL,
            l_comment VARCHAR(44) NOT NULL,
            PRIMARY KEY (l_orderkey, l_linenumber)
        )""",
}


# ---------------------------------------------------------------------------
# Cross engine helpers
# ---------------------------------------------------------------------------


def date_lit(engine: str, iso: str) -> str:
    return f"'{iso}'" if engine == "sqlite" else f"DATE '{iso}'"


def date_add(engine: str, expr: str, n: int, unit: str) -> str:
    if engine == "mysql":
        return f"DATE_ADD({expr}, INTERVAL {n} {unit.upper()})"
    if engine == "sqlite":
        return f"date({expr}, '+{n} {unit}s')"
    return f"{expr} + INTERVAL '{n}' {unit}"  # duckdb, postgres


def date_sub(engine: str, expr: str, n: int, unit: str) -> str:
    if engine == "mysql":
        return f"DATE_SUB({expr}, INTERVAL {n} {unit.upper()})"
    if engine == "sqlite":
        return f"date({expr}, '-{n} {unit}s')"
    return f"{expr} - INTERVAL '{n}' {unit}"  # duckdb, postgres


def year_expr(engine: str, col: str) -> str:
    if engine == "sqlite":
        return f"CAST(strftime('%Y', {col}) AS INTEGER)"
    return f"EXTRACT(YEAR FROM {col})"  # duckdb, postgres, mysql all support this


# ---------------------------------------------------------------------------
# The 10 queries, one builder function each, parameterised by engine.
# ---------------------------------------------------------------------------


def q1(engine: str) -> str:
    return f"""
        SELECT
            l_returnflag, l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= {date_sub(engine, date_lit(engine, "1998-12-01"), 90, "day")}
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """


def q3(engine: str) -> str:
    return f"""
        SELECT
            l_orderkey,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue,
            o_orderdate, o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < {date_lit(engine, "1995-03-15")}
          AND l_shipdate > {date_lit(engine, "1995-03-15")}
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10
    """


def q4(engine: str) -> str:
    lo = date_lit(engine, "1993-07-01")
    return f"""
        SELECT o_orderpriority, COUNT(*) AS order_count
        FROM orders
        WHERE o_orderdate >= {lo}
          AND o_orderdate < {date_add(engine, lo, 3, "month")}
          AND EXISTS (
              SELECT * FROM lineitem
              WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate
          )
        GROUP BY o_orderpriority
        ORDER BY o_orderpriority
    """


def q5(engine: str) -> str:
    lo = date_lit(engine, "1994-01-01")
    return f"""
        SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND l_suppkey = s_suppkey
          AND c_nationkey = s_nationkey
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name = 'ASIA'
          AND o_orderdate >= {lo}
          AND o_orderdate < {date_add(engine, lo, 1, "year")}
        GROUP BY n_name
        ORDER BY revenue DESC
    """


def q6(engine: str) -> str:
    lo = date_lit(engine, "1994-01-01")
    return f"""
        SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= {lo}
          AND l_shipdate < {date_add(engine, lo, 1, "year")}
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24
    """


def q7(engine: str) -> str:
    return f"""
        SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
        FROM (
            SELECT
                n1.n_name AS supp_nation,
                n2.n_name AS cust_nation,
                {year_expr(engine, "l_shipdate")} AS l_year,
                l_extendedprice * (1 - l_discount) AS volume
            FROM supplier, lineitem, orders, customer, nation n1, nation n2
            WHERE s_suppkey = l_suppkey
              AND o_orderkey = l_orderkey
              AND c_custkey = o_custkey
              AND s_nationkey = n1.n_nationkey
              AND c_nationkey = n2.n_nationkey
              AND (
                  (n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
                  OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')
              )
              AND l_shipdate BETWEEN {date_lit(engine, "1995-01-01")}
                                  AND {date_lit(engine, "1996-12-31")}
        ) AS shipping
        GROUP BY supp_nation, cust_nation, l_year
        ORDER BY supp_nation, cust_nation, l_year
    """


def q10(engine: str) -> str:
    lo = date_lit(engine, "1993-10-01")
    return f"""
        SELECT
            c_custkey, c_name,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue,
            c_acctbal, n_name, c_address, c_phone, c_comment
        FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate >= {lo}
          AND o_orderdate < {date_add(engine, lo, 3, "month")}
          AND l_returnflag = 'R'
          AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        ORDER BY revenue DESC
        LIMIT 20
    """


def q12(engine: str) -> str:
    lo = date_lit(engine, "1994-01-01")
    return f"""
        SELECT
            l_shipmode,
            SUM(CASE WHEN o_orderpriority = '1-URGENT'
                        OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
            SUM(CASE WHEN o_orderpriority <> '1-URGENT'
                        AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
        FROM orders, lineitem
        WHERE o_orderkey = l_orderkey
          AND l_shipmode IN ('MAIL', 'SHIP')
          AND l_commitdate < l_receiptdate
          AND l_shipdate < l_commitdate
          AND l_receiptdate >= {lo}
          AND l_receiptdate < {date_add(engine, lo, 1, "year")}
        GROUP BY l_shipmode
        ORDER BY l_shipmode
    """


def q14(engine: str) -> str:
    lo = date_lit(engine, "1995-09-01")
    return f"""
        SELECT
            100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                               THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
            / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= {lo}
          AND l_shipdate < {date_add(engine, lo, 1, "month")}
    """


def q18(_: str) -> str:
    return """
        SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice,
               SUM(l_quantity) AS sum_qty
        FROM customer, orders, lineitem
        WHERE o_orderkey IN (
            SELECT l_orderkey FROM lineitem
            GROUP BY l_orderkey HAVING SUM(l_quantity) > 300
        )
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
        GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
        ORDER BY o_totalprice DESC, o_orderdate
        LIMIT 100
    """


QUERY_BUILDERS = {
    "q1": q1,
    "q3": q3,
    "q4": q4,
    "q5": q5,
    "q6": q6,
    "q7": q7,
    "q10": q10,
    "q12": q12,
    "q14": q14,
    "q18": q18,
}
ALL_QUERIES = list(QUERY_BUILDERS)
ALL_ENGINES = ["duckdb", "postgres", "mysql", "sqlite"]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_tables_to_csv(
    con: duckdb.DuckDBPyConnection, csv_dir: str
) -> dict[str, str]:
    os.makedirs(csv_dir, exist_ok=True)
    paths = {}
    for table in TABLE_ORDER:
        path = os.path.join(csv_dir, f"{table}.csv")
        con.execute(f"COPY {table} TO '{path}' (FORMAT CSV, HEADER, DELIMITER ',')")
        paths[table] = path
        logger.info(f"  exported {table} -> {path}")
    return paths


# ---------------------------------------------------------------------------
# Per-engine connection + schema + load
# ---------------------------------------------------------------------------


@dataclass
class EngineConn:
    engine: str
    conn: object
    placeholder: str  # "%s" for postgres/mysql, "?" for sqlite


def connect_postgres(args: argparse.Namespace):
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError(
            "postgres engine requires psycopg2-binary: pip install psycopg2-binary"
        ) from e
    return psycopg2.connect(
        host=args.pg_host,
        port=args.pg_port,
        user=args.pg_user,
        password=args.pg_password,
        dbname=args.pg_dbname,
    )


def connect_mysql(args: argparse.Namespace):
    try:
        import pymysql
    except ImportError as e:
        raise RuntimeError("mysql engine requires pymysql: pip install pymysql") from e
    return pymysql.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_dbname,
        autocommit=False,
    )


def connect_sqlite(args: argparse.Namespace):
    import sqlite3

    return sqlite3.connect(args.sqlite_path)


def connect_engine(engine: str, args: argparse.Namespace) -> EngineConn:
    if engine == "postgres":
        return EngineConn(engine, connect_postgres(args), "%s")
    if engine == "mysql":
        return EngineConn(engine, connect_mysql(args), "%s")
    if engine == "sqlite":
        return EngineConn(engine, connect_sqlite(args), "?")
    raise ValueError(f"connect_engine not applicable to {engine}")


def create_schema(ec: EngineConn) -> None:
    cur = ec.conn.cursor()
    for table in reversed(TABLE_ORDER):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    for table in TABLE_ORDER:
        cur.execute(SCHEMA_DDL[table])
    ec.conn.commit()


def load_table(
    ec: EngineConn, table: str, csv_path: str, batch_size: int = 20000
) -> int:
    cur = ec.conn.cursor()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        columns = next(reader)
        insert_sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join([ec.placeholder] * len(columns))})"
        )
        n_rows = 0
        batch: list[tuple] = []
        for row in reader:
            batch.append(tuple(row))
            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                n_rows += len(batch)
                batch = []
        if batch:
            cur.executemany(insert_sql, batch)
            n_rows += len(batch)
    ec.conn.commit()
    return n_rows


def load_all_tables(ec: EngineConn, csv_paths: dict[str, str]) -> None:
    for table in TABLE_ORDER:
        t0 = time.perf_counter()
        n = load_table(ec, table, csv_paths[table])
        logger.info(
            f"  [{ec.engine}] loaded {table}: {n} rows ({time.perf_counter() - t0:.1f}s)"
        )


def analyze_all(ec: EngineConn) -> None:
    """Refresh planner after load.

    Right after a load, Postgres/MySQL/SQLite have empty
    or stale table statistics, which can negatively affect their query planners
    """
    cur = ec.conn.cursor()
    t0 = time.perf_counter()
    if ec.engine == "postgres":
        for table in TABLE_ORDER:
            cur.execute(f"ANALYZE {table}")
    elif ec.engine == "mysql":
        for table in TABLE_ORDER:
            cur.execute(f"ANALYZE TABLE {table}")
    elif ec.engine == "sqlite":
        cur.execute("ANALYZE")
    ec.conn.commit()
    logger.info(f"  [{ec.engine}] ANALYZE done ({time.perf_counter() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class QueryStats:
    engine: str
    query: str
    sf: float
    repeats: int
    times: list[float]
    row_count: int

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def cv(self) -> float:
        m = self.median
        return self.std / m if m > 0 else 0.0


def time_query_duckdb(
    con: duckdb.DuckDBPyConnection, sql: str, repeats: int
) -> tuple[list[float], int]:
    times = []
    row_count = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        rows = con.execute(sql).fetchall()
        times.append(time.perf_counter() - t0)
        row_count = len(rows)
    return times, row_count


def time_query_dbapi(ec: EngineConn, sql: str, repeats: int) -> tuple[list[float], int]:
    times = []
    row_count = 0
    cur = ec.conn.cursor()
    for _ in range(repeats):
        t0 = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        times.append(time.perf_counter() - t0)
        row_count = len(rows)
    return times, row_count


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "engine",
    "query",
    "sf",
    "repeats",
    "median_s",
    "std_s",
    "cv",
    "min_s",
    "max_s",
    "row_count",
    "timestamp",
]


def run_comparison(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    csv_dir = args.csv_dir or tempfile.mkdtemp(prefix="tpch_csv_")
    csv_paths: dict[str, str] | None = None

    logger.info(f"Generating TPC-H data (SF={args.sf}) via DuckDB dbgen")
    duck_con = setup_db(sf=args.sf, memory_limit=args.memory_limit)

    if not args.skip_load:
        logger.info(f"Exporting tables to CSV -> {csv_dir}")
        csv_paths = export_tables_to_csv(duck_con, csv_dir)

    docker_containers: list[str] = []
    if args.docker:
        docker_containers = ensure_docker_engines(args.engines, args)

    engine_conns: dict[str, EngineConn] = {}
    for engine in args.engines:
        if engine == "duckdb":
            continue
        logger.info(f"Connecting to {engine}")
        ec = connect_engine(engine, args)
        engine_conns[engine] = ec
        if not args.skip_load:
            logger.info(f"[{engine}] creating schema")
            create_schema(ec)
            logger.info(f"[{engine}] loading {len(TABLE_ORDER)} tables")
            load_all_tables(ec, csv_paths)
            logger.info(f"[{engine}] running ANALYZE")
            analyze_all(ec)
        else:
            logger.info(f"[{engine}] --skip_load: assuming data already present")

    try:
        for qname in args.queries:
            builder = QUERY_BUILDERS[qname]
            for engine in args.engines:
                sql = builder(engine)
                logger.info(f"[{engine}] {qname}  ({args.repeats} repeats)")
                if engine == "duckdb":
                    times, row_count = time_query_duckdb(duck_con, sql, args.repeats)
                else:
                    times, row_count = time_query_dbapi(
                        engine_conns[engine], sql, args.repeats
                    )

                stats = QueryStats(
                    engine, qname, args.sf, args.repeats, times, row_count
                )
                logger.info(
                    f"    median={stats.median:.4f}s  std={stats.std:.4f}s  "
                    f"rows={row_count}"
                )
                rows.append(
                    {
                        "engine": engine,
                        "query": qname,
                        "sf": args.sf,
                        "repeats": args.repeats,
                        "median_s": round(stats.median, 5),
                        "std_s": round(stats.std, 5),
                        "cv": round(stats.cv, 4),
                        "min_s": round(min(times), 5),
                        "max_s": round(max(times), 5),
                        "row_count": row_count,
                        "timestamp": timestamp,
                    }
                )
    finally:
        duck_con.close()
        for ec in engine_conns.values():
            ec.conn.close()
        if not args.keep_csvs and csv_paths and not args.csv_dir:
            shutil.rmtree(csv_dir, ignore_errors=True)
        if args.docker_teardown and docker_containers:
            docker_teardown(docker_containers)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DuckDB / PostgreSQL / MySQL / SQLite on TPC-H queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=ALL_ENGINES,
        default=ALL_ENGINES,
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        choices=ALL_QUERIES,
        default=ALL_QUERIES,
    )
    parser.add_argument(
        "--memory_limit", type=str, default="4GB", help="DuckDB memory limit"
    )
    parser.add_argument(
        "--skip_load",
        action="store_true",
        help="Assume schema+data already exist in every target engine; only run queries.",
    )
    parser.add_argument(
        "--csv_dir",
        default=None,
        help="Directory for the intermediate CSV export. Default: temp dir, deleted after run.",
    )
    parser.add_argument(
        "--keep_csvs",
        action="store_true",
        help="Don't delete the exported CSVs when using the default temp dir.",
    )
    parser.add_argument("--output_dir", default="logs")

    dk = parser.add_argument_group("docker")
    dk.add_argument(
        "--docker",
        action="store_true",
        help=(
            "Auto-start throwaway Postgres/MySQL containers (via the `docker` "
            "CLI) matching the --pg_*/--mysql_* connection settings, and wait "
            "until each is accepting connections before loading data. "
            "No-op for duckdb/sqlite."
        ),
    )
    dk.add_argument(
        "--docker_teardown",
        action="store_true",
        help="Remove the containers started by --docker when the run finishes. "
        "Default: leave them running so the next run can reuse them.",
    )
    dk.add_argument("--docker_pg_image", default="postgres:18")
    dk.add_argument("--docker_mysql_image", default="mysql:9")
    dk.add_argument("--docker_pg_name", default="tpch-pg")
    dk.add_argument("--docker_mysql_name", default="tpch-mysql")
    dk.add_argument(
        "--docker_mem_gb",
        type=float,
        default=5.0,
        help="Memory cap for each container, and the basis for "
        "shared_buffers/effective_cache_size (postgres) and "
        "innodb_buffer_pool_size (mysql).",
    )

    pg = parser.add_argument_group("postgres")
    pg.add_argument("--pg_host", default="localhost")
    pg.add_argument("--pg_port", type=int, default=5432)
    pg.add_argument("--pg_user", default="postgres")
    pg.add_argument("--pg_password", default="postgres")
    pg.add_argument("--pg_dbname", default="tpch")

    my = parser.add_argument_group("mysql")
    my.add_argument("--mysql_host", default="localhost")
    my.add_argument("--mysql_port", type=int, default=3306)
    my.add_argument("--mysql_user", default="root")
    my.add_argument("--mysql_password", default="")
    my.add_argument("--mysql_dbname", default="tpch")

    sq = parser.add_argument_group("sqlite")
    sq.add_argument(
        "--sqlite_path",
        default=None,
        help="Default: logs/tpch_compare_sf<SF>.sqlite (deleted/recreated unless --skip_load).",
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    setup_logging()
    args = parse_args(argv)

    if args.sqlite_path is None:
        args.sqlite_path = os.path.join(
            args.output_dir, f"tpch_compare_sf{args.sf}.sqlite"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    if (
        "sqlite" in args.engines
        and not args.skip_load
        and os.path.exists(args.sqlite_path)
    ):
        os.remove(args.sqlite_path)

    logger.info(f"Engines: {args.engines}")
    logger.info(f"Queries: {args.queries}")
    logger.info(f"SF={args.sf}  repeats={args.repeats}  skip_load={args.skip_load}")

    rows = run_comparison(args)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir, f"compare_engine_sf{args.sf}_{timestamp}.csv"
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
