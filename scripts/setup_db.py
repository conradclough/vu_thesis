import duckdb as ddb


def setup_db(
    sf: int = 4,
    memory_limit: str = "4GB",
    db_path: str = ":memory:",
) -> ddb.DuckDBPyConnection:
    """
    Set up an instance of the TPC-H database.

    Args:
        sf:           Scale factor for TPC-H database.
        memory_limit: Memory limit; must exceed the size of the TPC-H database.
        db_path:      Path for the DuckDB file.  Defaults to ":memory:".
                      Pass a file path (e.g. "/tmp/tpch_sf1.duckdb") when a
                      persistent, file-backed connection is required (which it
                      isn't for anything in the thesis)

    Returns:
        DuckDB connection (in-memory or file-backed depending on db_path).
    """
    con = ddb.connect(db_path)
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"CALL dbgen(sf={sf})")
    return con


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    con = setup_db()
    size = con.execute("PRAGMA database_size").fetchall()
    logger.info(f"Database size: {size}")
