"""Execution substrate (not a stage): the engine interface all stages run against.

Two jobs: base-table validation in stages 1-2 (parallel workers, shared node) and the
timed measurement in stage 3 (serial, node-exclusive, one core, median over attempts).
Backends: DuckDB, Postgres via pg_lab, Umbra. Also runs the output-equality check on
every executed rewrite.

Self-contained (no project imports) so it can be scp'd to the remote server standalone.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import duckdb
import json
import os
import re
import subprocess
import threading
import time
import uuid
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from tqdm import tqdm


def _json_default(o):
    """Fallback serializer for json.dump of result artifacts.

    DB drivers return types json can't encode: Umbra/Postgres (psycopg2) hand
    back NUMERIC columns — including integer AVG/SUM aggregates — as
    decimal.Decimal, and temporal columns as date/time objects. These reach the
    output via the sampled result rows stored in each entry's "results".  Decimal
    → float (matching DuckDB's double output for the same aggregates, so the
    artifact is consistent across engines); date/time → ISO-8601 string;
    anything else → str, so an unexpected type degrades one cell instead of
    crashing the whole run at the final dump.
    """
    import datetime
    from decimal import Decimal
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (datetime.date, datetime.datetime, datetime.time)):
        return o.isoformat()
    return str(o)


# sql_predicate_converter is scp'd alongside this file for cost-aggregate mode
try:
    from sql_predicate_converter import apply_sql_rules, reconstruct_with_fixed_join_order
except ImportError:
    try:
        from systematic_eval.sql_predicate_converter import apply_sql_rules, reconstruct_with_fixed_join_order
    except ImportError:
        apply_sql_rules = None
        reconstruct_with_fixed_join_order = None


# ===========================================================================
# Umbra execution engine support (inline, self-contained)
# ===========================================================================

# ── IMDB table list and schema DDL ──────────────────────────────────────────
_UMBRA_TABLES = [
    "aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type",
    "company_name", "company_type", "complete_cast", "info_type", "keyword",
    "kind_type", "link_type", "movie_companies", "movie_info", "movie_info_idx",
    "movie_keyword", "movie_link", "name", "person_info", "role_type", "title",
]

# CREATE TABLE DDL for Umbra (no PRIMARY KEY / FK constraints — Umbra is a
# query-processing engine that does not enforce constraints).
_UMBRA_SCHEMA_SQL = """
CREATE TABLE aka_name (
    id integer NOT NULL,
    person_id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE aka_title (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer NOT NULL
);

CREATE TABLE char_name (
    id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE comp_cast_type (
    id integer NOT NULL,
    kind character varying(32) NOT NULL
);

CREATE TABLE company_name (
    id integer NOT NULL,
    name text NOT NULL,
    country_code character varying(255),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_type (
    id integer NOT NULL,
    kind character varying(32) NOT NULL
);

CREATE TABLE complete_cast (
    id integer NOT NULL,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);

CREATE TABLE info_type (
    id integer NOT NULL,
    info character varying(32) NOT NULL
);

CREATE TABLE keyword (
    id integer NOT NULL,
    keyword text NOT NULL,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer NOT NULL,
    kind character varying(15) NOT NULL
);

CREATE TABLE link_type (
    id integer NOT NULL,
    link character varying(32) NOT NULL
);

CREATE TABLE movie_companies (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text
);

CREATE TABLE movie_info (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE movie_keyword (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE movie_link (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE name (
    id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE person_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE role_type (
    id integer NOT NULL,
    role character varying(32) NOT NULL
);

CREATE TABLE title (
    id integer NOT NULL,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);
"""


def _umbra_get_tables() -> list[str]:
    return _UMBRA_TABLES


def _umbra_get_schema() -> str:
    return _UMBRA_SCHEMA_SQL


def _duckdb_type_to_umbra(duckdb_type: str) -> str:
    """Map a DuckDB information_schema data_type onto an Umbra/Postgres type.

    Used when deriving the Umbra schema for non-IMDB datasets straight from
    their DuckDB file (whose tables were created via read_csv_auto, so types
    are whatever DuckDB inferred).  Unknown types fall back to text.
    """
    t = duckdb_type.strip().upper()
    # DECIMAL(p,s)/NUMERIC(p,s) carry precision/scale — pass through as numeric.
    if t.startswith("DECIMAL") or t.startswith("NUMERIC"):
        return "numeric" + t[t.index("(") :] if "(" in t else "numeric"
    mapping = {
        "BOOLEAN": "boolean", "BOOL": "boolean",
        "TINYINT": "smallint", "INT1": "smallint",
        "SMALLINT": "smallint", "INT2": "smallint", "SHORT": "smallint",
        "INTEGER": "integer", "INT": "integer", "INT4": "integer",
        "BIGINT": "bigint", "INT8": "bigint", "LONG": "bigint",
        "HUGEINT": "numeric(38,0)",
        "UTINYINT": "smallint", "USMALLINT": "integer",
        "UINTEGER": "bigint", "UBIGINT": "numeric(20,0)", "UHUGEINT": "numeric(39,0)",
        "REAL": "real", "FLOAT4": "real", "FLOAT": "real",
        "DOUBLE": "double precision", "FLOAT8": "double precision",
        "VARCHAR": "text", "CHAR": "text", "BPCHAR": "text",
        "TEXT": "text", "STRING": "text",
        "DATE": "date",
        "TIME": "time",
        "TIMESTAMP": "timestamp", "DATETIME": "timestamp",
        "TIMESTAMP WITH TIME ZONE": "timestamptz", "TIMESTAMPTZ": "timestamptz",
        "BLOB": "bytea", "BYTEA": "bytea",
        "UUID": "text",
    }
    return mapping.get(t, "text")


def _derive_umbra_schema_from_duckdb(db_file: str) -> tuple[list[str], str]:
    """Introspect a DuckDB file → (table list, CREATE TABLE DDL for Umbra).

    Column order follows ordinal_position so it matches `SELECT *` used by the
    CSV export in _copy_table_from_duckdb.  Constraints are dropped (Umbra does
    not enforce them, and nullable columns avoid COPY failures on empty fields).
    """
    con = duckdb.connect(database=db_file, read_only=True)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]
        ddl_blocks = []
        for table in tables:
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? "
                "ORDER BY ordinal_position",
                [table],
            ).fetchall()
            col_defs = [
                f'    "{name}" {_duckdb_type_to_umbra(dtype)}'
                for name, dtype in cols
            ]
            ddl_blocks.append(
                f'CREATE TABLE "{table}" (\n' + ",\n".join(col_defs) + "\n);"
            )
        return tables, "\n\n".join(ddl_blocks)
    finally:
        con.close()


# ── Umbra connection constants ────────────────────────────────────────────────
_UMBRA_CONTAINER = "umbradb"
_UMBRA_DATA_DIR = Path.home() / "umbra-db"


# ── UmbraRunner: adapted from umbra.py (parquet replaced by duckdb export) ──
class UmbraRunner:
    """Manages an Umbra Docker container and a psycopg2 connection pool.

    Data is loaded from an existing DuckDB file (imdb.duckdb, accidents.duckdb,
    …) by exporting each table to a temporary CSV staged in the Docker volume
    mount directory, then running COPY INTO Umbra.  The load is idempotent: if
    tables already contain data the load is skipped.
    """

    name = "Umbra"

    def __init__(
        self,
        db_file: str,
        host: str = "127.0.0.1",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        setup: bool = False,
        allow_auto_restarts: bool = False,
        container_name: str = "umbradb",
        container_image: str = "umbradb/umbra:latest",
        container_data_dir: Path | None = None,
        container_num_cores: int = 1,
        container_pin_core_id: int = 4,
        container_memory_gb: float = 0.0,
    ) -> None:
        self._db_file = db_file
        # Derive the benchmark identifier (→ Umbra database name) from the DuckDB
        # filename, e.g. accidents.duckdb → "accidents".  IMDB keeps its curated,
        # hand-written schema (so the already-loaded imdb_sf1 DB is untouched);
        # every other dataset has its table list + CREATE TABLE DDL introspected
        # straight from its DuckDB file.
        stem = Path(db_file).stem.lower()
        self._benchmark = "".join(c if (c.isalnum() or c == "_") else "_" for c in stem)
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        if self._benchmark == "imdb":
            self._tables = _umbra_get_tables()
            self._schema_sql = _umbra_get_schema()
        else:
            self._tables, self._schema_sql = _derive_umbra_schema_from_duckdb(db_file)
            print(
                f"UmbraRunner: derived schema for benchmark '{self._benchmark}' "
                f"from {db_file} ({len(self._tables)} tables)."
            )
        self._loaded_sf: float | None = None
        self._db_names: dict[float, str] = {}
        self._conns: dict[float, object] = {}
        self._con = None
        self._duckdb_con = None
        self._container_name = container_name
        self._container_image = container_image
        self._scale_factors = [1.0]  # IMDB has a single scale factor
        self._container_data_dir = (
            container_data_dir if container_data_dir is not None
            else (Path.home() / "umbra-db")
        )
        self._container_data_dir.mkdir(parents=True, exist_ok=True)

        self._container_num_cores = container_num_cores
        self._container_pin_core_id = container_pin_core_id
        self._container_memory_gb = container_memory_gb

        if self._container_num_cores != 1:
            raise ValueError("UmbraRunner currently only supports container_num_cores=1.")
        if self._container_pin_core_id < 0:
            raise ValueError("container_pin_core_id must be >= 0.")

        self._allow_auto_restarts = allow_auto_restarts
        self.setup_done = False
        if setup:
            self.setup()

    def setup(self):
        import psycopg2

        if self.setup_done:
            print("UmbraRunner: setup already done, skipping.")
            return

        if self._allow_auto_restarts:
            self._restart_container(query_mode=False)

        try:
            self._admin_con = psycopg2.connect(
                host=self._host, port=self._port,
                user=self._user, password=self._password,
                dbname="postgres",
            )
        except psycopg2.OperationalError as exc:
            print(f"UmbraRunner: failed to connect ({exc}); attempting container restart.")
            self._restart_container(query_mode=False)
            self._admin_con = psycopg2.connect(
                host=self._host, port=self._port,
                user=self._user, password=self._password,
                dbname="postgres",
            )

        self._admin_con.autocommit = True
        print(f"UmbraRunner: connected to Umbra at {self._host}:{self._port}")

        for scale_factor in self._scale_factors:
            self._load_sf(scale_factor)

        if self._allow_auto_restarts:
            self._restart_container(query_mode=True)
            try:
                self._admin_con.close()
            except Exception:
                pass
            self._admin_con = psycopg2.connect(
                host=self._host, port=self._port,
                user=self._user, password=self._password,
                dbname="postgres",
            )
            self._admin_con.autocommit = True

            for con in self._conns.values():
                try:
                    con.close()
                except Exception:
                    pass
            self._conns.clear()
            self._con = None
            self._loaded_sf = None

        self.setup_done = True

    @staticmethod
    def _format_sf(scale_factor: float) -> str:
        if int(scale_factor) == scale_factor:
            return str(int(scale_factor))
        return str(scale_factor).replace(".", "_")

    def _db_name_for_sf(self, scale_factor: float) -> str:
        return f"{self._benchmark}_sf{self._format_sf(scale_factor)}"

    def _load_sf(self, scale_factor: float, verbose: bool = True) -> None:
        import psycopg2

        db_name = self._db_name_for_sf(scale_factor)
        self._db_names[scale_factor] = db_name
        tables = self._tables

        con = None
        if self._database_exists(db_name):
            con = psycopg2.connect(
                host=self._host, port=self._port,
                user=self._user, password=self._password,
                dbname=db_name,
            )
            con.autocommit = True
            if self._has_all_tables_with_data(con, tables):
                self._conns[scale_factor] = con
                if verbose:
                    print(f"UmbraRunner: reusing existing SF{scale_factor} database '{db_name}' (tables already loaded).")
                return

            print(f"UmbraRunner: existing database '{db_name}' missing data; reloading tables.")
            self._reset_existing_tables(con, tables)
        else:
            admin_cur = self._admin_con.cursor()
            admin_cur.execute(f"CREATE DATABASE {db_name}")
            admin_cur.close()

            con = psycopg2.connect(
                host=self._host, port=self._port,
                user=self._user, password=self._password,
                dbname=db_name,
            )
            con.autocommit = True

        assert con is not None
        cur = con.cursor()
        cur.execute(self._schema_sql)

        for table in tqdm(tables, desc=f"Loading Umbra tables for SF{scale_factor} ({db_name})"):
            self._copy_table_from_duckdb(cur=cur, table=table)

        cur.close()
        self._conns[scale_factor] = con
        print(f"UmbraRunner: loaded SF{scale_factor} into database '{db_name}'.")

    def _copy_table_from_duckdb(self, cur, table: str) -> None:
        """Export a table from the DuckDB file to a temporary CSV, then COPY into Umbra."""
        if self._duckdb_con is None:
            self._duckdb_con = duckdb.connect(database=self._db_file, read_only=True)

        host_stage_dir = self._container_data_dir / ".ingest_tmp"
        host_stage_dir.mkdir(parents=True, exist_ok=True)

        file_id = uuid.uuid4().hex
        host_csv = host_stage_dir / f"{table}_{file_id}.csv"
        container_csv = f"/var/db/.ingest_tmp/{host_csv.name}"

        host_csv_literal = host_csv.as_posix().replace("'", "''")
        container_csv_literal = container_csv.replace("'", "''")

        try:
            self._duckdb_con.execute(
                f'COPY (SELECT * FROM "{table}") TO \'{host_csv_literal}\' '
                "(FORMAT CSV, HEADER FALSE, NULL '')"
            )
            quoted_table = '"' + table.replace('"', '""') + '"'
            cur.execute(
                f"COPY {quoted_table} FROM '{container_csv_literal}' "
                "(FORMAT CSV, HEADER FALSE)"
            )
        finally:
            try:
                os.remove(host_csv)
            except FileNotFoundError:
                pass

    def _build_run_cmd(self, query_mode: bool) -> list[str]:
        cmd = ["docker", "run", "-d", "--name", self._container_name]
        if query_mode:
            cmd.extend(["--cpus", str(self._container_num_cores),
                        "--cpuset-cpus", str(self._container_pin_core_id)])
        cmd.extend([
            "-v", f"{self._container_data_dir.as_posix()}:/var/db",
            "-p", f"{self._port}:5432",
            "--ulimit", "nofile=1048576:1048576",
            "--ulimit", "memlock=8388608:8388608",
        ])
        # Hard RAM cap (== --memory-swap so there's no swap headroom): a runaway
        # query fails via cgroup OOM inside the container rather than exhausting
        # the host. Kept in sync with umbra_setup.py's _memory_flags.
        if self._container_memory_gb and self._container_memory_gb > 0:
            mb = int(round(self._container_memory_gb * 1024))
            cmd.extend(["--memory", f"{mb}m", "--memory-swap", f"{mb}m"])
        cmd.append(self._container_image)
        return cmd

    def _restart_container(self, query_mode: bool) -> None:
        if self._host not in {"127.0.0.1", "localhost"}:
            return

        def _run(cmd):
            return subprocess.run(cmd, check=False, capture_output=True, text=True)

        rm = _run(["docker", "rm", "-f", self._container_name])
        print(f"docker rm: {rm.stdout.strip()} / {rm.stderr.strip()}")
        run_cmd = self._build_run_cmd(query_mode=query_mode)
        run = _run(run_cmd)
        if run.returncode != 0:
            raise RuntimeError(
                f"Failed to restart Umbra container '{self._container_name}': "
                f"{run.stderr.strip()}"
            )
        print(f"UmbraRunner: restarted container in {'query' if query_mode else 'load'} mode.")
        self._wait_until_ready()

    def _wait_until_ready(self, timeout_s: float = 30.0) -> None:
        import psycopg2

        deadline = time.time() + timeout_s
        last_err: str | None = None
        while time.time() < deadline:
            try:
                con = psycopg2.connect(
                    host=self._host, port=self._port,
                    user=self._user, password=self._password,
                    dbname="postgres",
                )
                con.close()
                return
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.5)
        raise RuntimeError(
            "Timed out waiting for Umbra to become ready."
            + (f" Last error: {last_err}" if last_err else "")
        )

    def _database_exists(self, db_name: str) -> bool:
        cur = self._admin_con.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone() is not None
        cur.close()
        return exists

    @staticmethod
    def _has_all_tables_with_data(con, tables: list[str]) -> bool:
        cur = con.cursor()
        try:
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                cur.execute(f"SELECT 1 FROM {quoted} LIMIT 1")
                if cur.fetchone() is None:
                    return False
            return True
        except Exception:
            return False
        finally:
            cur.close()

    @staticmethod
    def _reset_existing_tables(con, tables: list[str]) -> None:
        cur = con.cursor()
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            cur.execute(f"DROP TABLE IF EXISTS {quoted}")
        cur.close()

    def _switch_sf(self, scale_factor: float) -> None:
        if self._loaded_sf == scale_factor:
            return
        if scale_factor not in self._conns:
            self._load_sf(scale_factor, verbose=False)
        self._con = self._conns[scale_factor]
        self._loaded_sf = scale_factor
        print(f"UmbraRunner: switched to SF{scale_factor} database '{self._db_names[scale_factor]}'.")


# ── Module-level singleton (None = DuckDB path active) ───────────────────────
_umbra: "UmbraRunner | None" = None

# Warmup mode: "per_query" (default — 1 warmup before each measured query),
# "startup_only" (no per-query warmup; cache primed once via --mode warmup),
# "startup_and_per_query" (both: one-time prime AND per-query warmup),
# or "none" (no warmup at all). Set in main() from --warmup-mode.
_WARMUP_MODE: str = "per_query"

# DuckDB spill cap. A runaway query (e.g. a base-table validation query whose
# rule joins under-constrain the join, yielding a near-cartesian product) can
# spill tens of GB to <db>.tmp/ and get the whole process OOM/disk killed —
# which is uncatchable and leaves no output file. Bounding the temp directory
# turns that fatal kill into a normal DuckDB error that execute_query's
# try/except handles, so the run completes and writes its own result file.
# "0" disables the cap. Set in main() from --max-temp-size.
_MAX_TEMP_SIZE: str = "0"

# Per-query wall-clock timeout (seconds). A single query run exceeding this is
# interrupted (con.interrupt() from a watchdog thread), which raises and is
# handled like any execution failure — so the offending rule is skipped and
# discarded rather than hanging the whole run. 0 = no timeout. Set in main()
# from --query-timeout.
_QUERY_TIMEOUT_S: float = 0.0

# DuckDB scheduler threads granted to each connection. Pinned to 1 for the
# (serial, node-exclusive) performance-measurement run so timing is reproducible.
# During parallel validation it is raised to the worker count: `SET threads` is a
# *global* instance setting, so leaving it at 1 would serialize concurrent
# queries and defeat the pool. Timing reproducibility is irrelevant for
# validation (correctness-only). Set in main() from --workers.
_DUCKDB_THREADS: int = 1

# Per-thread profiling-output filename. execute_query writes DuckDB profiling to
# this file and reads it back; under the validation ThreadPoolExecutor each
# worker needs its own file or concurrent runs clobber each other. Defaults to
# "evaluation.json" on the main thread.
_profile_local = threading.local()
_worker_counter = itertools.count()


def _current_profile_file() -> str:
    return getattr(_profile_local, "path", "evaluation.json")


def _init_validation_worker() -> None:
    """ThreadPoolExecutor initializer: give each worker a unique profile file."""
    _profile_local.path = f"evaluation_{next(_worker_counter)}.json"


def _per_query_warmup_enabled() -> bool:
    return _WARMUP_MODE in ("per_query", "startup_and_per_query")


def _apply_duckdb_guards(con) -> None:
    """Apply the spill cap to a fresh DuckDB connection (no-op when '0')."""
    if _MAX_TEMP_SIZE and _MAX_TEMP_SIZE != "0":
        con.execute(f"SET max_temp_directory_size = '{_MAX_TEMP_SIZE}'")


class QueryExecutionError(RuntimeError):
    """The engine could not execute a query at all (parse/bind/timeout/runtime).

    Distinct from a query that legitimately returns zero rows. The backends used
    to report both as ``[], 0.0, {}, []``, which let an unexecutable query be
    recorded as a 0.0s run whose (empty) output "matched" the original — a
    silent no-op measurement rather than a visible failure.
    """


# Every query that could not be executed, reported as a block at end of run so a
# failure cannot scroll past unnoticed in a long log.
_EXECUTION_ERRORS: list[dict] = []


def _log_execution_error(sql_name: str, stage: str, error: Exception) -> dict:
    """Print a failed query loudly and queue it for the end-of-run summary."""
    message = str(error)
    print(f"ERROR: {sql_name}: {stage} could not be executed — excluded from results.\n"
          f"  {message}")
    _EXECUTION_ERRORS.append({"query": sql_name, "stage": stage, "message": message})
    return {"stage": stage, "message": message}


def _report_execution_errors() -> None:
    """Summarise every unexecutable query at end of run."""
    if not _EXECUTION_ERRORS:
        return
    print(f"\n{'=' * 70}\n{len(_EXECUTION_ERRORS)} query/queries could not be executed "
          f"and were excluded from the results:")
    for rec in _EXECUTION_ERRORS:
        print(f"  - {rec['query']} ({rec['stage']}): {rec['message'].splitlines()[0]}")
    print("=" * 70)


# DuckDB >= 1.4 parses AT as a keyword (the time-travel clause), so the JOB
# workload's `aka_title AS at` no longer reads as an alias and 15a-15d fail to
# parse. Quote such identifiers on the way to DuckDB only: the workload text and
# everything we record stay untouched, so the queries remain byte-identical
# across the DuckDB/Postgres/Umbra backends.
_DUCKDB_RESERVED_ALIASES = frozenset({"at"})


def _quote_reserved_aliases(query: str) -> str:
    """Double-quote identifiers DuckDB treats as reserved keywords.

    Rewrites only standalone identifier tokens, skipping string literals and
    already-quoted identifiers, so a literal such as '%at%' is left alone.
    """
    out: list[str] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch in ("'", '"'):
            # Copy the literal/quoted identifier verbatim ('' and "" escape).
            j = i + 1
            while j < n:
                if query[j] == ch:
                    if j + 1 < n and query[j + 1] == ch:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(query[i:j])
            i = j
            continue
        if ch.isalnum() or ch == "_":
            j = i
            while j < n and (query[j].isalnum() or query[j] == "_"):
                j += 1
            word = query[i:j]
            out.append(f'"{word}"' if word.lower() in _DUCKDB_RESERVED_ALIASES else word)
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _execute_with_timeout(con, query):
    """Run ``con.execute(query).fetchall()`` under the per-query timeout.

    When _QUERY_TIMEOUT_S is set, a watchdog thread calls con.interrupt() after
    the deadline, causing the in-flight query to raise. We re-raise it as a
    TimeoutError with a clear message so the caller's except clause discards the
    rule. With no timeout configured this is a plain passthrough.
    """
    query = _quote_reserved_aliases(query)
    if not _QUERY_TIMEOUT_S or _QUERY_TIMEOUT_S <= 0:
        return con.execute(query).fetchall()

    state = {"timed_out": False}

    def _interrupt():
        state["timed_out"] = True
        try:
            con.interrupt()
        except Exception:
            pass

    timer = threading.Timer(_QUERY_TIMEOUT_S, _interrupt)
    timer.start()
    try:
        return con.execute(query).fetchall()
    except Exception:
        if state["timed_out"]:
            raise TimeoutError(
                f"query exceeded {_QUERY_TIMEOUT_S:.0f}s timeout — skipping rule"
            )
        raise
    finally:
        timer.cancel()


@contextmanager
def _umbra_timeout_guard():
    """Per-statement wall-clock timeout for Umbra, analogous to _execute_with_timeout.

    A watchdog thread calls psycopg2's thread-safe connection.cancel()
    (libpq PQcancel) once _QUERY_TIMEOUT_S elapses, so the in-flight statement
    is cancelled server-side and raises; we re-raise it as TimeoutError so the
    caller's except clause rolls back and discards the rule (same handling as
    the DuckDB path).  Arm this around each individual execute — so every run,
    including each rule-subset combination in the powerset, gets its own fresh
    budget rather than a single deadline spanning the whole per-query
    evaluation.  With no timeout configured this is a plain passthrough.

    If Umbra does not honour the wire-protocol cancel this degrades gracefully:
    the statement simply runs to natural completion (no crash, no false skip).
    """
    if not _QUERY_TIMEOUT_S or _QUERY_TIMEOUT_S <= 0:
        yield
        return

    state = {"timed_out": False}

    def _cancel():
        state["timed_out"] = True
        try:
            if _umbra is not None:
                _umbra._con.cancel()
        except Exception:
            pass

    timer = threading.Timer(_QUERY_TIMEOUT_S, _cancel)
    timer.start()
    try:
        yield
    except Exception:
        if state["timed_out"]:
            raise TimeoutError(
                f"query exceeded {_QUERY_TIMEOUT_S:.0f}s timeout — skipping rule"
            )
        raise
    finally:
        timer.cancel()


def init_umbra(db_file: str, port: int = 5432, memory_gb: float = 0.0) -> None:
    """Initialize the global UmbraRunner.  Called once from main() when --engine umbra."""
    global _umbra
    _umbra = UmbraRunner(
        db_file=db_file,
        host="127.0.0.1",
        port=port,
        container_name=_UMBRA_CONTAINER,
        container_data_dir=_UMBRA_DATA_DIR,
        container_pin_core_id=4,
        container_memory_gb=memory_gb,
        allow_auto_restarts=False,   # container pre-started by umbra_setup.py
        setup=True,                  # connects + loads data (idempotent)
    )
    _umbra._switch_sf(1.0)           # sets _umbra._con to the imdb_sf1 database


# ── Umbra query execution: timing + EXPLAIN ANALYZE plan ────────────────────

def _run_query_umbra(query: str, attempts: int = 3) -> tuple[list, float, dict, list[float]]:
    """Execute query on Umbra; return (result_rows, median_time_s, plan_dict).

    - Timing: derived from Umbra's EXPLAIN ANALYZE pipeline durations (engine-internal).
    - Plan: 1 warmup + N EXPLAIN ANALYZE runs; per-pipeline median durations.
    - median_time_s = sum of median pipeline durations.
    - Returns time in seconds (same unit as DuckDB's latency field).
    """
    assert _umbra is not None
    try:
        cur = _umbra._con.cursor()

        # Plain execute for result rows (also warms buffer pool)
        with _umbra_timeout_guard():
            cur.execute(query)
            result = cur.fetchall()

        # EXPLAIN ANALYZE: optional warmup (discarded) + N measured runs
        if _per_query_warmup_enabled():
            _umbra_run_explain_analyze_once(query, cur)  # warmup

        runs: list[tuple[dict, dict[int, float], tuple]] = []
        for _ in range(attempts):
            runs.append(_umbra_run_explain_analyze_once(query, cur))

        cur.close()

        # Group runs by pipeline signature to detect plan changes
        sig_counts = Counter(sig for _, _, sig in runs)
        majority_sig, majority_count = sig_counts.most_common(1)[0]

        if len(sig_counts) > 1:
            print(f"Warning: Umbra produced {len(sig_counts)} different plan variants "
                  f"across {attempts} runs for query; using majority variant "
                  f"({majority_count}/{attempts} runs)")

        # Filter to runs matching the majority plan variant
        majority_runs = [(pj, dur) for pj, dur, sig in runs if sig == majority_sig]
        plan_json = majority_runs[0][0]
        majority_durations = [dur for _, dur in majority_runs]

        # Compute median duration per pipeline
        pipeline_indices = sorted(majority_durations[0].keys())
        median_pipeline_durations: dict[int, float] = {}
        for idx in pipeline_indices:
            values = sorted(d.get(idx, 0.0) for d in majority_durations)
            median_pipeline_durations[idx] = values[len(values) // 2]

        # Total time = sum of median pipeline durations
        median_time_s = sum(median_pipeline_durations.values())

        # Per-run total times (sum of pipeline durations within each run) for variance reporting
        run_totals = [sum(d.values()) for d in majority_durations]

        # Build plan_dict tree using median durations
        plan_dict = _umbra_explain_analyze(plan_json, median_pipeline_durations, median_time_s)

        return result, median_time_s, plan_dict, run_totals
    except Exception as e:
        try:
            _umbra._con.rollback()
        except Exception:
            pass
        raise QueryExecutionError(f"Umbra could not execute query: {e}\nFor query: {query}") from e



# ── Umbra plan parsing: structured JSON format ───────────────────────────────
#
# Umbra supports EXPLAIN (ANALYZE, FORMAT JSON) which returns a properly nested
# JSON tree.  Each node has:
#   - "operator" / "physicalOperator": operator type
#   - "analyzePlanCardinality": actual row count (from ANALYZE)
#   - "cardinality": estimated row count
#   - "tablename": table name (tablescan nodes)
#   - "condition": join condition (expression tree)
#   - "restrictions": filter predicates (list of expression trees)
#   - "type": join type ("inner", "left", ...)
#   - "left" / "right": children of binary operators
#   - "input": child of unary operators (sort, limit, ...)
#
# Per-operator timing is not available in this format; operator_timing is set
# to 0.0 (heat coloring in plan graphs will be flat/green).


def _fmt_umbra_expr(expr: dict) -> str:
    """Recursively format Umbra's expression JSON into a human-readable string."""
    etype = expr.get("expression", "")
    if etype == "compare":
        left = _fmt_umbra_expr(expr["left"])
        right = _fmt_umbra_expr(expr["right"])
        return f"{left} {expr.get('direction', '=')} {right}"
    elif etype == "iuref":
        return expr.get("iu", "?")
    elif etype == "const":
        return str(expr.get("value", {}).get("value", "?"))
    elif etype == "and":
        parts = [_fmt_umbra_expr(e) for e in expr.get("arguments", [])]
        return " AND ".join(parts)
    elif etype == "or":
        parts = [_fmt_umbra_expr(e) for e in expr.get("arguments", [])]
        return " OR ".join(parts)
    else:
        return str(expr)


def _convert_umbra_node(node: dict, pipeline_map: dict[int, float],
                        pipeline_id_map: dict[int, int] | None = None) -> dict:
    """Recursively convert one Umbra plan JSON node to a DuckDB-compatible dict.

    pipeline_map: analyzePlanId → pipeline duration in seconds.
    pipeline_id_map: analyzePlanId → pipeline index (for grouping nodes by pipeline).
    operator_timing is set to the enclosing pipeline's wall-clock duration —
    the best approximation available (Umbra does not expose per-operator time).
    Operators sharing a pipeline will show the same timing value.
    """
    operator = node.get("operator", "unknown")
    phys = node.get("physicalOperator", "")
    op_name = f"{operator} ({phys})" if phys and phys != operator else operator

    extra: dict = {}
    est = node.get("cardinality")
    if est is not None:
        extra["Estimated Cardinality"] = est
    if "tablename" in node:
        extra["Table"] = node["tablename"]
    if operator == "join" and "type" in node:
        extra["Join Type"] = node["type"]
    if "condition" in node:
        extra["Conditions"] = _fmt_umbra_expr(node["condition"])
    if node.get("restrictions"):
        extra["Filters"] = [_fmt_umbra_expr(r) for r in node["restrictions"]]

    children = []
    for key in ("left", "right", "input"):
        child = node.get(key)
        if isinstance(child, dict):
            children.append(_convert_umbra_node(child, pipeline_map, pipeline_id_map))

    analyze_id = node.get("analyzePlanId")
    timing = pipeline_map.get(analyze_id, 0.0) if analyze_id is not None else 0.0
    pipeline_id = (pipeline_id_map.get(analyze_id) if pipeline_id_map and analyze_id is not None else None)

    result = {
        "operator_name": op_name,
        "operator_timing": timing,
        "operator_cardinality": node.get("analyzePlanCardinality", 0),
        "operator_rows_scanned": node.get("analyzePlanCardinality", 0),
        "extra_info": extra,
        "children": children,
    }
    if pipeline_id is not None:
        result["pipeline_id"] = pipeline_id
    return result


def _umbra_run_explain_analyze_once(query: str, cur) -> tuple[dict, dict[int, float], tuple]:
    """Run one EXPLAIN (ANALYZE, FORMAT JSON) and return (plan_json, pipeline_durations, signature).

    pipeline_durations: pipeline_index → duration in seconds.
    signature: hashable key identifying the pipeline structure (for detecting plan changes).
    Does not catch exceptions — caller handles that.
    """
    with _umbra_timeout_guard():
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
        rows = cur.fetchall()
    raw = rows[0][0] if rows else ""
    plan_json = json.loads(raw) if isinstance(raw, str) else raw

    pipeline_durations: dict[int, float] = {}
    sig_parts: list[frozenset] = []
    for idx, p in enumerate(plan_json.get("analyzePlanPipelines", [])):
        pipeline_durations[idx] = p["duration"] / 1_000_000  # μs → s
        sig_parts.append(frozenset(p["operators"]))

    signature = (len(sig_parts), tuple(sig_parts))
    return plan_json, pipeline_durations, signature


def _umbra_explain_analyze(plan_json: dict, median_pipeline_durations: dict[int, float],
                           total_time_s: float) -> dict:
    """Build a DuckDB-compatible profile dict from pre-computed plan and median pipeline durations."""
    fallback = {
        "latency": total_time_s,
        "operator_name": "QUERY_ROOT",
        "operator_timing": total_time_s,
        "operator_cardinality": 0,
        "operator_rows_scanned": 0,
        "extra_info": {},
        "children": [],
    }
    try:
        # Build analyzePlanId → pipeline duration map and pipeline membership map.
        # operators[] in each pipeline entry contains analyzePlanId values (0-based),
        # not operatorId (1-based) — verified from EXPLAIN output.
        pipeline_map: dict[int, float] = {}
        pipeline_id_map: dict[int, int] = {}
        for idx, p in enumerate(plan_json.get("analyzePlanPipelines", [])):
            dur_s = median_pipeline_durations.get(idx, 0.0)
            for analyze_id in p["operators"]:
                pipeline_map[analyze_id] = dur_s
                pipeline_id_map[analyze_id] = idx

        root = _convert_umbra_node(plan_json["plan"], pipeline_map, pipeline_id_map)
        root["latency"] = total_time_s
        return root
    except Exception as exc:
        print(f"Umbra plan JSON parse failed: {exc}")
        return fallback


# ── Umbra cost estimation: EXPLAIN only (no execution) ──────────────────────

def _sum_umbra_cardinality(node: dict) -> float:
    """Recursively sum estimated cardinalities from an Umbra JSON plan node."""
    total = node.get("cardinality", 0)
    for key in ("left", "right", "input"):
        child = node.get(key)
        if isinstance(child, dict):
            total += _sum_umbra_cardinality(child)
    return total


def _estimate_cost_umbra(query: str) -> tuple[float, dict]:
    """Run EXPLAIN (FORMAT JSON) (no execution) and return a cardinality-based cost estimate.

    Uses Umbra's structured JSON plan to sum estimated cardinalities across all
    operators as a cost proxy (analogous to DuckDB's weighted cardinality sum).

    Returns (cost, plan_json).  On failure returns (float('inf'), {}).
    """
    assert _umbra is not None
    try:
        cur = _umbra._con.cursor()
        cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
        rows = cur.fetchall()
        cur.close()
        raw = rows[0][0] if rows else ""
        plan_json = json.loads(raw) if isinstance(raw, str) else raw

        plan_node = plan_json.get("plan")
        if plan_node:
            return _sum_umbra_cardinality(plan_node), plan_json
        return float("inf"), plan_json
    except Exception as exc:
        print(f"Umbra EXPLAIN failed: {exc}")
        print(f"  For query: {query[:200]}")
        return float("inf"), {}


# ===========================================================================
# PostgreSQL (pg_lab) execution engine support (inline, self-contained)
# ===========================================================================
#
# pg_lab is a research fork of PostgreSQL 16 that accepts hint comments of
# the form `/*=pg_lab= JoinOrder((a b) c) */` — used below to enforce a
# specific join tree on refined queries (fix_join_order).
#
# Data ingestion mirrors UmbraRunner: CSV exported from imdb.duckdb on the
# host, then COPY'd into the target database through a shared bind-mount.
# The container's bind-mount lives at /pg_lab (image convention) on the
# container side and ~/pg_lab-db on the host (matches pg_lab_setup.py).
# ---------------------------------------------------------------------------

_POSTGRES_CONTAINER = "pg_lab"
_POSTGRES_DATA_DIR = Path.home() / "pg_lab-db"
_POSTGRES_STAGE_SUBDIR = ".ingest_tmp"
# In-container path to the bind-mounted staging directory (for COPY FROM).
_POSTGRES_STAGE_IN_CONTAINER = "/pg_lab/.ingest_tmp"


class PostgresRunner:
    """Manages a pg_lab Postgres connection and idempotent IMDB loading.

    The container lifecycle is handled externally by `pg_lab_setup.py`;
    this class only connects, creates the target database if missing,
    and loads IMDB tables from an existing DuckDB file via CSV COPY.
    """

    name = "Postgres"

    def __init__(
        self,
        db_file: str,
        host: str = "127.0.0.1",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        dbname: str = "imdb",
        container_data_dir: Path | None = None,
    ) -> None:
        self._db_file = db_file
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._dbname = dbname
        self._schema_sql = _umbra_get_schema()
        self._container_data_dir = (
            container_data_dir if container_data_dir is not None
            else _POSTGRES_DATA_DIR
        )
        self._container_data_dir.mkdir(parents=True, exist_ok=True)
        self._con = None
        self._duckdb_con = None
        self.setup_done = False

    def setup(self) -> None:
        import psycopg2

        if self.setup_done:
            return

        admin_con = psycopg2.connect(
            host=self._host, port=self._port,
            user=self._user, password=self._password,
            dbname="postgres",
        )
        admin_con.autocommit = True
        try:
            cur = admin_con.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self._dbname,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{self._dbname}"')
                print(f"PostgresRunner: created database '{self._dbname}'.")
            cur.close()
        finally:
            admin_con.close()

        self._con = psycopg2.connect(
            host=self._host, port=self._port,
            user=self._user, password=self._password,
            dbname=self._dbname,
        )
        self._con.autocommit = True
        print(f"PostgresRunner: connected to {self._host}:{self._port}/{self._dbname}")

        tables = _umbra_get_tables()
        cur = self._con.cursor()
        try:
            if UmbraRunner._has_all_tables_with_data(self._con, tables):
                print(f"PostgresRunner: database '{self._dbname}' already populated; "
                      f"skipping load.")
            else:
                # Reset any partial schema and reload
                for table in tables:
                    cur.execute(f'DROP TABLE IF EXISTS "{table}"')
                cur.execute(self._schema_sql)
                for table in tqdm(tables, desc=f"Loading Postgres tables into '{self._dbname}'"):
                    self._copy_table_from_duckdb(cur=cur, table=table)
                cur.execute("ANALYZE")
                print(f"PostgresRunner: loaded {len(tables)} tables and ran ANALYZE.")
            self._ensure_primary_keys(cur, tables)
            self._ensure_numeric_indexes(cur)
        finally:
            cur.close()

        self.setup_done = True

    def _ensure_primary_keys(self, cur, tables: list[str]) -> None:
        # Every IMDB table's PK is `id` — the schema-creation DDL declares
        # it NOT NULL but omits the constraint (Umbra-compatible).
        cur.execute("""
            SELECT t.table_name
            FROM information_schema.table_constraints t
            WHERE t.table_schema = 'public'
              AND t.constraint_type = 'PRIMARY KEY'
        """)
        have_pk = {row[0] for row in cur.fetchall()}

        to_add = [t for t in tables if t not in have_pk]
        if not to_add:
            print(f"PostgresRunner: primary keys already present "
                  f"({len(tables)} tables).")
            return

        for table in tqdm(to_add, desc=f"Adding primary keys in '{self._dbname}'"):
            cur.execute(f'ALTER TABLE "{table}" ADD PRIMARY KEY ("id")')
        cur.execute("ANALYZE")
        print(f"PostgresRunner: added {len(to_add)} primary keys "
              f"({len(tables) - len(to_add)} already existed).")

    def _ensure_numeric_indexes(self, cur) -> None:
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name <> 'id'
              AND data_type IN (
                  'smallint', 'integer', 'bigint',
                  'real', 'double precision', 'numeric'
              )
            ORDER BY table_name, ordinal_position
        """)
        targets = cur.fetchall()

        cur.execute("""
            SELECT indexname FROM pg_indexes WHERE schemaname = 'public'
        """)
        existing = {row[0] for row in cur.fetchall()}

        to_create = [(t, c, f"idx_{t}_{c}") for t, c in targets
                     if f"idx_{t}_{c}" not in existing]
        if not to_create:
            print(f"PostgresRunner: numeric-column indexes already present "
                  f"({len(targets)} columns).")
            return

        for table, col, idx in tqdm(to_create,
                                    desc=f"Building indexes in '{self._dbname}'"):
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "{idx}" '
                f'ON "{table}" ("{col}")'
            )
        cur.execute("ANALYZE")
        print(f"PostgresRunner: built {len(to_create)} new indexes "
              f"({len(targets) - len(to_create)} already existed).")

    def _copy_table_from_duckdb(self, cur, table: str) -> None:
        """Export a table from imdb.duckdb and stream it into Postgres via STDIN.

        Streaming (not a shared file) avoids permissions headaches — the pg_lab
        container's bind-mount is owned by the container uid, so a host-side
        staging dir inside the mount is not writable by the pipeline user.
        Instead, duckdb writes CSV to a host-local tempfile and psycopg2's
        copy_expert pushes it to the server over the existing TCP connection.
        """
        import tempfile

        if self._duckdb_con is None:
            self._duckdb_con = duckdb.connect(database=self._db_file, read_only=True)

        fd, path = tempfile.mkstemp(prefix=f"{table}_", suffix=".csv")
        os.close(fd)
        try:
            path_literal = path.replace("'", "''")
            self._duckdb_con.execute(
                f'COPY (SELECT * FROM "{table}") TO \'{path_literal}\' '
                "(FORMAT CSV, HEADER FALSE, NULL '')"
            )
            with open(path, "rb") as f:
                cur.copy_expert(
                    f'COPY "{table}" FROM STDIN WITH (FORMAT csv, HEADER false, NULL \'\')',
                    f,
                )
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


# ── Module-level singleton (None = not active) ──────────────────────────────
_postgres: "PostgresRunner | None" = None


def init_postgres(
    db_file: str,
    host: str = "127.0.0.1",
    port: int = 5432,
    user: str = "postgres",
    password: str = "postgres",
    dbname: str = "imdb",
) -> None:
    """Initialize the global PostgresRunner.  Called once from main() when --engine postgres."""
    global _postgres
    _postgres = PostgresRunner(
        db_file=db_file,
        host=host, port=port, user=user, password=password, dbname=dbname,
        container_data_dir=_POSTGRES_DATA_DIR,
    )
    _postgres.setup()


# ── Postgres query execution: timing + EXPLAIN ANALYZE plan ─────────────────

def _run_query_postgres(query: str, attempts: int = 3) -> tuple[list, float, dict, list[float]]:
    """Execute query on Postgres; return (result_rows, median_time_s, plan_dict).

    Timing is taken from the "Execution Time" field of EXPLAIN (ANALYZE, FORMAT JSON),
    which excludes client-side fetch overhead — comparable to Umbra's pipeline
    duration sum and DuckDB's profiling latency.
    """
    assert _postgres is not None
    try:
        cur = _postgres._con.cursor()

        cur.execute(query)
        result = cur.fetchall()

        # Optional warmup EXPLAIN ANALYZE (discarded)
        if _per_query_warmup_enabled():
            _postgres_run_explain_analyze_once(query, cur)

        runs: list[tuple[dict, float, tuple]] = []
        for _ in range(attempts):
            runs.append(_postgres_run_explain_analyze_once(query, cur))

        cur.close()

        sig_counts = Counter(sig for _, _, sig in runs)
        majority_sig, majority_count = sig_counts.most_common(1)[0]
        if len(sig_counts) > 1:
            print(f"Warning: Postgres produced {len(sig_counts)} different plan variants "
                  f"across {attempts} runs; using majority variant "
                  f"({majority_count}/{attempts})")

        majority = [(pj, t) for pj, t, sig in runs if sig == majority_sig]
        plan_json = majority[0][0]
        majority_times = [t for _, t in majority]
        times = sorted(majority_times)
        median_time_s = times[len(times) // 2]

        plan_dict = _postgres_explain_analyze(plan_json, median_time_s)
        return result, median_time_s, plan_dict, majority_times
    except Exception as e:
        try:
            _postgres._con.rollback()
        except Exception:
            pass
        raise QueryExecutionError(f"Postgres could not execute query: {e}\nFor query: {query}") from e


def _postgres_run_explain_analyze_once(query: str, cur) -> tuple[dict, float, tuple]:
    """Run one EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON); return (plan_json, time_s, signature)."""
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
    rows = cur.fetchall()
    raw = rows[0][0] if rows else []
    if isinstance(raw, str):
        raw = json.loads(raw)
    top = raw[0] if isinstance(raw, list) else raw
    exec_ms = top.get("Execution Time", 0.0)
    time_s = exec_ms / 1000.0
    sig = _postgres_plan_signature(top.get("Plan", {}))
    return top, time_s, sig


def _postgres_plan_signature(node: dict) -> tuple:
    """Structural fingerprint of a Postgres plan (for detecting plan changes across runs)."""
    parts = (node.get("Node Type"), node.get("Relation Name"), node.get("Join Type"))
    children = tuple(_postgres_plan_signature(c) for c in node.get("Plans", []))
    return (parts, children)


# ── Postgres plan parsing: EXPLAIN JSON → DuckDB-compatible dict ────────────

def _postgres_node_conditions(node: dict) -> tuple[str | None, list[str] | None]:
    """Extract human-readable conditions/filters from a Postgres plan node."""
    cond_keys = ("Hash Cond", "Merge Cond", "Join Filter", "Index Cond", "Recheck Cond")
    cond = None
    for k in cond_keys:
        if k in node:
            cond = node[k]
            break
    filters: list[str] = []
    for k in ("Filter", "One-Time Filter"):
        if k in node:
            filters.append(node[k])
    return cond, (filters or None)


def _convert_postgres_node(node: dict) -> dict:
    """Convert one Postgres EXPLAIN JSON node to a DuckDB-compatible dict."""
    op = node.get("Node Type", "unknown")
    strategy = node.get("Strategy") or node.get("Join Type")
    op_name = f"{op} ({strategy})" if strategy else op

    extra: dict = {}
    if "Plan Rows" in node:
        extra["Estimated Cardinality"] = node["Plan Rows"]
    rel = node.get("Relation Name") or node.get("Alias")
    if rel:
        extra["Table"] = rel
    cond, filters = _postgres_node_conditions(node)
    if cond:
        extra["Conditions"] = cond
    if filters:
        extra["Filters"] = filters
    if "Index Name" in node:
        extra["Index Name"] = node["Index Name"]

    # Postgres EXPLAIN ANALYZE reports "Actual Total Time" as a per-loop average
    # that *includes* the time of all child nodes. To match DuckDB/Umbra semantics
    # (where operator_timing is exclusive — per-operator), we compute:
    #     exclusive_ms = ATT * Loops  −  Σ (child.ATT * child.Loops)
    # The Loops factor matters for inner sides of Nested Loops, which run many times.
    loops = node.get("Actual Loops", 1) or 1
    self_total_ms = node.get("Actual Total Time", 0.0) * loops
    children_total_ms = sum(
        (c.get("Actual Total Time", 0.0) * (c.get("Actual Loops", 1) or 1))
        for c in node.get("Plans", [])
    )
    exclusive_ms = max(self_total_ms - children_total_ms, 0.0)
    timing = exclusive_ms / 1000.0  # ms → s
    actual_rows = node.get("Actual Rows", 0)

    return {
        "operator_name": op_name,
        "operator_timing": timing,
        "operator_cardinality": actual_rows,
        "operator_rows_scanned": actual_rows,
        "extra_info": extra,
        "children": [_convert_postgres_node(c) for c in node.get("Plans", [])],
        "pipeline_id": 0,  # Postgres has no pipeline concept; single group
    }


def _postgres_explain_analyze(plan_top: dict, total_time_s: float) -> dict:
    fallback = {
        "latency": total_time_s,
        "operator_name": "QUERY_ROOT",
        "operator_timing": total_time_s,
        "operator_cardinality": 0,
        "operator_rows_scanned": 0,
        "extra_info": {},
        "children": [],
    }
    try:
        root_plan = plan_top.get("Plan")
        if not root_plan:
            return fallback
        root = _convert_postgres_node(root_plan)
        root["latency"] = total_time_s
        # Preserve the raw Postgres EXPLAIN JSON for downstream consumers that
        # need the original shape (Node Type, Plans, Alias) — notably the
        # pg_lab JoinOrder hint builder used by --fix-join-order.
        root["_postgres_raw_plan"] = plan_top
        return root
    except Exception as exc:
        print(f"Postgres plan JSON parse failed: {exc}")
        return fallback


# ── Postgres cost estimation: EXPLAIN only (no execution) ───────────────────

def _estimate_cost_postgres(query: str) -> tuple[float, dict]:
    """Return (total_cost_root_node, plan_json).  On failure (inf, {})."""
    assert _postgres is not None
    try:
        cur = _postgres._con.cursor()
        cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
        rows = cur.fetchall()
        cur.close()
        raw = rows[0][0] if rows else []
        if isinstance(raw, str):
            raw = json.loads(raw)
        top = raw[0] if isinstance(raw, list) else raw
        root_plan = top.get("Plan", {})
        # Prefer planner's own Total Cost (what Postgres actually optimizes).
        cost = root_plan.get("Total Cost")
        if cost is None:
            # Fallback: sum Plan Rows recursively, analogous to Umbra.
            def _sum(n):
                total = n.get("Plan Rows", 0) or 0
                for c in n.get("Plans", []):
                    total += _sum(c)
                return total
            cost = _sum(root_plan)
        return float(cost), top
    except Exception as exc:
        print(f"Postgres EXPLAIN failed: {exc}")
        print(f"  For query: {query[:200]}")
        return float("inf"), {}


# ── pg_lab JoinOrder hint construction (for fix_join_order) ─────────────────

def _pg_collect_join_order(node: dict) -> str | None:
    """Walk a Postgres plan tree and emit a pg_lab JoinOrder() inner expression.

    Scans produce bare aliases; joins produce a parenthesised (left right) pair.
    Returns None if the tree is empty or ill-formed.
    """
    if not isinstance(node, dict):
        return None
    op = node.get("Node Type", "")
    if "Scan" in op:
        alias = node.get("Alias") or node.get("Relation Name")
        return alias
    if "Join" in op or op in {"Nested Loop", "Hash Join", "Merge Join"}:
        children = node.get("Plans", []) or []
        parts = [_pg_collect_join_order(c) for c in children]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            # Single-child join is unusual; fall through to single child.
            return parts[0] if parts else None
        # Left-deep / bushy: emit all children nested left-associatively.
        acc = f"({parts[0]} {parts[1]})"
        for extra in parts[2:]:
            acc = f"({acc} {extra})"
        return acc
    # Unary/other nodes: descend.
    children = node.get("Plans", []) or []
    if len(children) == 1:
        return _pg_collect_join_order(children[0])
    # Multi-child non-join: emit as a grouped sequence.
    parts = [_pg_collect_join_order(c) for c in children]
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    acc = f"({parts[0]} {parts[1]})"
    for extra in parts[2:]:
        acc = f"({acc} {extra})"
    return acc


_PG_JOIN_TOKEN = {
    "Nested Loop": "NestLoop",
    "Hash Join": "HashJoin",
    "Merge Join": "MergeJoin",
}


def _pg_aliases_under(node: dict) -> list[str]:
    """All scan aliases (or relation names) beneath *node*, in plan-tree order."""
    if not isinstance(node, dict):
        return []
    op = node.get("Node Type", "")
    if "Scan" in op:
        a = node.get("Alias") or node.get("Relation Name")
        return [a] if a else []
    out: list[str] = []
    for c in node.get("Plans", []) or []:
        out.extend(_pg_aliases_under(c))
    return out


def _pg_collect_operator_hints(node: dict) -> tuple[list[str], list[str]]:
    """Walk the plan; return (join_hints, scan_hints) as pg_lab directive strings.

    Join hints take the form `<JoinAlgo>(a b ... n)` covering all leaf aliases
    beneath the join — pg_lab matches the directive against any intermediate
    that joins exactly that set of relations.  Scan hints are per-base-relation.
    """
    join_hints: list[str] = []
    scan_hints: list[str] = []

    def visit(n: dict) -> None:
        if not isinstance(n, dict):
            return
        op = n.get("Node Type", "")
        if "Scan" in op:
            alias = n.get("Alias") or n.get("Relation Name")
            if alias:
                if op == "Seq Scan":
                    scan_hints.append(f"SeqScan({alias})")
                elif op in ("Index Scan", "Index Only Scan"):
                    scan_hints.append(f"IdxScan({alias})")
                elif op == "Bitmap Heap Scan":
                    # BitmapScan covers the heap+index pair; emit once at the heap.
                    scan_hints.append(f"BitmapScan({alias})")
                # Other scan types (CTE/Function/Values/Subquery) have no
                # pg_lab token — leave the optimizer free.
        elif op in _PG_JOIN_TOKEN:
            aliases = _pg_aliases_under(n)
            if len(aliases) >= 2:
                token = _PG_JOIN_TOKEN[op]
                join_hints.append(f"{token}({' '.join(aliases)})")
        for c in n.get("Plans", []) or []:
            visit(c)

    visit(node)
    return join_hints, scan_hints


def build_pg_lab_join_order_hint(postgres_plan: dict) -> str | None:
    """Build a `/*=pg_lab= ... */` hint pinning the original plan's join order
    (and, implicitly, the inner/outer i.e. build/probe side of each join, which
    pg_lab's JoinOrder directive enforces) plus the join algorithm of every
    join.  Base-table scan strategies are deliberately left free, so the
    injected predicate can still make the planner switch e.g. from a Seq Scan
    to an Index Scan — the access-path effect the predicate is supposed to have.

    Relative to DuckDB's pinning scope (disabled_optimizers=
    'join_order,build_side_probe_side', which fixes order + build/probe only)
    this is one notch stricter: the join algorithms are fixed here as well.

    *postgres_plan* may be the raw EXPLAIN JSON top-level dict (has "Plan") or
    the converted DuckDB-compatible profile dict that stashes the raw plan
    under "_postgres_raw_plan" (see _postgres_explain_analyze).  Returns None
    if the plan has no join structure (single-table query) or walking fails.
    """
    if not isinstance(postgres_plan, dict):
        return None
    if "_postgres_raw_plan" in postgres_plan:
        raw = postgres_plan["_postgres_raw_plan"]
        root_plan = raw.get("Plan") if isinstance(raw, dict) else None
    elif "Plan" in postgres_plan:
        root_plan = postgres_plan["Plan"]
    else:
        root_plan = postgres_plan
    if not isinstance(root_plan, dict):
        return None

    expr = _pg_collect_join_order(root_plan)
    if not expr or not expr.startswith("("):
        # Single table or no joins — nothing to enforce.
        return None

    # Pin the join tree (order + build/probe side) and the join algorithms, but
    # leave the base-table scan strategies to the optimizer (see docstring).
    join_hints, _scan_hints = _pg_collect_operator_hints(root_plan)

    lines = [f" JoinOrder({expr})"]
    lines.extend(f" {h}" for h in join_hints)
    return "/*=pg_lab=\n" + "\n".join(lines) + "\n*/"


def apply_pg_lab_hint(sql: str, hint: str) -> str:
    """Prepend a pg_lab hint comment to *sql*.  pg_lab's parser reads the
    first block comment of the form /*=pg_lab= ... */ in the statement."""
    return f"{hint}\n{sql}"


# ── Single source of truth for the execution-ready SQL ─────────────────────

def build_execution_sql(
    original_sql: str,
    refined_sql: str,
    original_plan: dict | None,
    *,
    engine: str,
    fix_join_order: bool,
) -> tuple[str, bool]:
    """Return ``(exec_sql, plan_pinned)`` for *refined_sql*.

    Applies join-order pinning when requested: a pg_lab hint on Postgres, explicit
    ``JOIN..ON`` reconstruction plus ``disabled_optimizers`` on DuckDB. Umbra has neither.
    The original always runs unpinned.

    Both cost estimation and final execution must call this helper so the
    string they hand to the engine is identical — otherwise the "expected
    cost reduction" reflects a plan the experiment will never run.

    When ``fix_join_order`` is False this is a no-op: it returns
    ``(refined_sql, False)`` and callers run the optimizer normally.

    When ``fix_join_order`` is True the join order MUST be pinned; the only
    successful return is ``(exec_sql, True)`` (pair True with
    ``disable_join_order=True`` when invoking ``execute_query``).  Any inability
    to pin raises ``RuntimeError`` — we never silently fall back to un-pinned
    execution, because that would invalidate the whole run (the measurement
    would reflect a plan the experiment never intended to run).  The sole
    exception is a query with no join structure (single table), for which there
    is genuinely nothing to reorder and ``refined_sql`` is returned as-is.
    """
    if not fix_join_order:
        return refined_sql, False

    if not original_plan:
        raise RuntimeError(
            f"fix_join_order requested but no execution plan is available "
            f"(engine={engine}); cannot pin the join order for query: "
            f"{original_sql[:200]}"
        )

    if engine == "postgres":
        hint = build_pg_lab_join_order_hint(original_plan)
        if hint is None:
            # No join structure in the plan (single-table / no-join query):
            # nothing to pin, so run refined_sql unchanged.
            return refined_sql, True
        return apply_pg_lab_hint(refined_sql, hint), True

    if engine == "duckdb":
        if reconstruct_with_fixed_join_order is None:
            raise RuntimeError(
                "fix_join_order requested but sql_predicate_converter could not "
                "be imported; reconstruct_with_fixed_join_order is unavailable"
            )
        rebuilt = reconstruct_with_fixed_join_order(
            original_sql, refined_sql, original_plan)
        if rebuilt is None:
            raise RuntimeError(
                "fix_join_order requested but join-order reconstruction failed "
                f"(engine=duckdb) for query: {original_sql[:200]}\n"
                "The DuckDB profiling plan could not be mapped back to the SQL. "
                "A common cause is a DuckDB version whose profiling emits "
                "fully-qualified table names ('db.schema.table') that no longer "
                "match the query's bare table names. Pin DuckDB to a compatible "
                "version (see remote_requirements.txt / pyproject.toml)."
            )
        return rebuilt, True

    # Umbra (and any future engine without a join-order pinning mechanism):
    # it cannot honour fix_join_order, so fail loudly instead of ignoring it.
    raise RuntimeError(
        f"fix_join_order requested but engine={engine!r} has no join-order "
        f"pinning mechanism; disable fix_join_order for this engine"
    )


# ===========================================================================
# DuckDB execution
# ===========================================================================

def execute_query(query, db_file="imdb.duckdb", attempts=3, disable_join_order=False):
    # Dispatch to Postgres / Umbra when active
    if _postgres is not None:
        return _run_query_postgres(query, attempts)
    if _umbra is not None:
        return _run_query_umbra(query, attempts)

    con = duckdb.connect(db_file)
    try:
        _apply_duckdb_guards(con)
        # Single-threaded for reproducible timing on the perf-measurement run
        # (_DUCKDB_THREADS == 1); raised to the worker count during parallel
        # validation so concurrent queries don't serialize on the global setting.
        con.execute(f"SET threads TO {_DUCKDB_THREADS}")
        if disable_join_order:
            con.execute("SET disabled_optimizers = 'join_order,build_side_probe_side'")
        con.execute("PRAGMA enable_profiling = 'json'")
        profile_file = _current_profile_file()
        con.execute(f"PRAGMA profiling_output = '{profile_file}'")

        # Optional warmup run: populate buffer pool / OS page cache, discard timing
        if _per_query_warmup_enabled():
            _execute_with_timeout(con, query)

        result = []
        cpu_times = []
        profiling_runs = []
        for _ in range(attempts):
            result = _execute_with_timeout(con, query)

            # Read back the DuckDB profiling JSON for this run's latency. Keep
            # this isolated from the result: under parallel validation the
            # profiling file write can rarely lag/race, and a read failure must
            # only drop this attempt's *timing*, never discard the query result
            # (which would corrupt the correctness check).
            try:
                with open(profile_file, "r") as file:
                    performance_data = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                performance_data = {}
            cpu_times.append(performance_data.get("latency", -1))
            profiling_runs.append(performance_data)

        # Use median to discard outliers from OS scheduling jitter
        cpu_times.sort()
        median_idx = len(cpu_times) // 2 if cpu_times else 0
        cpu_time = cpu_times[median_idx] if cpu_times else -1

        # Return the profiling data from the median-time run
        median_latency = cpu_time
        profiling_data = profiling_runs[0] if profiling_runs else {}
        for p in profiling_runs:
            if p.get("latency", -1) == median_latency:
                profiling_data = p
                break

        return result, cpu_time, profiling_data, cpu_times
    except Exception as e:
        raise QueryExecutionError(f"DuckDB could not execute query: {e}\nFor query: {query}") from e


def execute_query_no_profiling(query, db_file="imdb.duckdb"):
    """Execute a query and return (success, result_rows)."""
    con = duckdb.connect(db_file)
    try:
        _apply_duckdb_guards(con)
        result = _execute_with_timeout(con, query)
        return True, result
    except Exception as e:
        print(f"Exception when executing validation query: {e}")
        print(f"For query: {query}")
        return False, []


# ---------------------------------------------------------------------------
# Base-table validation — the soundness gate.
#
# Builds A = rows satisfying `requires` and B = rows also satisfying `implies` from the
# rule's conditions and derived joins; a row of A missing from B is a counterexample.
# Error, timeout, or a vacuous rule (A empty) all count as failure, never as a pass.
# ---------------------------------------------------------------------------

def _format_value(value):
    """Format a Python value as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_format_value(v) for v in value) + ")"
    return str(value)


def _quote_ident(name: str) -> str:
    """Double-quote a single SQL identifier (doubling any embedded quote).

    Needed because base-table validation builds SQL by hand.  DuckDB folds
    unquoted identifiers case-insensitively, but Umbra/Postgres fold them to
    lowercase — so an unquoted mixed-case name like
    ``On_Time_On_Time_Performance_2016_1`` silently fails to resolve on Umbra
    and the whole validation errors out (rule wrongly discarded).  Quoting is a
    no-op for already-lowercase schemas (e.g. JOB) and case-insensitive on
    DuckDB, so it is safe across all engines.
    """
    return '"' + name.replace('"', '""') + '"'


def _quote_column(dotted: str) -> str:
    """Quote a possibly dotted ``table.column`` reference part-by-part.

    A bare ``*`` part (e.g. ``alias.*``) is passed through unquoted.
    """
    return ".".join(p if p == "*" else _quote_ident(p) for p in dotted.split("."))


def _normalize_op(op, value):
    """Normalize LLM operator variants to canonical SQL form."""
    op_upper = op.strip().upper()
    if op_upper == "IS NOT NULL":
        return "IS NOT", None
    if op_upper == "IS NULL":
        return "IS", None
    return op, value


def _condition_to_sql(cond):
    """Recursively convert a rule condition dict to a SQL WHERE fragment.

    Handles leaf conditions (column op value) and logical nodes (AND/OR).
    """
    # Logical node
    if "conditions" in cond and "op" in cond and "column" not in cond:
        logical_op = cond["op"].upper()
        parts = [_condition_to_sql(sub) for sub in cond["conditions"]]
        joined = f" {logical_op} ".join(parts)
        return f"({joined})"

    # Leaf condition
    column = _quote_column(cond["column"])  # "table.column" -> "table"."column"
    op = cond["op"]
    value = cond.get("value")
    op, value = _normalize_op(op, value)
    op_upper = op.strip().upper()

    if op_upper == "IN":
        return f"{column} IN {_format_value(value)}"
    elif op_upper == "NOT IN":
        return f"{column} NOT IN {_format_value(value)}"
    elif op_upper == "IS":
        return f"{column} IS {_format_value(value)}"
    elif op_upper == "IS NOT":
        return f"{column} IS NOT {_format_value(value)}"
    elif op_upper == "LIKE":
        return f"{column} LIKE {_format_value(value)}"
    elif op_upper == "NOT LIKE":
        return f"{column} NOT LIKE {_format_value(value)}"
    elif op_upper == "BETWEEN":
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{column} BETWEEN {_format_value(value[0])} AND {_format_value(value[1])}"
        return f"{column} BETWEEN {_format_value(value)}"
    else:
        return f"{column} {op} {_format_value(value)}"


def _implies_to_sql(implies):
    """Convert the implies list to a SQL WHERE fragment (ANDed together)."""
    parts = []
    for impl in implies:
        column = _quote_column(impl["column"])
        op = impl["op"]
        value = impl.get("value")
        op, value = _normalize_op(op, value)
        op_upper = op.strip().upper()

        if op_upper == "IN":
            parts.append(f"{column} IN {_format_value(value)}")
        elif op_upper == "NOT IN":
            parts.append(f"{column} NOT IN {_format_value(value)}")
        elif op_upper == "IS":
            parts.append(f"{column} IS {_format_value(value)}")
        elif op_upper == "IS NOT":
            parts.append(f"{column} IS NOT {_format_value(value)}")
        elif op_upper == "LIKE":
            parts.append(f"{column} LIKE {_format_value(value)}")
        elif op_upper == "NOT LIKE":
            parts.append(f"{column} NOT LIKE {_format_value(value)}")
        elif op_upper == "BETWEEN":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                parts.append(f"{column} BETWEEN {_format_value(value[0])} AND {_format_value(value[1])}")
            else:
                parts.append(f"{column} BETWEEN {_format_value(value)}")
        else:
            parts.append(f"{column} {op} {_format_value(value)}")
    return " AND ".join(parts)


def _translate_condition_to_aliases(cond, base_to_alias):
    """Recursively rewrite a condition dict replacing base table names with aliases."""
    if "conditions" in cond and "column" not in cond:
        # Logical node (AND/OR)
        return {
            "op": cond["op"],
            "conditions": [_translate_condition_to_aliases(c, base_to_alias) for c in cond["conditions"]],
        }
    # Leaf condition
    new_cond = dict(cond)
    if "column" in new_cond:
        parts = new_cond["column"].split(".", 1)
        if len(parts) == 2:
            table, col = parts
            new_cond["column"] = f"{base_to_alias.get(table, table)}.{col}"
    return new_cond


def _translate_implies_to_aliases(implies, base_to_alias):
    """Rewrite implies list replacing base table names with aliases."""
    result = []
    for impl in implies:
        new_impl = dict(impl)
        if "column" in new_impl:
            parts = new_impl["column"].split(".", 1)
            if len(parts) == 2:
                table, col = parts
                new_impl["column"] = f"{base_to_alias.get(table, table)}.{col}"
        result.append(new_impl)
    return result


def build_validation_queries(rule):
    """Build two SQL queries from a rule for base-table validation.

    When the rule contains an ``alias_map`` (alias → base table) the
    joins are expected to use alias names.  The generated queries use
    ``base_table AS alias`` in FROM clauses so that multiple aliases of
    the same base table remain distinguishable.

    Returns (requires_only_sql, requires_and_implies_sql) or None if
    table information cannot be determined.
    """
    joins = rule.get("joins", [])
    requires = rule.get("requires", {})
    implies = rule.get("implies", [])
    alias_map = rule.get("alias_map")  # alias → base table (may be None)

    # ---- Collect table references used in joins --------------------------
    tables = {}  # preserves insertion order (dict)
    for j in joins:
        for side in ("left", "right"):
            tbl = j[side].split(".")[0]
            tables.setdefault(tbl, True)

    # Also collect tables from conditions (handles single-table rules
    # with no joins).  When an alias_map is present the condition
    # columns still use base table names, so we must convert.
    if alias_map:
        # Build reverse map: base → alias (prefer alias from joins when
        # available so we stay consistent with the join graph).
        base_to_alias: dict[str, str] = {}
        for alias, base in alias_map.items():
            if alias != base and alias in tables:
                base_to_alias[base] = alias
        # Fallback for tables not yet in the join set
        for alias, base in alias_map.items():
            if alias != base:
                base_to_alias.setdefault(base, alias)
    else:
        base_to_alias = {}

    def _collect_tables(cond):
        if "column" in cond:
            tbl = cond["column"].split(".")[0]
            resolved = base_to_alias.get(tbl, tbl)
            tables.setdefault(resolved, True)
        for sub in cond.get("conditions", []):
            _collect_tables(sub)

    if isinstance(requires, list):
        for r in requires:
            _collect_tables(r)
    elif isinstance(requires, dict):
        _collect_tables(requires)
    for impl in implies:
        if "column" in impl:
            tbl = impl["column"].split(".")[0]
            resolved = base_to_alias.get(tbl, tbl)
            tables.setdefault(resolved, True)

    table_list = list(tables.keys())
    if not table_list:
        return None

    # ---- Build FROM + JOIN ON clauses ------------------------------------
    first_table = table_list[0]
    join_clauses = []
    joined = {first_table}
    remaining = set(table_list[1:])

    max_iterations = len(remaining) * 2
    iteration = 0
    while remaining and iteration < max_iterations:
        iteration += 1
        found = False
        for j in joins:
            left_tbl = j["left"].split(".")[0]
            right_tbl = j["right"].split(".")[0]
            if left_tbl in joined and right_tbl in remaining:
                tbl_label = _from_label(right_tbl, alias_map)
                join_clauses.append(f"JOIN {tbl_label} ON {_quote_column(j['left'])} = {_quote_column(j['right'])}")
                joined.add(right_tbl)
                remaining.discard(right_tbl)
                found = True
            elif right_tbl in joined and left_tbl in remaining:
                tbl_label = _from_label(left_tbl, alias_map)
                join_clauses.append(f"JOIN {tbl_label} ON {_quote_column(j['left'])} = {_quote_column(j['right'])}")
                joined.add(left_tbl)
                remaining.discard(left_tbl)
                found = True
        if not found:
            remaining_list = list(remaining)
            print(f"WARNING: Cannot connect tables {remaining_list} to {joined} via joins — skipping base-table validation for rule {rule.get('id', 'unknown')}")
            return None

    first_label = _from_label(first_table, alias_map)
    from_clause = f"FROM {first_label} " + " ".join(join_clauses)

    # ---- Build WHERE clauses (translated to alias names) -----------------
    if alias_map:
        if isinstance(requires, list):
            translated_requires = [_translate_condition_to_aliases(c, base_to_alias) for c in requires]
            requires_sql = " AND ".join(_condition_to_sql(c) for c in translated_requires)
        elif isinstance(requires, dict):
            translated_requires = _translate_condition_to_aliases(requires, base_to_alias)
            requires_sql = _condition_to_sql(translated_requires)
        else:
            return None
        translated_implies = _translate_implies_to_aliases(implies, base_to_alias)
        implies_sql = _implies_to_sql(translated_implies) if translated_implies else ""
    else:
        # No alias_map: use base table names
        if isinstance(requires, list):
            requires_sql = " AND ".join(_condition_to_sql(c) for c in requires)
        elif isinstance(requires, dict):
            requires_sql = _condition_to_sql(requires)
        else:
            return None
        implies_sql = _implies_to_sql(implies) if implies else ""

    # ---- Handle join_elimination rules differently -------------------------
    is_join_elimination = rule.get("type") == "join_elimination" and rule.get("eliminates")
    if is_join_elimination:
        eliminated_base_tables = set(rule["eliminates"])
        # Determine which aliases correspond to eliminated tables
        eliminated_aliases = set()
        surviving_aliases = []
        for tbl in table_list:
            base = alias_map.get(tbl, tbl) if alias_map else tbl
            if base in eliminated_base_tables:
                eliminated_aliases.add(tbl)
            else:
                surviving_aliases.append(tbl)

        if not surviving_aliases:
            print(f"WARNING: All tables would be eliminated for rule {rule.get('id', 'unknown')}")
            return None

        # SELECT only surviving table columns
        select_cols = ", ".join(f"{_quote_ident(a)}.*" for a in surviving_aliases)

        # Query A: all tables (including eliminated), project onto surviving only
        query_a = f"SELECT {select_cols} {from_clause} WHERE {requires_sql}"

        # Query B: only surviving tables, with implies predicates
        surv_first = surviving_aliases[0]
        surv_join_clauses = []
        surv_joined = {surv_first}
        surv_remaining = set(surviving_aliases[1:])
        max_iter = len(surv_remaining) * 2
        iteration = 0
        while surv_remaining and iteration < max_iter:
            iteration += 1
            found = False
            for j in joins:
                lt = j["left"].split(".")[0]
                rt = j["right"].split(".")[0]
                if lt in eliminated_aliases or rt in eliminated_aliases:
                    continue  # Skip joins involving eliminated tables
                if lt in surv_joined and rt in surv_remaining:
                    tbl_label = _from_label(rt, alias_map)
                    surv_join_clauses.append(f"JOIN {tbl_label} ON {_quote_column(j['left'])} = {_quote_column(j['right'])}")
                    surv_joined.add(rt)
                    surv_remaining.discard(rt)
                    found = True
                elif rt in surv_joined and lt in surv_remaining:
                    tbl_label = _from_label(lt, alias_map)
                    surv_join_clauses.append(f"JOIN {tbl_label} ON {_quote_column(j['left'])} = {_quote_column(j['right'])}")
                    surv_joined.add(lt)
                    surv_remaining.discard(lt)
                    found = True
            if not found:
                break

        surv_first_label = _from_label(surv_first, alias_map)
        surv_from_clause = f"FROM {surv_first_label} " + " ".join(surv_join_clauses)

        if implies_sql:
            query_b = f"SELECT {select_cols} {surv_from_clause} WHERE {implies_sql}"
        else:
            query_b = f"SELECT {select_cols} {surv_from_clause}"

        return query_a, query_b

    # ---- Standard filter rule validation -----------------------------------
    query_a = f"SELECT * {from_clause} WHERE {requires_sql}"
    if implies_sql:
        query_b = f"SELECT * {from_clause} WHERE {requires_sql} AND {implies_sql}"
    else:
        query_b = query_a

    return query_a, query_b


def _from_label(table_or_alias: str, alias_map: dict[str, str] | None) -> str:
    """Return ``base_table AS alias`` if *table_or_alias* is an alias, else just the name."""
    if alias_map:
        base = alias_map.get(table_or_alias)
        if base and base != table_or_alias:
            return f"{_quote_ident(base)} AS {_quote_ident(table_or_alias)}"
    return _quote_ident(table_or_alias)


def _extract_rules(sql_data):
    """Extract rule dicts from an entry, handling both transfer and aggregated formats."""
    rules = []
    # Aggregated format: {"rules": [{"name": ..., "rule": {...}}, ...]}
    if "rules" in sql_data and isinstance(sql_data["rules"], list):
        for r in sql_data["rules"]:
            rule = r.get("rule") if isinstance(r, dict) else None
            if isinstance(rule, dict):
                rules.append(rule)
    # Single-rule format: {"response": {...}}
    elif "response" in sql_data and isinstance(sql_data["response"], dict):
        resp = sql_data["response"]
        if "requires" in resp and "implies" in resp:
            rules.append(resp)
    return rules


def _run_validation_probe(query, db_file):
    """Execute a *bounded* base-table validation query; return (rows, seconds).

    "Bounded" means the query returns at most a handful of rows — a COUNT or a
    LIMIT-capped sample.  This is the deliberate replacement for pulling the full
    validation row sets into the driver via ``fetchall()``.  The heavy join still
    runs *server-side*, where the Umbra container's RAM cgroup cap applies (a
    runaway join OOMs inside the container, contained); only the tiny result
    crosses the wire, so it can no longer balloon the uncapped driver process and
    exhaust host memory.  Returns ``(None, 0.0)`` on error/timeout so the caller
    treats the rule as failing validation (discarded / refined).
    """
    # Umbra / Postgres: run once on the pooled connection under the per-query
    # cancel watchdog.  DuckDB: fresh connection with the standard guards.
    if _umbra is not None:
        try:
            cur = _umbra._con.cursor()
            t0 = time.perf_counter()
            with _umbra_timeout_guard():
                cur.execute(query)
                rows = cur.fetchall()
            elapsed = time.perf_counter() - t0
            cur.close()
            return rows, elapsed
        except Exception as e:
            print(f"Validation probe failed (Umbra): {e}\n  For query: {query[:200]}")
            try:
                _umbra._con.rollback()
            except Exception:
                pass
            return None, 0.0

    if _postgres is not None:
        try:
            cur = _postgres._con.cursor()
            if _QUERY_TIMEOUT_S and _QUERY_TIMEOUT_S > 0:
                cur.execute(f"SET statement_timeout = {int(_QUERY_TIMEOUT_S * 1000)}")
            t0 = time.perf_counter()
            cur.execute(query)
            rows = cur.fetchall()
            elapsed = time.perf_counter() - t0
            cur.close()
            return rows, elapsed
        except Exception as e:
            print(f"Validation probe failed (Postgres): {e}\n  For query: {query[:200]}")
            try:
                _postgres._con.rollback()
            except Exception:
                pass
            return None, 0.0

    con = duckdb.connect(db_file)
    try:
        _apply_duckdb_guards(con)
        con.execute(f"SET threads TO {_DUCKDB_THREADS}")
        t0 = time.perf_counter()
        rows = _execute_with_timeout(con, query)
        return rows, time.perf_counter() - t0
    except Exception as e:
        print(f"Validation probe failed (DuckDB): {e}\n  For query: {query[:200]}")
        return None, 0.0
    finally:
        con.close()


def run_base_table_validation(sql_data, db_file, attempts=3):
    """Run base-table validation for all rules in an entry.

    Returns a dict with validation results, or None if no rules have joins.

    Rows are compared as *multisets*: order is irrelevant, multiplicity is not.
    A rewrite that changes how often a row appears is not correct, even when the
    distinct rows agree — the workloads aggregate over the join output, so
    duplicates carry meaning.

    The comparison is computed *server-side* (two COUNTs plus an ``EXCEPT ALL``
    violation sample) so only tiny results are pulled into the driver.  The full
    row multisets are never fetched: a runaway many-to-many join can produce
    billions of rows, and materializing them here (the old ``fetchall()``)
    exhausted host memory *outside* the container's cgroup cap.
    """
    rules = _extract_rules(sql_data)
    if not rules:
        return None

    all_results = []
    for rule in rules:
        pair = build_validation_queries(rule)
        if pair is None:
            all_results.append({
                "rule_id": rule.get("id", "unknown"),
                "status": "skipped_no_joins",
            })
            continue

        query_a, query_b = pair
        rule_id = rule.get("id", "unknown")

        # Raw (non-DISTINCT) counts: multiset cardinality. The heavy join runs
        # server-side; only the scalar count comes back.
        cnt_a_rows, time_a = _run_validation_probe(
            f"SELECT count(*) FROM ({query_a}) _a", db_file)
        cnt_b_rows, time_b = _run_validation_probe(
            f"SELECT count(*) FROM ({query_b}) _b", db_file)

        if cnt_a_rows is None or cnt_b_rows is None:
            # A probe failed or timed out (e.g. a runaway join cancelled or
            # OOM-killed inside the container): treat the rule as failing
            # validation so it is discarded / fed back to refinement.
            print(f"WARNING: base-table validation probe failed for rule "
                  f"{rule_id} — marking invalid")
            all_results.append({
                "rule_id": rule_id,
                "status": "error",
                "outputs_match": False,
                "requires_row_count": None,
                "requires_and_implies_row_count": None,
                "violation_count": None,
                "violations_sample": [],
                "execution_time": {
                    "requires_only": round(time_a, 4),
                    "requires_and_implies": round(time_b, 4),
                    "time_saved": 0,
                    "percent_saved": 0,
                },
                "query_requires": query_a,
                "query_requires_and_implies": query_b,
            })
            continue

        cnt_a = cnt_a_rows[0][0] if cnt_a_rows else 0
        cnt_b = cnt_b_rows[0][0] if cnt_b_rows else 0

        # Guard against vacuous truth: if the requires-only query returns
        # zero rows the validation is meaningless (∅ == ∅ always holds).
        if cnt_a == 0:
            print(f"WARNING: requires-only query returned 0 rows for rule "
                  f"{rule_id} — skipping (vacuous validation)")
            all_results.append({
                "rule_id": rule_id,
                "status": "skipped_empty_requires",
                "outputs_match": False,
                "requires_row_count": 0,
                "requires_and_implies_row_count": cnt_b,
                "violation_count": 0,
                "violations_sample": [],
                "execution_time": {
                    "requires_only": round(time_a, 4),
                    "requires_and_implies": round(time_b, 4),
                    "time_saved": 0,
                    "percent_saved": 0,
                },
                "query_requires": query_a,
                "query_requires_and_implies": query_b,
            })
            continue

        # Violations = (requires-only) EXCEPT ALL (requires-and-implies): rows
        # that satisfy `requires` but not `implies`, counted with multiplicity.
        # EXCEPT ALL, not EXCEPT: plain EXCEPT deduplicates both sides, so a row
        # occurring three times on the left and once on the right yields no
        # violation at all.  Pull at most 11 rows — 10 for the sample, the 11th
        # only to know the count exceeds 10 (so we avoid a second heavy join in
        # the common valid case, where the difference is empty).
        sample_rows, _ = _run_validation_probe(
            f"SELECT * FROM (({query_a}) EXCEPT ALL ({query_b})) _v LIMIT 11", db_file)
        if sample_rows is None:
            sample_rows = []
            violation_count = None
        elif len(sample_rows) <= 10:
            violation_count = len(sample_rows)
        else:
            cnt_rows, _ = _run_validation_probe(
                f"SELECT count(*) FROM (({query_a}) EXCEPT ALL ({query_b})) _v", db_file)
            violation_count = cnt_rows[0][0] if cnt_rows else None

        # match == (multiset_a == multiset_b).  A zero violation count means
        # multiset_a ⊑ multiset_b; combined with equal cardinalities this is
        # exactly multiset equality — and covers both standard rules (where
        # multiset_b ⊑ multiset_a structurally) and join-elimination rules
        # (where neither side is a sub-multiset of the other).
        match = (violation_count == 0) and (cnt_a == cnt_b)

        time_saved = time_a - time_b
        pct_saved = (time_saved / time_a * 100.0) if time_a > 0 else 0.0

        all_results.append({
            "rule_id": rule_id,
            "status": "valid" if match else "invalid",
            "outputs_match": match,
            "requires_row_count": cnt_a,
            "requires_and_implies_row_count": cnt_b,
            "violation_count": violation_count,
            "violations_sample": [str(r) for r in sample_rows[:10]],
            "execution_time": {
                "requires_only": round(time_a, 4),
                "requires_and_implies": round(time_b, 4),
                "time_saved": round(time_saved, 4),
                "percent_saved": round(pct_saved, 2),
            },
            "query_requires": query_a,
            "query_requires_and_implies": query_b,
        })

    if not all_results:
        return None
    return {
        "rules_validated": len(all_results),
        "all_valid": all(
            r.get("outputs_match", False)
            or r.get("status") in ("skipped_no_joins",)
            for r in all_results
        ),
        "details": all_results,
    }


# ---------------------------------------------------------------------------
# Cost estimation via EXPLAIN (no query execution)
# ---------------------------------------------------------------------------

# Per-operator weights for the DuckDB selection signal. All 1.0 (left untuned), so the
# signal is a plain sum of estimated cardinalities.
OPERATOR_WEIGHTS = {
    "SEQ_SCAN": 1.0,
    "INDEX_SCAN": 1.0,
    "FILTER": 1.0,
    "PROJECTION": 1.0,
    "HASH_JOIN": 1.0,
    "NESTED_LOOP_JOIN": 1.0,
    "PIECEWISE_MERGE_JOIN": 1.0,
    "BLOCKWISE_NL_JOIN": 1.0,
    "CROSS_PRODUCT": 1.0,
    "ORDER_BY": 1.0,
    "HASH_GROUP_BY": 1.0,
    "PERFECT_HASH_GROUP_BY": 1.0,
    "UNGROUPED_AGGREGATE": 1.0,
    "STREAMING_LIMIT": 1.0,
    "LIMIT": 1.0,
    "DISTINCT": 1.0,
    "WINDOW": 1.0,
    "TOP_N": 1.0,
}
DEFAULT_OPERATOR_WEIGHT = 1.0


def _sum_plan_cost(node):
    """Sum the estimated cardinalities of an EXPLAIN JSON plan (weights are all 1.0)."""
    if not isinstance(node, dict):
        return 0.0

    extra = node.get("extra_info", {})
    cardinality = 0
    if isinstance(extra, dict):
        ec_str = extra.get("Estimated Cardinality", "0")
        try:
            cardinality = int(ec_str)
        except (ValueError, TypeError):
            cardinality = 0

    op_name = node.get("name", "").upper()
    weight = OPERATOR_WEIGHTS.get(op_name, DEFAULT_OPERATOR_WEIGHT)
    cost = weight * cardinality

    for child in node.get("children", []):
        cost += _sum_plan_cost(child)

    return cost


def estimate_query_cost(query, db_file="imdb.duckdb"):
    """Run EXPLAIN and return a cost estimate.

    For DuckDB: EXPLAIN (FORMAT JSON) → weighted-cardinality sum.
    For Umbra: EXPLAIN (FORMAT JSON) → cardinality sum (no execution).

    Returns (cost, plan).  On failure returns (float('inf'), {}).
    """
    # Dispatch to Postgres / Umbra when active
    if _postgres is not None:
        return _estimate_cost_postgres(query)
    if _umbra is not None:
        return _estimate_cost_umbra(query)

    con = duckdb.connect(db_file)
    try:
        rows = con.execute(
            f"EXPLAIN (FORMAT JSON) {_quote_reserved_aliases(query)}").fetchall()
        plan_text = rows[0][1] if rows else "{}"
        plan = json.loads(plan_text)

        if isinstance(plan, list):
            total_cost = sum(_sum_plan_cost(n) for n in plan)
        else:
            total_cost = _sum_plan_cost(plan)

        return total_cost, plan
    except Exception as e:
        print(f"Cost estimation failed: {e}")
        print(f"  For query: {query[:200]}")
        return float("inf"), {}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Cost-aggregate mode: apply rules to queries with EXPLAIN-based filtering
# ---------------------------------------------------------------------------

def _combo_search(
    original_sql: str,
    fired_rules: list,
    original_cost: float,
    db_file: str,
    cost_save_threshold: float = 0.0,
    *,
    original_plan: dict | None = None,
    engine: str = "duckdb",
    fix_join_order: bool = False,
) -> tuple:
    """Branch-and-bound over the rule-subset lattice — the deployable selector.

    Level 1 (individual rules): always evaluated, never pruned.
    Level 2+ (combinations): pruned when rewritten_cost >= original_cost.
    Deduplication: a set is only extended with rules at higher indices, so each
    unique subset is evaluated exactly once.

    The prune is a heuristic, not a sound bound (cost is not monotone in subset size).
    It affects the selector view only; the oracle is the minimum over *measured* subsets.

    A subset only wins when its cost is strictly below
    ``original_cost * (1 - cost_save_threshold)`` — i.e. it must save at least
    that fraction relative to the original. ``cost_save_threshold = 0.0``
    reduces to the original ``cost < original_cost`` rule.

    Returns:
        best_indices: sorted list of indices into fired_rules for winning subset
                      (empty list if no subset beats the threshold)
        best_cost:    cost of the winning subset
        explored:     list of dicts {indices, rules, cost, pruned} for all evaluated subsets
    """
    n = len(fired_rules)
    if n == 0:
        return [], original_cost, [], original_sql

    win_cutoff = original_cost * (1.0 - cost_save_threshold)

    explored = []
    best_cost = original_cost
    best_indices = []
    best_exec_sql = original_sql  # falls through if no subset wins

    def _exec_sql_for(rewritten_sql: str) -> str:
        sql_str, _pinned = build_execution_sql(
            original_sql, rewritten_sql, original_plan,
            engine=engine, fix_join_order=fix_join_order,
        )
        return sql_str

    # Level 1: individual rules — evaluate all, never prune
    active_sets = []  # list of (tuple-of-indices,) that survived pruning
    for i in range(n):
        rule = fired_rules[i]["rule"]
        rewritten_sql, _ = apply_sql_rules(original_sql, [rule])
        exec_sql = _exec_sql_for(rewritten_sql)
        cost, _ = estimate_query_cost(exec_sql, db_file=db_file)

        explored.append({
            "indices": [i],
            "rules": [fired_rules[i]["name"]],
            "cost": round(cost, 2),
            "pruned": False,
        })

        if cost < win_cutoff and cost < best_cost:
            best_cost = cost
            best_indices = [i]
            best_exec_sql = exec_sql

        active_sets.append((i,))  # individual rules always survive to next level

    # Level 2+: extend each surviving set with one higher-index rule
    while active_sets:
        next_active = []
        for current_indices in active_sets:
            max_idx = current_indices[-1]
            for j in range(max_idx + 1, n):
                new_indices = current_indices + (j,)
                new_rules = [fired_rules[k]["rule"] for k in new_indices]
                new_sql, _ = apply_sql_rules(original_sql, new_rules)
                exec_sql = _exec_sql_for(new_sql)
                cost, _ = estimate_query_cost(exec_sql, db_file=db_file)

                pruned = cost >= original_cost
                explored.append({
                    "indices": list(new_indices),
                    "rules": [fired_rules[k]["name"] for k in new_indices],
                    "cost": round(cost, 2),
                    "pruned": pruned,
                })

                if not pruned:
                    next_active.append(new_indices)
                    if cost < win_cutoff and cost < best_cost:
                        best_cost = cost
                        best_indices = list(new_indices)
                        best_exec_sql = exec_sql

        active_sets = next_active

    return best_indices, best_cost, explored, best_exec_sql


def run_cost_aggregate(args):
    """Apply rules to queries, keeping the rule combination with lowest estimated cost.

    Uses branch-pruning combinatorial search: individual rules are always evaluated;
    combinations of 2+ rules are pruned when their cost exceeds the original query cost.
    The subset with globally lowest cost across all explored combinations wins.

    Input JSON format (from local aggregation):
        { "rules": [...], "queries": {"name": "SELECT ..."} }

    Output JSON format (same as rule_summary_transfer.json):
        { "query_name": {"sql": ..., "rules": [...], "refined_sql": ..., "cost_estimation": {...}} }
    """
    if apply_sql_rules is None:
        print("ERROR: apply_sql_rules not available — cannot run cost aggregation")
        print("Make sure sql_predicate_converter.py is in the same directory")
        return

    with open(args.input, "r") as f:
        data = json.load(f)

    rule_entries = data["rules"]
    queries = data["queries"]

    result = {}
    for sql_name, sql in tqdm(queries.items(), desc="Cost-filtered aggregation"):
        # Find which rules fire for this query
        fired_rules = []
        for entry in rule_entries:
            rule = entry.get("rule")
            if not rule:
                continue
            _, single_fired = apply_sql_rules(sql, [rule])
            if single_fired:
                fired_rules.append({"name": entry.get("name"), "rule": rule})

        if not fired_rules:
            continue

        # Keep the original plan: needed to build the same hint /
        # reconstruction that final execution will apply, so cost estimation
        # operates on the exact SQL string that will be benchmarked.
        original_cost, original_plan = estimate_query_cost(sql, db_file=args.db_file)
        threshold_pct = args.cost_save_threshold * 100.0
        print(f"\n{sql_name}: {len(fired_rules)} candidate rules, original cost={original_cost:.1f} "
              f"(min required saving: {threshold_pct:.2f}%)")

        best_indices, best_cost, explored, best_exec_sql = _combo_search(
            sql, fired_rules, original_cost, args.db_file,
            cost_save_threshold=args.cost_save_threshold,
            original_plan=original_plan,
            engine=args.engine,
            fix_join_order=args.fix_join_order,
        )

        per_rule_details = [
            {
                "name": e["rules"][0],
                "cost": e["cost"],
                "cost_reduction": round(original_cost - e["cost"], 2),
            }
            for e in explored if len(e["indices"]) == 1
        ]

        cost_estimation = {
            "original_cost": round(original_cost, 2),
            "cost_save_threshold": args.cost_save_threshold,
            "fix_join_order": bool(args.fix_join_order),
            "per_rule": per_rule_details,
            "combo_search": {
                "explored_count": len(explored),
                "explored": explored,
                "winner_rules": [fired_rules[i]["name"] for i in best_indices],
                "winner_cost": round(best_cost, 2),
            },
        }

        kept_rules = [fired_rules[i] for i in best_indices]
        rejected_rules = [fired_rules[i] for i in range(len(fired_rules)) if i not in set(best_indices)]
        winner_names = [fired_rules[i]["name"] for i in best_indices]

        if args.keep_full_pool:
            # Oracle-safe: keep every fired rule so final-exec enumerates the full
            # powerset; the cost winner travels only as metadata for the optimizer
            # view. refined_sql is the all-rules SQL so the query-validation /
            # per-subset path still runs even when the optimizer declines all rules.
            all_rule_dicts = [e["rule"] for e in fired_rules]
            refined_sql, _ = apply_sql_rules(sql, all_rule_dicts)
            if best_indices:
                cost_estimation["final_cost"] = round(best_cost, 2)
                cost_estimation["execution_sql"] = best_exec_sql
            # Marker so statistics drives the optimizer view from this winner only
            # for full-pool (oracle) runs; pruned optimizer-only runs keep the
            # legacy variance-capable view.
            cost_estimation["full_pool_kept"] = True
            print(f"  Winner: {winner_names or '(none — optimizer declines)'}, "
                  f"cost={best_cost:.1f} (explored {len(explored)} subsets); "
                  f"keeping full pool of {len(fired_rules)} rules for the oracle")
            result[sql_name] = {
                "sql": sql,
                "rules": fired_rules,
                "refined_sql": refined_sql,
                "cost_estimation": cost_estimation,
            }
            continue

        # Default (pruning) behavior — pure optimizer cost runs stay cheap.
        if not kept_rules:
            print(f"  No beneficial combination found for {sql_name}")
            result[sql_name] = {
                "sql": sql,
                "rules": [],
                "rules_rejected": rejected_rules,
                "cost_estimation": cost_estimation,
            }
            continue

        kept_rule_dicts = [e["rule"] for e in kept_rules]
        refined_sql, _ = apply_sql_rules(sql, kept_rule_dicts)

        cost_estimation["final_cost"] = round(best_cost, 2)
        # The exact SQL string that was scored — final execution must use this
        # same string (modulo deterministic re-derivation from the same plan).
        cost_estimation["execution_sql"] = best_exec_sql
        print(f"  Winner: {winner_names}, cost={best_cost:.1f} (explored {len(explored)} subsets)")

        result[sql_name] = {
            "sql": sql,
            "rules": kept_rules,
            "rules_rejected": rejected_rules,
            "refined_sql": refined_sql,
            "cost_estimation": cost_estimation,
        }

    has_winner = sum(
        1 for v in result.values()
        if v.get("cost_estimation", {}).get("combo_search", {}).get("winner_rules")
    )
    print(f"\nTotal: {has_winner} of {len(queries)} queries have a cost-beneficial rule subset")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4, default=_json_default)


def run_optimizer_winners(args):
    """Pick the cost-winning rule subset per query without executing queries.

    Reads a rule_summary_transfer.json (each query: {"sql", "rules": [{name, rule}...]})
    and, per query, runs the EXPLAIN-only cost search (``_combo_search``) to find the
    subset a cost-based optimizer would choose. Writes a winners sidecar:

        { "query_name": {"winner_rules": [...], "winner_cost": x, "original_cost": y} }

    Used to retrofit an existing oracle run (executed with cost estimation off) with a
    corrected optimizer view: statistics maps each winner to its already-measured
    runtime in per_subset_results. No query is executed here — EXPLAIN only.
    """
    if apply_sql_rules is None:
        print("ERROR: apply_sql_rules not available — cannot run optimizer-winners")
        print("Make sure sql_predicate_converter.py is in the same directory")
        return

    with open(args.input, "r") as f:
        data = json.load(f)

    winners = {}
    for sql_name, entry in tqdm(data.items(), desc="Optimizer winner search"):
        if not isinstance(entry, dict):
            continue
        sql = entry.get("sql")
        fired_rules = entry.get("rules") or []
        if not sql or not fired_rules:
            continue

        original_cost, original_plan = estimate_query_cost(sql, db_file=args.db_file)
        best_indices, best_cost, explored, _ = _combo_search(
            sql, fired_rules, original_cost, args.db_file,
            cost_save_threshold=args.cost_save_threshold,
            original_plan=original_plan,
            engine=args.engine,
            fix_join_order=args.fix_join_order,
        )
        winner_rules = [fired_rules[i]["name"] for i in best_indices]
        winners[sql_name] = {
            "winner_rules": winner_rules,
            "winner_cost": round(best_cost, 2),
            "original_cost": round(original_cost, 2),
        }
        print(f"{sql_name}: winner={winner_rules or '(none — optimizer declines)'} "
              f"cost {original_cost:.1f} -> {best_cost:.1f} "
              f"(explored {len(explored)}, pool {len(fired_rules)})")

    n_win = sum(1 for v in winners.values() if v["winner_rules"])
    print(f"\nTotal: {n_win} of {len(winners)} queries have a cost-beneficial subset")

    with open(args.output, "w") as f:
        json.dump(winners, f, indent=4, default=_json_default)


def run_warmup(args):
    """Prime the engine's buffer pool / page cache by running queries once,
    with no measurement.

    Source priority: if ``--sql-dir`` is given, read every ``*.sql`` file under
    that directory (sorted by filename). Otherwise fall back to ``--input`` (a
    JSON dict whose values contain an ``sql`` field — same shape as
    transfer.json), sorted by key.

    ``--warmup-limit`` caps the number of queries (0 or negative = no cap).

    Used with ``warmup_mode: startup_only`` or ``startup_and_per_query`` after
    the DB container starts.
    """
    items: list[tuple[str, str]] = []
    if getattr(args, "sql_dir", None):
        sql_dir = args.sql_dir
        for fname in sorted(os.listdir(sql_dir)):
            if not fname.endswith(".sql"):
                continue
            with open(os.path.join(sql_dir, fname), "r") as f:
                items.append((fname, f.read()))
    else:
        with open(args.input, "r") as f:
            data = json.load(f)
        for k in sorted(data.keys()):
            sql = data[k].get("sql") if isinstance(data[k], dict) else None
            if sql:
                items.append((k, sql))

    limit = args.warmup_limit
    if limit and limit > 0:
        items = items[:limit]

    print(f"Warmup: running {len(items)} queries to prime engine caches "
          f"(engine={args.engine}).")
    for k, sql in items:
        try:
            if _postgres is not None:
                cur = _postgres._con.cursor()
                cur.execute(sql)
                cur.fetchall()
                cur.close()
            elif _umbra is not None:
                cur = _umbra._con.cursor()
                cur.execute(sql)
                cur.fetchall()
                cur.close()
            else:
                con = duckdb.connect(args.db_file)
                try:
                    con.execute("SET threads TO 1")
                    con.execute(_quote_reserved_aliases(sql)).fetchall()
                finally:
                    con.close()
        except Exception as e:
            print(f"  warmup query '{k}' failed: {e}")
    print("Warmup: done.")


def run_baseline(args):
    """Measure the *original* runtime of queries that had no rule applied.

    The final-execution run (``rule_summary_result.json``) only measures queries
    where at least one rule fired; queries with no applicable rule are never
    timed. Those queries still count towards the *runtime-weighted* whole-workload
    improvement on the ``*_speedup_bar_all`` plots (as denominator weight, since
    their rewritten runtime equals their original). This mode fills that gap.

    Reads ``--input`` (a JSON dict ``{query_name: sql}``; a transfer.json-style
    ``{name: {"sql": ...}}`` value is also accepted) and writes ``--output`` as
    ``{query_name: {"original_query": median_s, "original_runtimes": [per-attempt]}}``.

    Timing is identical to the measurement run: median of ``--attempts`` runs via
    ``execute_query``, honoring the ``--warmup-mode`` and ``--query-timeout``
    settings. Meant to run serial/exclusive, like the final execution.
    """
    with open(args.input, "r") as f:
        data = json.load(f)

    def _sql_of(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("sql")
        return None

    items: list[tuple[str, str]] = []
    for k in sorted(data.keys()):
        sql = _sql_of(data[k])
        if sql:
            items.append((k, sql))

    print(f"Baseline: measuring original runtime of {len(items)} no-rule queries "
          f"(engine={args.engine}, attempts={args.attempts}).")
    out: dict = {}
    for name, sql in items:
        try:
            _, median_time, _, runtimes = execute_query(
                sql, db_file=args.db_file, attempts=args.attempts)
        except QueryExecutionError as exc:
            _log_execution_error(name, "baseline", exc)
            continue
        except Exception as exc:  # unexpected engine error — log and keep going
            _log_execution_error(name, "baseline", exc)
            continue
        out[name] = {
            # 7 decimals (0.1 us) not 4 (0.1 ms): a fast engine like Umbra runs
            # the most selective queries in single-digit microseconds, which
            # round(_, 4) collapsed to 0.0 — a genuine sub-0.1 ms runtime, not a
            # failed measurement. A 0.0 here is later dropped from the workload
            # population (percent_saved needs a nonzero original), so the coarse
            # rounding silently shrank Umbra's denominator.
            "original_query": round(median_time, 7),
            "original_runtimes": [round(t, 7) for t in runtimes],
        }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Baseline: wrote {len(out)} runtimes to {args.output}.")
    _report_execution_errors()


# ---------------------------------------------------------------------------
# ZeroShot-aggregate mode: select rule subsets by learned predicted runtime
# ---------------------------------------------------------------------------

def _explain_verbose_postgres(query: str) -> "list[str] | None":
    """Run EXPLAIN (VERBOSE) (no execution); return the plan as text lines.

    Feeds the vendored cross_db_benchmark parser in the scorer. Returns None on failure.
    """
    assert _postgres is not None
    try:
        cur = _postgres._con.cursor()
        cur.execute(f"EXPLAIN (VERBOSE) {query}")
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]
    except Exception as exc:
        print(f"Postgres EXPLAIN (VERBOSE) failed: {exc}")
        print(f"  For query: {query[:200]}")
        try:
            _postgres._con.rollback()
        except Exception:
            pass
        return None


def _enumerate_candidate_sqls(original_sql, fired_rules, *, original_plan, engine,
                              fix_join_order, max_subsets):
    """Enumerate UNIQUE candidate rewrites, deduped by resulting SQL.

    Returns a list of {"indices": [...], "exec_sql": str}; index [] is the original.
    Subsets that produce identical refined SQL collapse to one candidate (the
    "unnecessary" subsets) — the smallest index set producing a given SQL is kept.
    When the number of unique candidates exceeds *max_subsets*, the original, all
    singletons, and the all-rules subset are kept unconditionally; the remaining
    budget is filled with the largest subsets first (the all-rules SQL is never dropped).
    """
    from itertools import combinations as _combos

    def _exec_for(refined_sql):
        s, _ = build_execution_sql(original_sql, refined_sql, original_plan,
                                   engine=engine, fix_join_order=fix_join_order)
        return s

    n = len(fired_rules)
    unique: dict[str, tuple] = {}  # refined_sql -> smallest combo producing it
    for r in range(1, n + 1):
        for combo in _combos(range(n), r):
            rules = [fired_rules[i]["rule"] for i in combo]
            refined_sql, has_new = apply_sql_rules(original_sql, rules)
            if not has_new or refined_sql == original_sql:
                continue
            if refined_sql not in unique:
                unique[refined_sql] = combo

    items = list(unique.items())  # (refined_sql, combo)
    if len(items) > max_subsets:
        all_rules_sql, _ = apply_sql_rules(original_sql, [fr["rule"] for fr in fired_rules])
        keep_keys = {s for s, c in items if len(c) == 1}
        keep_keys.add(all_rules_sql)
        kept = [(s, c) for s, c in items if s in keep_keys]
        rest = sorted((it for it in items if it[0] not in keep_keys),
                      key=lambda x: len(x[1]), reverse=True)
        kept += rest[: max(0, max_subsets - len(kept))]
        print(f"  capped to {len(kept)} unique candidates (max_subsets={max_subsets}); "
              f"kept singletons + all-rules + largest")
        items = kept

    candidates = [{"indices": [], "exec_sql": _exec_for(original_sql)}]
    for refined_sql, combo in items:
        candidates.append({"indices": list(combo), "exec_sql": _exec_for(refined_sql)})
    return candidates


def run_zeroshot_aggregate(args):
    """Select, per query, the rule subset with lowest learned-model *predicted runtime*.

    Enumerates unique candidate rewrites, runs EXPLAIN (VERBOSE) for each, hands them to
    the external scorer (isolated venv) for batched prediction, and picks the
    lowest-predicted subset. Emits rule_summary_transfer.json with the SAME shape as
    run_cost_aggregate (predicted seconds stand in for "cost"), so final execution and
    statistics work unchanged. Honors --keep-full-pool for the oracle/optimizer view.
    """
    import subprocess
    import tempfile

    if apply_sql_rules is None:
        print("ERROR: apply_sql_rules not available — cannot run zeroshot aggregation")
        return
    if _postgres is None:
        print("ERROR: --mode zeroshot-aggregate requires --engine postgres")
        return
    for req in ("scorer_python", "scorer_script", "zeroshot_model_type",
                "zeroshot_model_dir", "zeroshot_statistics_file", "zeroshot_database_stats"):
        if not getattr(args, req, None):
            print(f"ERROR: --{req.replace('_', '-')} is required for zeroshot-aggregate")
            return

    with open(args.input, "r") as f:
        data = json.load(f)
    rule_entries = data["rules"]
    queries = data["queries"]

    # 1) enumerate unique candidates + EXPLAIN VERBOSE
    candidates: dict[str, dict] = {}   # cand_id -> {query, indices, verbose_plan, sql}
    per_query: dict[str, dict] = {}    # sql_name -> {sql, fired_rules, by_indices}
    for sql_name, sql in tqdm(queries.items(), desc="ZeroShot: EXPLAIN candidates"):
        fired_rules = []
        for entry in rule_entries:
            rule = entry.get("rule")
            if not rule:
                continue
            _, fired = apply_sql_rules(sql, [rule])
            if fired:
                fired_rules.append({"name": entry.get("name"), "rule": rule})
        if not fired_rules:
            continue
        _, original_plan = estimate_query_cost(sql, db_file=args.db_file)
        cand_list = _enumerate_candidate_sqls(
            sql, fired_rules, original_plan=original_plan, engine=args.engine,
            fix_join_order=args.fix_join_order, max_subsets=args.max_subsets,
        )
        by_indices = {}
        for cand in cand_list:
            verbose = _explain_verbose_postgres(cand["exec_sql"])
            if verbose is None:
                continue
            idxsig = ",".join(map(str, cand["indices"])) if cand["indices"] else "orig"
            cand_id = f"{sql_name}::{idxsig}"
            candidates[cand_id] = {"query": sql_name, "indices": cand["indices"],
                                   "verbose_plan": verbose, "sql": cand["exec_sql"]}
            by_indices[tuple(cand["indices"])] = cand_id
        per_query[sql_name] = {"sql": sql, "fired_rules": fired_rules, "by_indices": by_indices}

    if not candidates:
        print("No candidates to score.")
        with open(args.output, "w") as f:
            json.dump({}, f, indent=4)
        return

    # 2) external scorer (separate venv): candidates.json -> predictions.json
    with tempfile.TemporaryDirectory() as td:
        cand_path = os.path.join(td, "candidates.json")
        pred_path = os.path.join(td, "predictions.json")
        with open(cand_path, "w") as f:
            json.dump(candidates, f, default=_json_default)
        cmd = [
            args.scorer_python, args.scorer_script,
            "--mode", "predict-candidates",
            "--input", cand_path, "--output", pred_path,
            "--database-stats", args.zeroshot_database_stats,
            "--model-type", args.zeroshot_model_type,
            "--model-dir", args.zeroshot_model_dir,
            "--seed", str(args.zeroshot_seed),
            "--statistics-file", args.zeroshot_statistics_file,
        ]
        print(f"Running scorer:\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        with open(pred_path, "r") as f:
            predictions = json.load(f)

    # 3) per query: pick lowest-predicted subset; emit run_cost_aggregate-shaped output
    result = {}
    for sql_name, info in per_query.items():
        sql = info["sql"]
        fired_rules = info["fired_rules"]
        by_indices = info["by_indices"]

        orig_id = by_indices.get(())
        orig_pred = predictions.get(orig_id) if orig_id else None

        scored = []  # (indices_list, pred_seconds)
        for idx_tuple, cand_id in by_indices.items():
            p = predictions.get(cand_id)
            if p is None:
                continue
            scored.append((list(idx_tuple), float(p)))
        if not scored:
            continue
        scored.sort(key=lambda x: x[1])

        best_indices, best_pred = scored[0]
        win_cutoff = (orig_pred * (1.0 - args.cost_save_threshold)) if orig_pred is not None else None
        if not best_indices or (win_cutoff is not None and best_pred >= win_cutoff):
            best_indices, best_pred = [], (orig_pred if orig_pred is not None else best_pred)

        explored = [{"indices": idx, "rules": [fired_rules[i]["name"] for i in idx],
                     "pred_s": round(p, 6)} for idx, p in scored]
        per_rule = [{"name": fired_rules[idx[0]]["name"], "pred_s": round(p, 6),
                     "pred_reduction_s": (round(orig_pred - p, 6) if orig_pred is not None else None)}
                    for idx, p in scored if len(idx) == 1]
        winner_names = [fired_rules[i]["name"] for i in best_indices]
        best_exec_sql = (candidates[by_indices[tuple(best_indices)]]["sql"]
                         if tuple(best_indices) in by_indices else sql)

        cost_estimation = {
            "cost_model": "zeroshot",
            "original_cost": round(orig_pred, 6) if orig_pred is not None else None,
            "cost_save_threshold": args.cost_save_threshold,
            "fix_join_order": bool(args.fix_join_order),
            "per_rule": per_rule,
            "combo_search": {
                "explored_count": len(explored),
                "explored": explored,
                "winner_rules": winner_names,
                "winner_cost": round(best_pred, 6) if best_pred is not None else None,
            },
        }

        if args.keep_full_pool:
            all_rule_dicts = [e["rule"] for e in fired_rules]
            refined_sql, _ = apply_sql_rules(sql, all_rule_dicts)
            if best_indices:
                cost_estimation["final_cost"] = round(best_pred, 6)
                cost_estimation["execution_sql"] = best_exec_sql
            cost_estimation["full_pool_kept"] = True
            print(f"  {sql_name}: learned winner {winner_names or '(none)'}, "
                  f"pred={best_pred} (scored {len(explored)}); keeping full pool of "
                  f"{len(fired_rules)} rules")
            result[sql_name] = {
                "sql": sql, "rules": fired_rules,
                "refined_sql": refined_sql, "cost_estimation": cost_estimation,
            }
            continue

        kept_rules = [fired_rules[i] for i in best_indices]
        rejected_rules = [fired_rules[i] for i in range(len(fired_rules)) if i not in set(best_indices)]
        if not kept_rules:
            print(f"  {sql_name}: no predicted-faster subset")
            result[sql_name] = {
                "sql": sql, "rules": [], "rules_rejected": rejected_rules,
                "cost_estimation": cost_estimation,
            }
            continue
        refined_sql, _ = apply_sql_rules(sql, [e["rule"] for e in kept_rules])
        cost_estimation["final_cost"] = round(best_pred, 6)
        cost_estimation["execution_sql"] = best_exec_sql
        print(f"  {sql_name}: learned winner {winner_names}, pred={best_pred} "
              f"(scored {len(explored)})")
        result[sql_name] = {
            "sql": sql, "rules": kept_rules, "rules_rejected": rejected_rules,
            "refined_sql": refined_sql, "cost_estimation": cost_estimation,
        }

    n_win = sum(1 for v in result.values()
                if v.get("cost_estimation", {}).get("combo_search", {}).get("winner_rules"))
    print(f"\nTotal: {n_win} of {len(queries)} queries have a predicted-faster subset")
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4, default=_json_default)


def _merge_contention(data: dict, samples_path: str = "contention_samples.log") -> None:
    """Annotate each measured entry with the server contention during its
    ``summary.exec_window``, read from resource_monitor.py's samples log.

    Runs once at end of the execute run, after all measurement — so it adds zero
    overhead to query timing. Best-effort: silently returns when the log is
    absent (monitor disabled / non-umbra engine). Writes, per entry:
    ``summary.contention = {sibling_busy_max, sibling_busy_mean, pin_khz_min,
    n_samples}``.
    """
    import bisect

    if not os.path.exists(samples_path):
        return
    walls: list[float] = []
    busy: list[float | None] = []
    khz: list[int | None] = []
    try:
        with open(samples_path) as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                parts = line.rstrip("\n").split(",")
                if len(parts) < 4:
                    continue
                try:
                    walls.append(float(parts[1]))
                except ValueError:
                    continue
                busy.append(float(parts[2]) if parts[2] else None)
                khz.append(int(parts[3]) if parts[3] else None)
    except OSError:
        return
    if not walls:
        return

    n_annotated = 0
    for entry in data.values():
        summ = entry.get("summary")
        if not isinstance(summ, dict):
            continue
        win = summ.get("exec_window")
        if not (isinstance(win, list) and len(win) == 2):
            continue
        lo = bisect.bisect_left(walls, win[0])
        hi = bisect.bisect_right(walls, win[1])
        b = [x for x in busy[lo:hi] if x is not None]
        k = [x for x in khz[lo:hi] if x is not None]
        summ["contention"] = {
            "sibling_busy_max": round(max(b), 4) if b else None,
            "sibling_busy_mean": round(sum(b) / len(b), 4) if b else None,
            "pin_khz_min": min(k) if k else None,
            "n_samples": hi - lo,
        }
        n_annotated += 1
    print(f"Contention merge: annotated {n_annotated} queries from {len(walls)} samples "
          f"({samples_path}).")


def main():
    parser = argparse.ArgumentParser(description="Execute SQL queries and compare results.")
    parser.add_argument("--mode", default="execute",
                        choices=["execute", "cost-aggregate", "zeroshot-aggregate",
                                 "optimizer-winners", "warmup", "baseline"],
                        help="Mode: 'execute' runs queries; 'cost-aggregate' does cost-filtered rule application; "
                             "'zeroshot-aggregate' selects rule subsets by learned-model predicted runtime "
                             "(EXPLAIN VERBOSE + external scorer, no execution); "
                             "'optimizer-winners' picks the cost-winning rule subset per query (EXPLAIN only, no "
                             "execution) and writes a winners sidecar; "
                             "'warmup' runs each query in --input once (no measurement) to prime caches; "
                             "'baseline' measures the original runtime of no-rule queries (median of --attempts) "
                             "for the runtime-weighted whole-workload metric.")
    parser.add_argument("--input", required=False, default=None,
                        help="Input JSON file. Required for execute/cost-aggregate; "
                             "optional for --mode warmup when --sql-dir is given.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument("--db-file", default="imdb.duckdb", help="DuckDB database file.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Performance improvement threshold (fraction).")
    parser.add_argument("--cost-save-threshold", type=float, default=0.0,
                        help="Minimum relative cost saving (fraction) required for a rule subset to win in "
                             "cost-aggregate mode; 0.0 (default) = any reduction, 0.01 = >=1%% saving.")
    parser.add_argument("--keep-full-pool", action="store_true", default=False,
                        help="In cost-aggregate mode, keep the full fired-rule pool per query (instead of "
                             "pruning to the cost-winning subset) and record the winner only as metadata. "
                             "Required for oracle/both runs so final-exec enumerates the full powerset; the "
                             "optimizer view selects the cost winner downstream in statistics.")
    parser.add_argument("--attempts", type=int, default=3, help="Number of timing attempts per query.")
    parser.add_argument("--validation-mode", default="query", choices=["query", "base_tables", "both"],
                        help="Validation mode: 'query' (original vs transformed), 'base_tables' (rule on full data), or 'both'.")
    parser.add_argument("--engine", default="duckdb", choices=["duckdb", "umbra", "postgres"],
                        help="Execution engine: 'duckdb' (default), 'umbra', or 'postgres' (pg_lab).")
    parser.add_argument("--umbra-port", type=int, default=5432,
                        help="Host port the Umbra container is listening on (default: 5432).")
    parser.add_argument("--umbra-memory-gb", type=float, default=0.0,
                        help="Hard RAM cap (GB) for a self-managed Umbra container; only applies "
                             "when this process launches the container (0 = unlimited). In the "
                             "pipeline the container is pre-started by umbra_setup.py, which "
                             "receives the cap directly.")
    parser.add_argument("--postgres-host", default="127.0.0.1",
                        help="Postgres (pg_lab) host (default: 127.0.0.1).")
    parser.add_argument("--postgres-port", type=int, default=5432,
                        help="Host port the pg_lab container is listening on (default: 5432).")
    parser.add_argument("--postgres-user", default="postgres",
                        help="Postgres user (default: postgres).")
    parser.add_argument("--postgres-password", default="postgres",
                        help="Postgres password (default: postgres).")
    parser.add_argument("--postgres-dbname", default="imdb",
                        help="Postgres target database (default: imdb).")
    parser.add_argument("--fix-join-order", action="store_true", default=False,
                        help="Run refined queries with the original query's join order "
                             "(DuckDB: reconstruct SQL; Postgres: pg_lab JoinOrder hint).")
    parser.add_argument("--warmup-mode", default="per_query",
                        choices=["per_query", "startup_only", "startup_and_per_query", "none"],
                        help="Warmup behavior: 'per_query' (default — 1 warmup before each query), "
                             "'startup_only' (no per-query warmup; assumes --mode warmup ran once after "
                             "container start), 'startup_and_per_query' (both: startup prime + per-query "
                             "warmup), or 'none' (no warmup at all).")
    parser.add_argument("--warmup-limit", type=int, default=0,
                        help="Cap on number of queries to warm up in --mode warmup. "
                             "0 (default) = no cap, run all.")
    parser.add_argument("--sql-dir", default=None,
                        help="Directory of .sql files to use as warmup input in --mode warmup "
                             "(takes precedence over --input).")
    parser.add_argument("--max-temp-size", default="0",
                        help="Cap on DuckDB's on-disk spill (e.g. '50GB'). Bounds runaway "
                             "queries so they fail catchably instead of OOM/disk-killing the "
                             "process. '0' (default) = unlimited.")
    parser.add_argument("--query-timeout", type=float, default=0.0,
                        help="Per-query wall-clock timeout in seconds. A single query run "
                             "exceeding this is interrupted and its rule is skipped/discarded. "
                             "0 (default) = no timeout.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Validation parallelism (DuckDB only). >1 processes queries "
                             "concurrently for correctness checking; 0 = auto (half the "
                             "node's cores). Leave at 1 for the performance-measurement run "
                             "so timing stays reproducible.")
    # --- zeroshot-aggregate (learned cost model) options ---
    parser.add_argument("--max-subsets", type=int, default=4096,
                        help="zeroshot-aggregate: cap on UNIQUE candidate subsets scored per query "
                             "(after dedup). Singletons + the all-rules subset are never dropped.")
    parser.add_argument("--scorer-python", default=None,
                        help="zeroshot-aggregate: python interpreter of the scorer venv.")
    parser.add_argument("--scorer-script", default=None,
                        help="zeroshot-aggregate: path to scorer/score_plans.py.")
    parser.add_argument("--zeroshot-model-type", default=None)
    parser.add_argument("--zeroshot-model-dir", default=None,
                        help="Checkpoint dir, e.g. .../models/pm-zeroshot-hi-limitpf10001/imdb")
    parser.add_argument("--zeroshot-seed", type=int, default=9)
    parser.add_argument("--zeroshot-statistics-file", default=None)
    parser.add_argument("--zeroshot-database-stats", default=None,
                        help="JSON with {'database_stats': {...}} for the IMDB instance.")

    args = parser.parse_args()

    global _WARMUP_MODE, _MAX_TEMP_SIZE, _QUERY_TIMEOUT_S, _DUCKDB_THREADS
    _WARMUP_MODE = args.warmup_mode
    _MAX_TEMP_SIZE = args.max_temp_size
    _QUERY_TIMEOUT_S = args.query_timeout

    # Resolve validation parallelism. 0 = auto (half the node's cores). Parallel
    # validation is DuckDB-only; force serial for other engines (they share a
    # single connection / single-core container). Concurrent queries also make
    # wall-clock timings non-authoritative, which is fine for correctness-only
    # validation but not for the perf-measurement run (which passes --workers 1).
    if args.workers == 0:
        args.workers = max(1, (os.cpu_count() or 2) // 2)
    if args.workers > 1 and args.engine != "duckdb":
        print(f"Warning: --workers {args.workers} ignored for engine "
              f"'{args.engine}' (parallel validation is DuckDB-only); running serial.")
        args.workers = 1
    if args.workers > 1:
        _DUCKDB_THREADS = args.workers
        if args.validation_mode in ("query", "both"):
            print("Warning: parallel validation makes query-level/per-subset timings "
                  "non-authoritative (correctness is unaffected). Use --workers 1 if "
                  "this run relies on validation-phase timing (e.g. time_filtering).")

    # Initialize Umbra if requested (connects + loads data idempotently)
    if args.engine == "umbra":
        init_umbra(args.db_file, port=args.umbra_port, memory_gb=args.umbra_memory_gb)
    elif args.engine == "postgres":
        init_postgres(
            args.db_file,
            host=args.postgres_host, port=args.postgres_port,
            user=args.postgres_user, password=args.postgres_password,
            dbname=args.postgres_dbname,
        )

    if args.mode == "cost-aggregate":
        run_cost_aggregate(args)
        return

    if args.mode == "zeroshot-aggregate":
        run_zeroshot_aggregate(args)
        return

    if args.mode == "optimizer-winners":
        run_optimizer_winners(args)
        return

    if args.mode == "warmup":
        run_warmup(args)
        return

    if args.mode == "baseline":
        run_baseline(args)
        return

    with open(args.input, "r") as file:
        data = json.load(file)

    do_query = args.validation_mode in ("query", "both")
    do_base_tables = args.validation_mode in ("base_tables", "both")

    def _process_entry(sql_name, sql_data):
        # --- Query-level validation ---
        if do_query:
            original_query = sql_data["sql"]
            if "refined_sql" not in sql_data:
                sql_data["rule_applied"] = False
                return
            sql_data["rule_applied"] = True
            llm_transformed_query = sql_data["refined_sql"]

            _win_start = time.time()  # wall-clock start of this query's measurement
            # The original query is the baseline every number here is relative
            # to: if it cannot run there is nothing to measure, so record the
            # failure and move on rather than booking a 0.0s "match".
            try:
                result_1, time_1, profile_1, runtimes_1 = execute_query(
                    original_query, db_file=args.db_file, attempts=args.attempts)
            except QueryExecutionError as e:
                sql_data["execution_error"] = _log_execution_error(
                    sql_name, "original_query", e)
                return

            # With fix_join_order this either pins the plan or raises — there is
            # no silent un-pinned fallback (see build_execution_sql).
            exec_sql, plan_pinned = build_execution_sql(
                original_query, llm_transformed_query, profile_1,
                engine=args.engine, fix_join_order=args.fix_join_order,
            )
            fixed_sql = exec_sql if plan_pinned else None
            # A rewritten query that will not run is a failed rewrite, not a
            # failed workload: record it and keep going.
            try:
                result_2, time_2, profile_2, runtimes_2 = execute_query(
                    exec_sql, db_file=args.db_file, attempts=args.attempts,
                    disable_join_order=plan_pinned,
                )
            except QueryExecutionError as e:
                sql_data["execution_error"] = _log_execution_error(
                    sql_name, "refined_sql", e)
                return
            # Multiset comparison: row order is irrelevant, row multiplicity is
            # not.  A rewrite that drops or adds duplicates is incorrect even
            # when the distinct rows agree — with strip_min the workload queries
            # project join output, so duplicates are part of the result.
            # Counter differences keep multiplicity, so a row returned three
            # times too often shows up three times in the diff below.
            ms1 = Counter(result_1)
            ms2 = Counter(result_2)

            results_match = ms1 == ms2
            only_in_original = list((ms1 - ms2).elements())
            only_in_llm_made = list((ms2 - ms1).elements())
            performance_improvement = time_1 - time_2 > args.threshold * time_1

            head = {
                "execution_time": {
                    # 7 decimals (0.1 us): see the note in run_baseline — round(_, 4)
                    # zeroed genuine microsecond runtimes on a fast engine (Umbra).
                    "original_query": round(time_1, 7),
                    "llm_transformed_query": round(time_2, 7),
                    "original_runtimes": [round(t, 7) for t in runtimes_1],
                    "llm_transformed_runtimes": [round(t, 7) for t in runtimes_2],
                    "performance_improvement": performance_improvement,
                },
                "outputs_match": results_match,
                "match_and_improvement": results_match and performance_improvement,
            }

            results = {
                "queries": {
                    "diff": {
                        "only_in_original": only_in_original[:100],
                        "only_in_llm_made": only_in_llm_made[:100],
                    },
                    "original_query": {"result": result_1[:100], "query_plan": profile_1},
                    "llm_transformed_query": {"result": result_2[:100], "query_plan": profile_2},
                }
            }

            sql_data["summary"] = head
            sql_data["results"] = results
            if fixed_sql is not None:
                sql_data["fixed_join_order_sql"] = fixed_sql

            # --- Per-subset execution (all rule combinations) ---
            fired_rules = sql_data.get("rules", [])
            if fired_rules and apply_sql_rules is not None:
                per_subset_results = []
                k = len(fired_rules)
                for size in range(1, k + 1):
                    for combo in combinations(fired_rules, size):
                        rule_names = [e["name"] for e in combo]
                        if size == k:
                            # Size-k (all-rules) subset: reuse call #2's
                            # measurements rather than re-executing the same
                            # SQL. Log when the subset SQL would have differed
                            # from refined_sql (rule-chaining case, where
                            # eligible_rules ⊋ fired_rules).
                            would_be_sql, _ = apply_sql_rules(
                                original_query, [e["rule"] for e in fired_rules]
                            )
                            if would_be_sql != llm_transformed_query:
                                print(
                                    f"  Note: size-k subset SQL would differ from "
                                    f"refined_sql for {sql_name} — likely rule "
                                    f"chaining; reusing refined_sql."
                                )
                            r_res = result_2
                            r_time = time_2
                            r_profile = profile_2
                            r_match = results_match
                            r_improvement = performance_improvement
                        else:
                            subset_sql, subset_fired = apply_sql_rules(
                                original_query, [e["rule"] for e in combo]
                            )
                            if not subset_fired:
                                continue
                            subset_exec_sql, subset_pinned = build_execution_sql(
                                original_query, subset_sql, profile_1,
                                engine=args.engine, fix_join_order=args.fix_join_order,
                            )
                            try:
                                r_res, r_time, r_profile, r_runtimes = execute_query(
                                    subset_exec_sql, db_file=args.db_file, attempts=args.attempts,
                                    disable_join_order=subset_pinned,
                                )
                            except QueryExecutionError as e:
                                # An unrunnable subset must not count as a 0.0s
                                # win for the oracle bound — drop it.
                                sql_data.setdefault("subset_execution_errors", []).append(
                                    dict(rule_names=rule_names, **_log_execution_error(
                                        f"{sql_name} [{' + '.join(rule_names)}]",
                                        "subset_sql", e)))
                                continue
                            r_match = Counter(r_res) == ms1
                            r_improvement = time_1 - r_time > args.threshold * time_1
                        per_subset_results.append({
                            "rule_names": rule_names,
                            "summary": {
                                "execution_time": {
                                    # 7 decimals (0.1 us): see the note in run_baseline.
                                    "original_query": round(time_1, 7),
                                    "subset_transformed_query": round(r_time, 7),
                                    "performance_improvement": r_improvement,
                                },
                                "outputs_match": r_match,
                                "match_and_improvement": r_match and r_improvement,
                            },
                            "results": {
                                "queries": {
                                    "original_query": {
                                        "result": result_1[:100],
                                        "query_plan": profile_1,
                                    },
                                    "subset_transformed_query": {
                                        "result": r_res[:100],
                                        "query_plan": r_profile,
                                    },
                                }
                            },
                        })
                sql_data["per_subset_results"] = per_subset_results

            # Wall-clock window covering this query's whole measurement (original
            # + rewrite + every subset). Merged with the contention samples log at
            # end of run so each query can be annotated with how busy the server
            # was while it was measured (trust scoring, statistics.py).
            sql_data["summary"]["exec_window"] = [_win_start, time.time()]

        # --- Base-table validation ---
        if do_base_tables:
            bt_result = run_base_table_validation(sql_data, args.db_file, attempts=args.attempts)
            if bt_result is not None:
                sql_data["base_table_validation"] = bt_result

    if args.workers > 1:
        # Parallel validation: process queries concurrently on a shared node.
        # Each entry mutates its own (distinct) dict key, so this is GIL-safe;
        # the json.dump below runs only after every task has joined.
        with ThreadPoolExecutor(
            max_workers=args.workers, initializer=_init_validation_worker
        ) as pool:
            futures = [
                pool.submit(_process_entry, name, sql_data)
                for name, sql_data in data.items()
            ]
            for fut in tqdm(
                as_completed(futures), total=len(futures),
                desc="Processing SQL queries",
            ):
                fut.result()  # re-raise any worker exception
    else:
        for sql_name, sql_data in tqdm(data.items(), desc="Processing SQL queries"):
            _process_entry(sql_name, sql_data)

    # Discard entries where no rule was applied (no refined_sql)
    data = {k: v for k, v in data.items() if v.get("rule_applied", True)}

    # Annotate each query with the server contention measured during its window
    # (no-op when the resource monitor did not run / no samples log present).
    _merge_contention(data)

    with open(args.output, "w") as json_file:
        json.dump(data, json_file, indent=4, default=_json_default)

    _report_execution_errors()


if __name__ == "__main__":
    main()
