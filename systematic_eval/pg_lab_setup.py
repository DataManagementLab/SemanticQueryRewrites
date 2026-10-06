#!/usr/bin/env python3
"""Standalone pg_lab Docker lifecycle manager.

Self-contained (no project imports) — scp'd to the remote server alongside
execution.py. Handles building the pg_lab image (if missing), starting and
stopping the container, and applying a custom postgresql.conf via
`ALTER SYSTEM` + restart (works without knowing the image's internal layout).

Data loading is handled by execution.py's init_postgres() on first run.

Usage:
    python3 pg_lab_setup.py --start --conf postgresql16.conf
    python3 pg_lab_setup.py --start --conf postgresql16.conf --set random_page_cost=1.1
    python3 pg_lab_setup.py --teardown
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONTAINER = "pg_lab"
IMAGE = "pg_lab"
DATA_DIR = Path.home() / "pg_lab-db"
SRC_DIR = Path.home() / "pg_lab-src"
PG_LAB_REPO = "https://github.com/Optimizer-Playground/pg_lab"
PGVER = "16"
PIN_CORE = 4

# Keys from the conf file that are environment-specific and must NOT be
# applied via ALTER SYSTEM (they reference bare-metal paths that do not
# exist inside the container, or they must be set at startup, or they are
# reserved by the container image).
_SKIP_KEYS = {
    "data_directory", "hba_file", "ident_file", "external_pid_file",
    "unix_socket_directories", "cluster_name",
    # Reserved: not settable via ALTER SYSTEM, must come from the
    # container's own config file.  port is already the docker port mapping.
    "port",
}


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _image_exists(image: str) -> bool:
    return _run(["docker", "image", "inspect", image]).returncode == 0


def _build_image() -> None:
    if not SRC_DIR.exists():
        print(f"Cloning pg_lab into {SRC_DIR} ...")
        r = _run(["git", "clone", PG_LAB_REPO, str(SRC_DIR)])
        if r.returncode != 0:
            print(f"ERROR: git clone failed:\n{r.stderr}")
            sys.exit(1)

    timezone = "Europe/Berlin"
    tz_file = Path("/etc/timezone")
    if tz_file.exists():
        timezone = tz_file.read_text().strip() or timezone

    print(f"Building pg_lab image (PGVER={PGVER}, this takes a while) ...")
    r = subprocess.run(
        ["docker", "build",
         "--build-arg", f"TIMEZONE={timezone}",
         "-t", IMAGE, "."],
        cwd=str(SRC_DIR), check=False,
    )
    if r.returncode != 0:
        print("ERROR: docker build failed.")
        sys.exit(1)


def _find_pg_bin() -> str:
    """Locate the pg binary directory inside the container (where psql, pg_isready, ...
    live). pg_lab builds its own pg under /pg_lab, so the usual $PATH lookup fails.

    Retries while the container is still coming up — `docker run -d` can return
    before the entrypoint is ready to accept `docker exec`.
    """
    probe = (
        "for p in /pg_lab/postgres-pglab*/build/bin "
        "         /pg_lab/build/bin "
        "         /pg_lab/*/bin "
        "         /usr/lib/postgresql/*/bin "
        "         /usr/local/pgsql/bin; do "
        "  if [ -x \"$p/psql\" ]; then echo \"$p\"; exit 0; fi; "
        "done; "
        "find / -name psql -type f -executable 2>/dev/null "
        "  | head -1 | xargs -r -n1 dirname"
    )

    deadline = time.time() + 60.0
    last_r: subprocess.CompletedProcess | None = None
    while time.time() < deadline:
        if not _container_running():
            time.sleep(1.0)
            continue
        r = _run(["docker", "exec", CONTAINER, "bash", "-c", probe])
        last_r = r
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
        time.sleep(1.0)

    print("ERROR: could not locate pg binary directory inside the container.")
    if last_r is not None:
        print(f"  rc={last_r.returncode}  stdout: {last_r.stdout!r}\n  stderr: {last_r.stderr!r}")
    print(f"  container running: {_container_running()}")
    logs = _run(["docker", "logs", "--tail", "80", CONTAINER])
    print("Container logs (last 80 lines):")
    print(logs.stdout or logs.stderr or "(no logs)")
    sys.exit(1)


def _wait_socket_ready(timeout: float = 180.0) -> str:
    """Wait until pg is accepting connections on its unix socket inside the container.

    Returns the pg binary directory. Uses `pg_isready` via docker exec — does not
    depend on host-side TCP auth, so it succeeds before we've opened up pg_hba.conf.
    """
    pg_bin = _find_pg_bin()
    print(f"pg_lab_setup: pg binaries at {pg_bin}")

    deadline = time.time() + timeout
    last_out = ""
    while time.time() < deadline:
        r = _run(["docker", "exec", CONTAINER,
                  f"{pg_bin}/pg_isready"])
        if r.returncode == 0:
            return pg_bin
        last_out = (r.stdout + r.stderr).strip()
        time.sleep(1.0)
    print(f"ERROR: pg_lab unix socket not ready after {timeout:.0f}s: {last_out}")
    logs = _run(["docker", "logs", "--tail", "50", CONTAINER])
    print(logs.stdout or logs.stderr or "(no logs)")
    sys.exit(1)


def _ensure_postgres_role(pg_bin: str) -> None:
    """Ensure a `postgres` superuser exists (password 'postgres').

    pg_lab's initdb uses the container's default OS user as the bootstrap
    superuser — frequently not 'postgres'. The rest of our pipeline assumes
    the conventional postgres/postgres credentials, so we create that role
    here if it's missing.
    """
    psql = f"{pg_bin}/psql"

    # Who is the bootstrap superuser? (psql with no -U uses container's OS user)
    r = _run(["docker", "exec", CONTAINER, psql, "-d", "postgres",
              "-tAc", "SELECT current_user"])
    if r.returncode != 0 or not r.stdout.strip():
        print(f"ERROR: could not determine bootstrap superuser: "
              f"{r.stderr.strip() or r.stdout.strip()}")
        sys.exit(1)
    bootstrap_user = r.stdout.strip()
    print(f"pg_lab_setup: bootstrap superuser is '{bootstrap_user}'")

    if bootstrap_user == "postgres":
        return

    # Check whether a 'postgres' role already exists.
    r = _run(["docker", "exec", CONTAINER, psql, "-d", "postgres",
              "-tAc", "SELECT 1 FROM pg_roles WHERE rolname = 'postgres'"])
    if r.stdout.strip() == "1":
        return

    print("pg_lab_setup: creating 'postgres' superuser role ...")
    r = _run(["docker", "exec", CONTAINER, psql, "-d", "postgres",
              "-c", "CREATE ROLE postgres WITH SUPERUSER LOGIN PASSWORD 'postgres'"])
    if r.returncode != 0:
        print(f"ERROR: could not create postgres role: {r.stderr.strip()}")
        sys.exit(1)


def _open_pg_hba_for_docker_bridge(pg_bin: str) -> None:
    """Append a wildcard trust line to pg_hba.conf inside the container and reload."""
    psql = f"{pg_bin}/psql"

    # Use -d postgres without -U so psql defaults to the container's OS user
    # (works whether or not the 'postgres' role exists yet).
    r = _run(["docker", "exec", CONTAINER, psql, "-d", "postgres",
              "-tAc", "SHOW hba_file"])
    hba = r.stdout.strip()
    if r.returncode != 0 or not hba:
        print(f"ERROR: could not locate pg_hba.conf (SHOW hba_file failed): "
              f"{r.stderr.strip() or r.stdout.strip()}")
        sys.exit(1)

    r = _run(["docker", "exec", CONTAINER, psql, "-d", "postgres",
              "-tAc", "SHOW data_directory"])
    pgdata = r.stdout.strip()

    marker = "# pg_lab_setup: host access for docker bridge"
    bash = (
        f"if ! grep -q '{marker}' {hba}; then "
        f"  printf '\\n%s\\nhost all all 0.0.0.0/0 trust\\nhost all all ::/0 trust\\n' "
        f"    '{marker}' >> {hba}; "
        f"fi"
    )
    r = _run(["docker", "exec", CONTAINER, "bash", "-c", bash])
    if r.returncode != 0:
        print(f"ERROR: failed to append to pg_hba.conf: {r.stderr.strip()}")
        sys.exit(1)

    if pgdata:
        _run(["docker", "exec", CONTAINER,
              f"{pg_bin}/pg_ctl", "reload", "-D", pgdata])
    else:
        _run(["docker", "exec", CONTAINER, psql, "-d", "postgres",
              "-c", "SELECT pg_reload_conf()"])


def _wait_ready(port: int, timeout: float = 180.0) -> None:
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            con = psycopg2.connect(
                host="127.0.0.1", port=port,
                user="postgres", password="postgres",
                dbname="postgres",
            )
            con.close()
            return
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1.0)

    print(f"ERROR: pg_lab not ready after {timeout:.0f}s. Last error: {last_err}")
    print("\nContainer logs:")
    logs = _run(["docker", "logs", "--tail", "50", CONTAINER])
    print(logs.stdout or logs.stderr or "(no logs)")
    sys.exit(1)


def _unquote_conf_value(raw: str) -> str:
    """Strip surrounding single quotes from a postgresql.conf value, unescaping ''."""
    v = raw.strip()
    if len(v) >= 2 and v.startswith("'") and v.endswith("'"):
        v = v[1:-1].replace("''", "'")
    return v


def _parse_conf(conf_path: Path) -> list[tuple[str, str]]:
    """Parse a postgresql.conf into an ordered list of (key, value) pairs.

    Values are returned *unquoted* (outer '...' stripped, '' un-escaped) so callers
    can requote them correctly for whichever SQL syntax they use.
    """
    out: list[tuple[str, str]] = []
    line_re = re.compile(r"^\s*([a-zA-Z0-9_.]+)\s*=\s*(.+?)\s*(?:#.*)?$")
    for raw in conf_path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = line_re.match(raw)
        if not m:
            continue
        out.append((m.group(1), _unquote_conf_value(m.group(2))))
    return out


def _filter_available_preload_libs(cur, value: str) -> str:
    """Given a comma-separated list of shared_preload_libraries values, drop any
    whose extension isn't present in pg_available_extensions. Prevents a broken
    restart when e.g. pg_hint_plan isn't built into the pg_lab image."""
    libs = [x.strip() for x in value.split(",") if x.strip()]
    kept = []
    for lib in libs:
        cur.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = %s", (lib,))
        if cur.fetchone() is not None:
            kept.append(lib)
        else:
            print(f"  skip: shared_preload_libraries entry '{lib}' — extension not available in this pg build")
    return ",".join(kept)


def _apply_conf(conf_path: Path | None, port: int,
                overrides: dict[str, str] | None = None) -> bool:
    """Apply each setting in *conf_path*, then *overrides*, via `ALTER SYSTEM SET`.

    *overrides* come from `--set KEY=VALUE` and are applied last, so they win over
    a same-named key in the conf file. Settings absent from both are never touched
    and therefore keep postgres' own default (`ALTER SYSTEM RESET ALL` below).

    Returns True if any setting was applied (meaning a restart is advised).
    """
    import psycopg2
    from psycopg2 import sql

    pairs = _parse_conf(conf_path) if conf_path is not None else []
    pairs += list((overrides or {}).items())
    con = psycopg2.connect(
        host="127.0.0.1", port=port,
        user="postgres", password="postgres",
        dbname="postgres",
    )
    con.autocommit = True
    cur = con.cursor()

    # Start from a clean slate so stale values from a previous (broken) run
    # don't linger in postgresql.auto.conf.
    cur.execute("ALTER SYSTEM RESET ALL")

    applied = 0
    skipped = 0
    for key, value in pairs:
        klow = key.lower()
        if klow in _SKIP_KEYS:
            skipped += 1
            continue
        if klow == "shared_preload_libraries":
            value = _filter_available_preload_libs(cur, value)
            if not value:
                skipped += 1
                continue
        try:
            cur.execute(
                sql.SQL("ALTER SYSTEM SET {} = {}").format(
                    sql.Identifier(key), sql.Literal(value)))
            applied += 1
        except Exception as exc:
            print(f"  warn: ALTER SYSTEM SET {key} = {value!r}  -> {exc}")
    cur.close()
    con.close()
    source = conf_path.name if conf_path is not None else "(no conf file)"
    print(f"pg_lab_setup: applied {applied} settings, skipped {skipped} "
          f"environment-specific keys from {source}.")
    if overrides:
        shown = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"pg_lab_setup: command-line overrides applied last: {shown}")
    return applied > 0


def _container_running() -> bool:
    r = _run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER])
    return r.returncode == 0 and r.stdout.strip() == "true"


def _clear_stale_auto_conf() -> None:
    """Remove any postgresql.auto.conf left behind in the persistent volume.

    Prevents the container from failing to start when a prior run wrote
    settings (e.g. a missing preload library) that the current pg build
    rejects. `_apply_conf` will recreate it with the current conf file.

    Runs the deletion *inside* a transient pg_lab container so it has the
    same uid as the main container — avoids host-side permission errors
    when the bind-mount is owned by the container user, not the host user.
    """
    if not DATA_DIR.exists() or not _image_exists(IMAGE):
        return
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{DATA_DIR.as_posix()}:/pg_lab",
        "--entrypoint", "bash",
        IMAGE,
        "-c",
        "find /pg_lab -name postgresql.auto.conf -print -delete 2>/dev/null || true",
    ]
    r = _run(cmd)
    removed = [line for line in r.stdout.splitlines() if line.strip()]
    if removed:
        print(f"pg_lab_setup: cleared {len(removed)} stale postgresql.auto.conf file(s):")
        for line in removed:
            print(f"  - {line}")


def _start_fresh_container(port: int) -> None:
    _run(["docker", "rm", "-f", CONTAINER])
    _clear_stale_auto_conf()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # pg_lab's image convention: /pg_lab is the in-container mount point
    # that houses both the cluster's data dir and any extras.
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER,
        "--cpus", "1",
        "--cpuset-cpus", str(PIN_CORE),
        "-v", f"{DATA_DIR.as_posix()}:/pg_lab",
        "-p", f"{port}:5432",
        "--env", f"PGVER={PGVER}",
        "--ulimit", "nofile=1048576:1048576",
        "--ulimit", "memlock=8388608:8388608",
        IMAGE,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        stderr = r.stderr.strip()
        if "permission denied" in stderr.lower():
            print("─" * 60)
            print("ERROR: Docker permission denied.")
            print("Fix:   sudo usermod -aG docker $USER && newgrp docker")
            print("─" * 60)
        else:
            print(f"ERROR: docker run failed:\n{stderr}")
        sys.exit(1)

    print(f"pg_lab container '{CONTAINER}' started on core {PIN_CORE}, port {port}.")


def start(port: int, conf_path: Path | None,
          overrides: dict[str, str] | None = None) -> None:
    if not _image_exists(IMAGE):
        print(f"pg_lab image '{IMAGE}' not found locally — building it.")
        _build_image()
        if not _image_exists(IMAGE):
            print(f"ERROR: image '{IMAGE}' still missing after build.")
            sys.exit(1)

    _start_fresh_container(port)
    print("Waiting for pg_lab's unix socket to accept connections ...")
    pg_bin = _wait_socket_ready()
    print("Opening pg_hba.conf for the docker-bridge host ...")
    _open_pg_hba_for_docker_bridge(pg_bin)
    print("Ensuring 'postgres' superuser role exists ...")
    _ensure_postgres_role(pg_bin)
    print("Waiting for pg_lab to accept TCP connections from the host ...")
    _wait_ready(port=port)

    if conf_path is not None and not conf_path.exists():
        print(f"WARN: conf file {conf_path} not found; using image defaults.")
        conf_path = None

    if conf_path is not None or overrides:
        src = str(conf_path) if conf_path is not None else "command-line overrides only"
        print(f"Applying settings from {src} ...")
        needs_restart = _apply_conf(conf_path, port, overrides)
        if needs_restart:
            print("Restarting pg_lab so restart-required settings take effect ...")
            _run(["docker", "restart", CONTAINER])
            _wait_ready(port=port)

    print("pg_lab is ready.")


def teardown() -> None:
    _run(["docker", "stop", CONTAINER])
    _run(["docker", "rm", "-f", CONTAINER])
    print(f"pg_lab container '{CONTAINER}' stopped and removed.")


def _parse_overrides(raw: list[str] | None) -> dict[str, str]:
    """Turn repeated `--set key=value` args into an ordered dict."""
    out: dict[str, str] = {}
    for item in raw or []:
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            print(f"ERROR: --set expects KEY=VALUE, got {item!r}")
            sys.exit(2)
        out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or stop the pg_lab Docker container.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Start the pg_lab container.")
    group.add_argument("--teardown", action="store_true", help="Stop and remove the pg_lab container.")
    parser.add_argument("--port", type=int, default=5432,
                        help="Host port to expose pg_lab on (default: 5432).")
    parser.add_argument("--conf", type=str, default="postgresql16.conf",
                        help="Path to postgresql.conf overrides (default: postgresql16.conf in CWD).")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", dest="settings",
                        help="Extra setting applied via ALTER SYSTEM after --conf, overriding it "
                             "(e.g. --set random_page_cost=1.1). Repeatable. Settings not given "
                             "here or in --conf keep postgres' default.")
    args = parser.parse_args()

    if args.start:
        start(port=args.port,
              conf_path=Path(args.conf).resolve() if args.conf else None,
              overrides=_parse_overrides(args.settings))
    else:
        teardown()


if __name__ == "__main__":
    main()
