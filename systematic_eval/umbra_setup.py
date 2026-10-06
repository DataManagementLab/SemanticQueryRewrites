#!/usr/bin/env python3
"""Standalone Umbra Docker lifecycle manager.

Self-contained (no project imports) — scp'd to the remote server alongside
execution.py.  Handles starting and stopping the Umbra container only; data
loading is handled by execution.py's init_umbra() on first run.

Usage:
    python3 umbra_setup.py --start
    python3 umbra_setup.py --teardown
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONTAINER = "umbradb"
IMAGE = "umbradb/umbra:latest"
DATA_DIR = str(Path.home() / "umbra-db")
PIN_CORE = 4

# Lightweight server-contention sampler (see resource_monitor.py). Launched at
# --start, stopped at --teardown. The samples log is written to the current
# working directory (the remote workspace) so execution.py can merge it, and its
# PID lives under DATA_DIR so a later --teardown can find it.
MONITOR_ENABLED = True
MONITOR_SCRIPT = "resource_monitor.py"
SAMPLES_LOG = "contention_samples.log"
MONITOR_PIDFILE = str(Path(DATA_DIR) / ".monitor.pid")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _numa_node_of(core: int) -> int | None:
    """NUMA node id of a logical CPU from sysfs, or None if undeterminable."""
    for p in Path(f"/sys/devices/system/cpu/cpu{core}").glob("node[0-9]*"):
        try:
            return int(p.name[len("node"):])
        except ValueError:
            continue
    return None


def _start_monitor(pin_core: int) -> None:
    """Launch the contention sampler as a detached process (best-effort)."""
    if not MONITOR_ENABLED:
        return
    script = Path(__file__).resolve().parent / MONITOR_SCRIPT
    if not script.exists():
        print(f"resource_monitor: {script.name} not found — running without monitoring.")
        return
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script), "--start",
             "--pin-core", str(pin_core),
             "--out", SAMPLES_LOG,
             "--pidfile", MONITOR_PIDFILE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,  # survive this process exiting
        )
        print(f"Resource monitor started (pid {proc.pid}); samples → {SAMPLES_LOG}")
    except Exception as exc:  # noqa: BLE001 — monitoring must never break the run
        print(f"resource_monitor: failed to start ({exc}); continuing without monitoring.")


def _stop_monitor() -> None:
    """Stop the contention sampler via its PID file (best-effort)."""
    if not Path(MONITOR_PIDFILE).exists():
        return
    script = Path(__file__).resolve().parent / MONITOR_SCRIPT
    try:
        subprocess.run([sys.executable, str(script), "--stop",
                        "--pidfile", MONITOR_PIDFILE], check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"resource_monitor: failed to stop ({exc}).")


def _wait_ready(port: int = 5432, timeout: float = 120.0) -> None:
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed.  Run: pip install psycopg2-binary")
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
            time.sleep(0.5)

    print(f"ERROR: Umbra not ready after {timeout:.0f}s. Last error: {last_err}")
    print("\nContainer logs:")
    logs = _run(["docker", "logs", "--tail", "30", CONTAINER])
    print(logs.stdout or logs.stderr or "(no logs)")
    print(f"\nHint: check if port {port} is already in use by another service:")
    print(f"  ss -tlnp | grep {port}")
    sys.exit(1)


def _memory_flags(memory_gb: float) -> list[str]:
    """Docker RAM-cap flags for the container, or [] when unlimited.

    Sets --memory and --memory-swap to the same value so the limit is a hard
    ceiling on RAM (no swap), making a runaway query fail *inside* Umbra
    (cgroup OOM, contained) instead of exhausting the host and letting the
    kernel OOM-killer take down the driver.  Expressed in MB so fractional GB
    values (e.g. 0.5) are accepted by docker.
    """
    if not memory_gb or memory_gb <= 0:
        return []
    mb = int(round(memory_gb * 1024))
    return ["--memory", f"{mb}m", "--memory-swap", f"{mb}m"]


def start(port: int = 5432, memory_gb: float = 0.0) -> None:
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Make the data directory world-writable so the container process
    # (running as a different UID) can create and lock its database files.
    data_dir.chmod(0o777)
    # Remove stale lock files from previous runs that weren't cleanly shut down.
    for lock in data_dir.glob("*.lock"):
        lock.unlink(missing_ok=True)

    # Remove any existing container (stopped or running) to start fresh
    _run(["docker", "rm", "-f", CONTAINER])

    # Pin the container to the measurement core and, when the NUMA node is known,
    # bind its memory to that same node so allocations stay node-local (avoids
    # remote-node memory latency, a silent source of runtime variance). No root
    # required — docker honors both via the daemon.
    node = _numa_node_of(PIN_CORE)
    cpuset_mems = ["--cpuset-mems", str(node)] if node is not None else []

    # Force Umbra to a single worker thread — the analogue of DuckDB's
    # `SET threads TO 1` and Postgres' max_parallel_workers=0, without which the
    # engines are compared unfairly. Umbra sizes its worker pool from the cpulist
    # of the NUMA node it is pinned to (/sys/.../node<N>/cpulist), spawning one
    # worker per CPU in that list and *ignoring* the cpuset affinity. Pinning to
    # one core therefore still yields ~one worker per node CPU (measured on c07:
    # 37 runnable workers oversubscribing the single pinned core), which is both
    # far noisier and several times slower than a single worker. We shadow that
    # sysfs file with one listing only the pinned core, so Umbra starts exactly
    # one worker. Read-only bind mount, no root, no image patch. Skipped when the
    # node is undeterminable (same condition as --cpuset-mems above).
    single_thread_mounts: list[str] = []
    if node is not None:
        cpulist_file = data_dir / ".single_cpulist"
        cpulist_file.write_text(f"{PIN_CORE}\n")
        cpulist_file.chmod(0o644)
        single_thread_mounts = [
            "-v", f"{cpulist_file}:/sys/devices/system/node/node{node}/cpulist:ro",
        ]

    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER,
        "--cpus", "1",
        "--cpuset-cpus", str(PIN_CORE),
        *cpuset_mems,
        *single_thread_mounts,
        "-v", f"{DATA_DIR}:/var/db",
        "-p", f"{port}:5432",
        "--ulimit", "nofile=1048576:1048576",
        "--ulimit", "memlock=8388608:8388608",
        *_memory_flags(memory_gb),
        IMAGE,
    ]
    result = _run(cmd)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "permission denied" in stderr.lower() or "connect: permission denied" in stderr.lower():
            print("─" * 60)
            print("ERROR: Docker permission denied on the remote server.")
            print("Fix:   sudo usermod -aG docker $USER")
            print("       Then log out and back in (or run: newgrp docker)")
            print("─" * 60)
        else:
            print(f"ERROR: docker run failed:\n{stderr}")
        sys.exit(1)

    mem_note = f", memory cap {memory_gb:g} GB" if memory_gb and memory_gb > 0 else ""
    mems_note = f", mems node {node}" if node is not None else ""
    st_note = " (single-threaded via node cpulist shadow)" if single_thread_mounts else \
              " (WARNING: NUMA node unknown — single-thread shadow NOT applied)"
    print(f"Umbra container '{CONTAINER}' started on core {PIN_CORE}{mems_note}, port {port}{mem_note}{st_note}.")
    print("Waiting for Umbra to become ready...")
    _wait_ready(port=port)
    print("Umbra is ready.")

    # Start contention monitoring only once the container is up and measuring.
    _start_monitor(PIN_CORE)


def teardown() -> None:
    _stop_monitor()
    _run(["docker", "stop", CONTAINER])
    _run(["docker", "rm", "-f", CONTAINER])
    print(f"Umbra container '{CONTAINER}' stopped and removed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or stop the Umbra Docker container.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Start the Umbra container.")
    group.add_argument("--teardown", action="store_true", help="Stop and remove the Umbra container.")
    group.add_argument("--stop-monitor", action="store_true",
                       help="Stop only the resource monitor (safety-net cleanup).")
    parser.add_argument("--port", type=int, default=5432,
                        help="Host port to expose Umbra on (default: 5432).")
    parser.add_argument("--memory-gb", type=float, default=0.0,
                        help="Hard RAM cap for the container in GB (docker --memory/--memory-swap); "
                             "0 = unlimited (default).")
    args = parser.parse_args()

    if args.start:
        start(port=args.port, memory_gb=args.memory_gb)
    elif args.stop_monitor:
        _stop_monitor()
    else:
        teardown()


if __name__ == "__main__":
    main()
