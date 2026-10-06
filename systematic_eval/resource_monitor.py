#!/usr/bin/env python3
"""Standalone, ultra-light server-contention sampler.

Self-contained (no project imports) — scp'd to the remote server alongside
umbra_setup.py.  It records, during a measurement run, how busy the *sibling*
hyperthread of the pinned Umbra core is and what frequency the pinned core runs
at, so each measured query can later be annotated with the server contention it
actually experienced (see stages/execution.py's contention merge and
stages/statistics.py's trust scoring).

Design goal: it must NOT perturb the thing it measures.  Guarantees:
  * runs as a separate process, pinned to a core on the *other* NUMA node (so it
    cannot steal cycles or cache/memory bandwidth from Umbra's pinned core),
  * renices itself to the lowest priority (no root needed to lower own nice),
  * the hot loop does only two tiny pseudo-file reads (one /proc/stat line and
    one sysfs freq file) — no subprocess spawns (fork/exec is the expensive thing
    to avoid), no `ps`/`mpstat`,
  * buffers samples and flushes ~1x/second,
  * on exit writes a footer with its own CPU usage (os.times) so the run can be
    shown to have cost a negligible fraction of a core-second.

Usage:
    python3 resource_monitor.py --start --pin-core 4 --out contention.log
    python3 resource_monitor.py --stop  --pidfile /path/to/.monitor.pid
"""

import argparse
import glob
import os
import signal
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1


# ── Topology helpers (sysfs only, no privilege) ─────────────────────────────

def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_cpulist(spec: str) -> list[int]:
    """Parse a Linux cpulist like "0-3,8,12-14" into [0,1,2,3,8,12,13,14]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _sibling_cpus(pin_core: int) -> list[int]:
    """The other logical CPUs sharing pin_core's physical core (excludes pin)."""
    spec = _read(f"/sys/devices/system/cpu/cpu{pin_core}/topology/thread_siblings_list")
    if not spec:
        return []
    return [c for c in _parse_cpulist(spec) if c != pin_core]


def _node_of(cpu: int) -> int | None:
    for p in glob.glob(f"/sys/devices/system/cpu/cpu{cpu}/node[0-9]*"):
        try:
            return int(os.path.basename(p)[len("node"):])
        except ValueError:
            continue
    return None


def _pick_monitor_core(pin_core: int, exclude: set[int]) -> int:
    """An online CPU to pin the sampler to: prefer the *other* NUMA node, avoid
    the pinned core and its sibling(s). Falls back to any allowed CPU."""
    allowed = sorted(os.sched_getaffinity(0) - exclude)
    if not allowed:
        return pin_core  # degenerate; nothing else available
    pin_node = _node_of(pin_core)
    if pin_node is not None:
        other = [c for c in allowed if _node_of(c) not in (pin_node, None)]
        if other:
            # middle of the other node's allowed list — away from cpu0/IRQ cores
            return other[len(other) // 2]
    return allowed[len(allowed) // 2]


# ── Sampling ────────────────────────────────────────────────────────────────

def _cpu_busy_snapshot(stat_fh, prefix: str) -> tuple[int, int] | None:
    """Return (busy_jiffies, total_jiffies) for a CPU from an already-open
    /proc/stat handle (re-read via seek(0), no reopen), or None.

    busy excludes idle+iowait.
    """
    try:
        stat_fh.seek(0)
        for line in stat_fh.read().splitlines():
            if line.startswith(prefix):
                parts = [int(x) for x in line.split()[1:]]
                # user nice system idle iowait irq softirq steal guest guest_nice
                idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                total = sum(parts)
                return total - idle, total
    except OSError:
        return None
    return None


def _reread(fh) -> str | None:
    """Fresh contents of an open pseudo-file via seek(0) (no reopen)."""
    if fh is None:
        return None
    try:
        fh.seek(0)
        return fh.read().strip()
    except OSError:
        return None


def _run(pin_core: int, out_path: Path, interval_ms: float,
         monitor_core: int | None, pidfile: Path | None) -> None:
    siblings = _sibling_cpus(pin_core)
    sibling = siblings[0] if siblings else None
    exclude = {pin_core, *siblings}
    mon = monitor_core if monitor_core is not None else _pick_monitor_core(pin_core, exclude)

    # Pin ourselves off the measurement core and drop to lowest priority.
    affinity_ok = True
    try:
        os.sched_setaffinity(0, {mon})
    except OSError:
        affinity_ok = False
    try:
        os.nice(19)
    except OSError:
        pass

    freq_path = f"/sys/devices/system/cpu/cpu{pin_core}/cpufreq/scaling_cur_freq"
    interval_s = interval_ms / 1000.0

    if pidfile is not None:
        pidfile.write_text(str(os.getpid()), encoding="utf-8")

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    start_wall = time.time()
    start_mono = time.monotonic()
    n_samples = 0
    buf: list[str] = []

    fh = out_path.open("w", buffering=1, encoding="utf-8")
    fh.write(
        f"# resource_monitor schema={SCHEMA_VERSION} pin_core={pin_core} "
        f"sibling={sibling} monitor_core={mon} pin_node={_node_of(pin_core)} "
        f"mon_node={_node_of(mon)} affinity_ok={affinity_ok} "
        f"interval_ms={interval_ms:g} start_wall={start_wall:.6f} "
        f"start_mono={start_mono:.6f}\n"
    )
    fh.write("# monotonic,wall,sibling_busy_frac,pin_khz,load1\n")

    # Open the pseudo-files once; re-read each tick with seek(0) to avoid the
    # syscall cost of reopening 50x/second.
    stat_fh = open("/proc/stat")
    try:
        freq_fh = open(freq_path)
    except OSError:
        freq_fh = None
    try:
        load_fh = open("/proc/loadavg")
    except OSError:
        load_fh = None

    sib_prefix = f"cpu{sibling} " if sibling is not None else None
    prev = _cpu_busy_snapshot(stat_fh, sib_prefix) if sib_prefix else None
    load_every = max(1, int(round(1.0 / interval_s)))  # refresh loadavg ~1 Hz
    load1 = ""
    tick = 0
    next_flush = start_mono + 1.0

    while not stop["flag"]:
        loop_start = time.monotonic()

        # sibling busy fraction since last tick (cheap: one /proc/stat re-read)
        busy_frac = ""
        if sib_prefix:
            cur = _cpu_busy_snapshot(stat_fh, sib_prefix)
            if cur is not None and prev is not None:
                dtotal = cur[1] - prev[1]
                if dtotal > 0:
                    busy_frac = f"{(cur[0] - prev[0]) / dtotal:.4f}"
                prev = cur
            elif cur is not None:
                prev = cur

        khz = _reread(freq_fh) or ""

        if tick % load_every == 0:
            la = _reread(load_fh)
            load1 = la.split(" ", 1)[0] if la else ""

        buf.append(f"{loop_start:.6f},{time.time():.6f},{busy_frac},{khz},{load1}\n")
        n_samples += 1
        tick += 1

        if loop_start >= next_flush:
            fh.write("".join(buf))
            buf.clear()
            next_flush = loop_start + 1.0

        # keep a steady cadence despite the (tiny) work above
        sleep_for = interval_s - (time.monotonic() - loop_start)
        if sleep_for > 0:
            time.sleep(sleep_for)

    if buf:
        fh.write("".join(buf))
    stat_fh.close()
    if freq_fh is not None:
        freq_fh.close()
    if load_fh is not None:
        load_fh.close()
    t = os.times()  # (utime, stime, cutime, cstime, elapsed)
    elapsed = time.monotonic() - start_mono
    cpu_used = t.user + t.system
    fh.write(
        f"# footer n_samples={n_samples} elapsed_s={elapsed:.3f} "
        f"self_cpu_s={cpu_used:.4f} self_cpu_frac={cpu_used / elapsed if elapsed else 0:.6f}\n"
    )
    fh.close()
    if pidfile is not None:
        try:
            pidfile.unlink()
        except OSError:
            pass


def _stop(pidfile: Path) -> None:
    pid = _read(str(pidfile))
    if not pid:
        print(f"resource_monitor: no pidfile at {pidfile}")
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
        print(f"resource_monitor: sent SIGTERM to pid {pid}")
    except (ProcessLookupError, ValueError) as exc:
        print(f"resource_monitor: could not stop pid {pid}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight server-contention sampler.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Run the sampling loop (foreground).")
    group.add_argument("--stop", action="store_true", help="Stop a running sampler via --pidfile.")
    parser.add_argument("--pin-core", type=int, default=4,
                        help="Logical CPU the measured engine is pinned to (default 4).")
    parser.add_argument("--out", type=str, default="contention_samples.log",
                        help="Output samples file (start only).")
    parser.add_argument("--interval-ms", type=float, default=20.0,
                        help="Sampling period in ms (default 20 = 50 Hz).")
    parser.add_argument("--monitor-core", type=int, default=None,
                        help="Pin the sampler to this CPU (default: auto, other NUMA node).")
    parser.add_argument("--pidfile", type=str, default=None,
                        help="PID file to write on start / read on stop.")
    args = parser.parse_args()

    pidfile = Path(args.pidfile) if args.pidfile else None
    if args.stop:
        if pidfile is None:
            print("resource_monitor: --stop requires --pidfile")
            sys.exit(1)
        _stop(pidfile)
        return
    _run(args.pin_core, Path(args.out), args.interval_ms, args.monitor_core, pidfile)


if __name__ == "__main__":
    main()
