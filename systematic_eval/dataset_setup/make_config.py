"""Emit experiment configs for a dataset, cloned from the IMDb/basketball oracle configs.

Writes two configs into ``systematic_eval/config/``:
    experiment_<short>_oracle.yaml          - sql_dir = sql/<short>_200k
    experiment_<short>_complex_oracle.yaml  - sql_dir = sql/<short>_complex

Both mirror ``experiment_T_imdb_job_12_oracle_c07_4-2.yaml`` (DuckDB oracle pipeline) with the
basketball DuckDB safety knobs (max_temp_size, query_timeout_s). Loadable by
``config_loader.load_config``.

Usage:
    python3 -m systematic_eval.dataset_setup.make_config --short financial
"""

from __future__ import annotations

import argparse
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"

TEMPLATE = """model: "gpt-5.4-2026-03-05"
budget: 0.01

dataset:
  name: "{short}"
  sql_dir: "{sql_dir}"
  excluded_files:
    - "fkindexes.sql"
    - "schema.sql"
  db_file: "{short}.duckdb"
  strip_min: false
  query_limit: 100

prompts:
  system: "prompt01"
  generation: "prompt_wk01"

generation:
  rounds: 4

refinement:
  enabled: true
  iterations: 1
  samples: 2
  prompt: "refine_prompt02"

execution:
  performance_threshold: 0.001
  attempts: 3
  validation_mode: "base_tables"
  cost_estimation: false
  warmup_mode: per_query
  # Bound DuckDB spill so a runaway base-table validation query fails catchably
  # instead of OOM/disk-killing the process and leaving no result.json.
  max_temp_size: "50GB"
  # Per-query wall-clock timeout: any single run exceeding 4 min is interrupted
  # and its rule skipped/discarded.
  query_timeout_s: 240

remote:
  server: "c06"
  path: "/mnt/labstore/psiegler/c06_multi_query_comparison/"

aggregation:
  prefix_len: 3
  time_filtering: false
  cross_query_transfer: true

statistics:
  show_stats: variance
  runtime_aggregator: mean
  mode: both
"""


def write_configs(short: str) -> list[Path]:
    written: list[Path] = []
    # The file name is the experiment identity; the config carries no name of its own.
    variants = {
        f"experiment_{short}_oracle.yaml": f"sql/{short}_200k",
        f"experiment_{short}_complex_oracle.yaml": f"sql/{short}_complex",
    }
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, sql_dir in variants.items():
        path = CONFIG_ROOT / filename
        path.write_text(
            TEMPLATE.format(short=short, sql_dir=sql_dir),
            encoding="utf-8",
        )
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short", required=True, help="project-local dataset short name (e.g. financial)")
    args = ap.parse_args()
    for path in write_configs(args.short):
        print(f"wrote config -> {path}")


if __name__ == "__main__":
    main()
