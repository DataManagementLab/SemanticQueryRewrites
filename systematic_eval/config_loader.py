"""Experiment configuration loading from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str
    sql_dir: str
    excluded_files: list[str] = field(default_factory=lambda: ["fkindexes.sql", "schema.sql"])
    db_file: str = "imdb.duckdb"
    strip_min: bool = True
    query_limit: int | None = None


@dataclass
class PromptConfig:
    system: str = "prompt01"
    generation: str = "prompt10"


@dataclass
class GenerationConfig:
    # Independent LLM samples per query (breadth); deduped on requires+implies.
    rounds: int = 1


@dataclass
class RefinementConfig:
    enabled: bool = True
    iterations: int = 1  # sequential refine→execute cycles (depth)
    samples: int = 1  # independent refinement samples per failed rule (breadth)
    prompt: str = "refine_prompt02"


@dataclass
class ExecutionConfig:
    performance_threshold: float = 0.05  # inert: the flag it computes is recorded, never acted on
    attempts: int = 3
    # "base_tables" = soundness gate (required for cross-query transfer);
    # "query" = per-query output-equality check; "both" = run both.
    validation_mode: str = "query"
    warmup_mode: str = "per_query"  # "per_query", "startup_only", "startup_and_per_query", or "none"
    prime_os_cache: bool = True  # duckdb only: `cat <db_file> > /dev/null` once before measurements
    max_temp_size: str = "0"  # duckdb only: cap on-disk spill (e.g. "50GB") so runaway queries fail catchably instead of OOM-killing; "0" = unlimited
    query_timeout_s: float = 0.0  # per-query wall-clock timeout (seconds); a run exceeding this is interrupted and its rule discarded; 0 = no timeout
    validation_workers: int = 0  # duckdb only: parallel workers for the validation runs (initial + refinement); 0 = auto (half the node's cores), 1 = serial. Final perf-measurement run always runs serial/exclusive.
    measure_workload_baseline: bool = True  # measure originals of no-rule queries (median of `attempts`) so the runtime-weighted whole-workload improvement can be drawn on the *_speedup_bar_all plots
    cost_estimation: bool = False  # EXPLAIN-based cost filtering for cross-query transfer
    cost_save_threshold: float = 0.0  # required relative cost reduction (fraction) for a rule subset to win; 0.0 = any reduction
    fix_join_order: bool = False  # pin the rewrite to the original's join order (confounder control)
    engine: str = "duckdb"  # "duckdb", "umbra", or "postgres"
    umbra_port: int = 5432
    umbra_memory_gb: float = 0.0  # umbra only: hard cap on the container's RAM (docker --memory/--memory-swap) so a runaway join fails inside Umbra instead of OOM-killing the host; 0 = unlimited. Analogous to duckdb's max_temp_size.
    driver_memory_gb: float = 0.0  # umbra/postgres only: backstop RAM cap on the run_sql.py *driver* process (systemd-run --scope, host side). The container cap does not bound client-side fetches, so a runaway result set materialized in the driver could still exhaust the host; this OOM-kills the driver instead. 0 = no cap. No effect for duckdb (the engine runs inside the driver).
    postgres_port: int = 5432
    postgres_host: str = "127.0.0.1"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_dbname: str = "imdb"
    postgres_random_page_cost: float | None = None  # postgres only: ALTER SYSTEM SET random_page_cost. None = leave unset, i.e. postgres' own default (4.0)
    # Rule-subset selection signal during cost-filtered aggregation.
    # "planner": engine EXPLAIN cost (current). "zeroshot": learned-model predicted runtime
    # (requires engine=postgres; uses the external scorer subproject).
    cost_model: str = "planner"
    zeroshot_model_type: str | None = None      # ldb_models model_type string
    zeroshot_model_dir: str | None = None       # checkpoint dir, e.g. .../models/<name>/imdb
    zeroshot_seed: int = 9
    zeroshot_statistics_file: str | None = None  # feature_statistics json
    zeroshot_database_stats: str | None = None   # {"database_stats": {...}} json for the instance
    max_subsets: int = 4096                       # cap on unique candidate subsets scored per query

    def __post_init__(self) -> None:
        allowed_cm = {"planner", "zeroshot"}
        if self.cost_model not in allowed_cm:
            raise ValueError(
                f"execution.cost_model must be one of {sorted(allowed_cm)}, got {self.cost_model!r}"
            )
        if self.cost_model == "zeroshot" and self.engine != "postgres":
            raise ValueError(
                "execution.cost_model='zeroshot' requires execution.engine='postgres' "
                f"(got engine={self.engine!r})"
            )
        if self.postgres_random_page_cost is not None:
            if self.engine != "postgres":
                raise ValueError(
                    "execution.postgres_random_page_cost requires execution.engine='postgres' "
                    f"(got engine={self.engine!r})"
                )
            if self.postgres_random_page_cost <= 0:
                raise ValueError(
                    "execution.postgres_random_page_cost must be > 0, got "
                    f"{self.postgres_random_page_cost!r}"
                )


@dataclass
class RemoteConfig:
    server: str = "c06"
    path: str = "/mnt/labstore/psiegler/multi_query_comparison/"


@dataclass
class AggregationConfig:
    prefix_len: int = 8
    cross_query_transfer: bool = True  # apply each validated rule to every query, not only its source
    time_filtering: bool = True  # require a query-level time improvement (performance filter; oracle runs disable it)
    allowed_rule_types: list[str] | None = None  # None = all; "filter" is the only type the thesis evaluates


@dataclass
class StatisticsConfig:
    # NOTE on the two knobs below: both apply ONLY to the legacy all-rules optimizer
    # view (statistics.build_rows), the one view that reads the per-attempt timing
    # lists. The oracle view and the cost-winner optimizer view read
    # per_subset_results, which stores the execution stage's median and no per-attempt
    # list — there, the runtime is always that median and no spread columns exist,
    # whatever these two are set to. Any run with mode=oracle/both or cost_estimation
    # (i.e. every run reported in the thesis) is in that second case.
    #
    # "none" | "range" | "variance" — extra spread columns + bar-plot error bars when N>1.
    # range: min/max columns + asymmetric error bars; variance: sample-stddev (n-1) columns + symmetric ±σ bars.
    show_stats: str = "none"
    # "median" | "mean" — how to collapse the N per-query runs into a single runtime
    # value for percent-saved. This does NOT control how the N `execution.attempts`
    # are collapsed in the first place: the execution stage always reports a median
    # (per-pipeline median summed on Umbra) and that value is what every other view
    # uses.
    runtime_aggregator: str = "median"
    # Which statistics view(s) to emit:
    #   "optimizer" — the subset the planner/cost model picked (deployable, can regress)
    #   "oracle"    — fastest output-matching subset per query (ceiling, not deployable)
    #   "both"      — side by side from the same measured data
    mode: str = "optimizer"

    def __post_init__(self) -> None:
        allowed = {"none", "range", "variance"}
        if self.show_stats not in allowed:
            raise ValueError(
                f"statistics.show_stats must be one of {sorted(allowed)}, got {self.show_stats!r}"
            )
        allowed_agg = {"median", "mean"}
        if self.runtime_aggregator not in allowed_agg:
            raise ValueError(
                f"statistics.runtime_aggregator must be one of {sorted(allowed_agg)}, "
                f"got {self.runtime_aggregator!r}"
            )
        allowed_mode = {"optimizer", "oracle", "both"}
        if self.mode not in allowed_mode:
            raise ValueError(
                f"statistics.mode must be one of {sorted(allowed_mode)}, got {self.mode!r}"
            )


@dataclass
class ExperimentConfig:
    # An experiment is identified by its config file's name, not by a field in the file
    # (controller.sh, run_pipeline.py and world_knowledge_eval all use that stem).
    model: str
    budget: float
    dataset: DatasetConfig
    prompts: PromptConfig
    use_llm_cache: bool = True
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)


def load_config(config_path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        model=raw["model"],
        budget=raw.get("budget", 0.01),
        dataset=DatasetConfig(**raw["dataset"]),
        prompts=PromptConfig(**raw.get("prompts", {})),
        use_llm_cache=raw.get("use_llm_cache", True),
        generation=GenerationConfig(**raw.get("generation", {})),
        refinement=RefinementConfig(**raw.get("refinement", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        remote=RemoteConfig(**raw.get("remote", {})),
        aggregation=AggregationConfig(**raw.get("aggregation", {})),
        statistics=StatisticsConfig(**raw.get("statistics", {})),
    )
