"""Single entry point for the local (non-remote) parts of the evaluation pipeline.

Usage:
    python3 -m systematic_eval.run_pipeline --config config/experiment_T_imdb_job_12_oracle_c07_4-2.yaml --stage generate
    python3 -m systematic_eval.run_pipeline --config config/experiment_T_imdb_job_12_oracle_c07_4-2.yaml --stage refine --iteration 0
    python3 -m systematic_eval.run_pipeline --config config/experiment_T_imdb_job_12_oracle_c07_4-2.yaml --stage aggregate
    python3 -m systematic_eval.run_pipeline --config config/experiment_T_imdb_job_12_oracle_c07_4-2.yaml --stage stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

from systematic_eval.config_loader import load_config
from systematic_eval.prompt_loader import PromptLoader

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline stages.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["generate", "refine", "aggregate", "stats", "baseline-prep"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Refinement iteration number (for refine stage).",
    )
    parser.add_argument(
        "--transfer-dir",
        default=None,
        help="Per-experiment transfer data directory "
             "(default: systematic_eval/transfer_data/<config file stem>).",
    )
    parser.add_argument(
        "--no-plan-comparisons",
        action="store_true",
        help="Stats stage: skip the costly per-query plan comparison generation "
             "(still emits runtime stats, speedup plots, and CSV).",
    )
    parser.add_argument(
        "--full-plan-comparisons",
        action="store_true",
        help="Stats stage: render a plan comparison for every rule subset, including "
             "queries with more than 50 subsets (default: only the oracle and "
             "optimizer picks for those queries).",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    # An experiment is identified by its config file's stem — the same convention
    # controller.sh and world_knowledge_eval use.
    experiment = Path(args.config).stem
    transfer_dir = (
        Path(args.transfer_dir) if args.transfer_dir
        else ROOT / "systematic_eval" / "transfer_data" / experiment
    )

    if args.stage == "generate":
        prompts = PromptLoader(config.dataset.name)
        from systematic_eval.stages.generation import run_generation

        run_generation(config, prompts, transfer_dir)

    elif args.stage == "refine":
        prompts = PromptLoader(config.dataset.name)
        from systematic_eval.stages.refinement import run_refinement_iteration

        if args.iteration == 0:
            result_path = transfer_dir / "result.json"
            output_path = transfer_dir / "transfer2.json"
        else:
            result_path = transfer_dir / f"result{args.iteration + 1}.json"
            output_path = transfer_dir / f"transfer{args.iteration + 2}.json"

        run_refinement_iteration(result_path, output_path, config, prompts, args.iteration)

    elif args.stage == "aggregate":
        from systematic_eval.stages.aggregation import run_aggregation

        run_aggregation(config, transfer_dir)

    elif args.stage == "baseline-prep":
        from systematic_eval.stages.statistics import build_baseline_input

        build_baseline_input(
            transfer_dir,
            Path(config.dataset.sql_dir),
            excluded_files=config.dataset.excluded_files,
            query_limit=config.dataset.query_limit,
        )

    elif args.stage == "stats":
        from systematic_eval.stages.statistics import run_statistics

        # transfer_data/<experiment> → saved_results/<experiment>, for both the
        # controller-driven and the direct invocation.
        results_dir = transfer_dir.parent.parent / "saved_results" / transfer_dir.name
        run_statistics(
            transfer_dir,
            results_dir=results_dir,
            show_stats=config.statistics.show_stats,
            runtime_aggregator=config.statistics.runtime_aggregator,
            mode=config.statistics.mode,
            sql_dir=Path(config.dataset.sql_dir),
            excluded_files=config.dataset.excluded_files,
            query_limit=config.dataset.query_limit,
            skip_plan_comparisons=args.no_plan_comparisons,
            full_plan_comparisons=args.full_plan_comparisons,
        )


if __name__ == "__main__":
    main()
