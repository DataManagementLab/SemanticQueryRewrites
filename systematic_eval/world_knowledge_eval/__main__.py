"""CLI: judge how world-knowledge-dependent an oracle run's validated rules are.

    # extraction only (no API calls) — inspect the rule set + performance
    python3 -m systematic_eval.world_knowledge_eval \
        --experiment experiment_T_imdb_job_12_oracle_c07_4-2 --extract-only

    # smoke test: judge the first 2 rules
    python3 -m systematic_eval.world_knowledge_eval \
        --experiment experiment_T_imdb_job_12_oracle_c07_4-2 --limit 2

    # full run
    python3 -m systematic_eval.world_knowledge_eval \
        --experiment experiment_T_imdb_job_12_oracle_c07_4-2 --budget 5.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from systematic_eval.config_loader import load_config
from systematic_eval.prompt_loader import PromptLoader
from systematic_eval.world_knowledge_eval.extract import extract_rule_records
from systematic_eval.world_knowledge_eval.judge import judge_records

ROOT = Path(__file__).resolve().parents[2]
SE = ROOT / "systematic_eval"


def _fmt(x: float | None, nd: int = 2) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "rule_name", "rule_id", "score", "label",
        "n_queries_fired", "n_queries_beneficial",
        "median_percent_saved", "max_percent_saved",
        "justification", "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="World-knowledge evaluation of validated rewrite rules.")
    parser.add_argument("--experiment", required=True,
                        help="Experiment/config stem, e.g. experiment_T_imdb_job_12_oracle_c07_4-2 "
                             "(names the transfer_data/ and saved_results/ folders).")
    parser.add_argument("--config", default=None,
                        help="Config YAML (default: config/<experiment>.yaml).")
    parser.add_argument("--transfer-dir", default=None,
                        help="Override transfer_data/<experiment> directory.")
    parser.add_argument("--model", default=None,
                        help="OpenAI judge model (default: the config's generation model).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge only the first N rules (smoke test).")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="Cost budget passed to the LLM wrapper; execution above it prompts. "
                             "Use a negative value to bypass the prompt (None-equivalent).")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the LLM disk cache.")
    parser.add_argument("--extract-only", action="store_true",
                        help="Build and save rule records without any API calls.")
    parser.add_argument("--reuse-extract", action="store_true",
                        help="Load rules_extracted.json instead of re-streaming the "
                             "multi-GB rule_summary_result.json (skips the ~minutes-long extraction).")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else SE / "config" / f"{args.experiment}.yaml"
    config = load_config(config_path)

    transfer_dir = Path(args.transfer_dir) if args.transfer_dir else SE / "transfer_data" / args.experiment
    out_dir = SE / "saved_results" / args.experiment / "world_knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract validated rules + rationale + individual oracle performance.
    extracted_path = out_dir / "rules_extracted.json"
    if args.reuse_extract:
        if not extracted_path.exists():
            parser.error(f"--reuse-extract set but {extracted_path} does not exist; run once without it.")
        print(f"Reusing {extracted_path} ...")
        record_dicts = json.loads(extracted_path.read_text(encoding="utf-8"))["rules"]
        print(f"  {len(record_dicts)} unique validated rules; "
              f"{sum(1 for r in record_dicts if r.get('rationales'))} have a recorded rationale.")
    else:
        print(f"Extracting rule records from {transfer_dir} ...")
        records = extract_rule_records(transfer_dir)
        records.sort(key=lambda r: r.rule_name)
        print(f"  {len(records)} unique validated rules; "
              f"{sum(1 for r in records if r.rationales)} have a recorded rationale.")
        record_dicts = [r.to_dict() for r in records]
        extracted_path.write_text(
            json.dumps({"experiment": args.experiment, "rules": record_dicts}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Wrote {extracted_path}")

    if args.extract_only:
        print("--extract-only set; skipping judge.")
        return

    # 2. Judge (OpenAI, cached).
    to_judge = record_dicts[: args.limit] if args.limit else record_dicts
    dataset = config.dataset.name
    schema = PromptLoader(dataset).schema
    model = args.model or config.model
    budget = None if args.budget < 0 else args.budget
    print(f"Judging {len(to_judge)} rules on dataset '{dataset}' with model {model} (budget={budget}) ...")

    verdicts = judge_records(
        to_judge, schema=schema, dataset=dataset, model=model,
        budget=budget, use_cache=not args.no_cache,
    )

    # 3. Merge verdicts into records and write outputs.
    flat_rows: list[dict] = []
    for rec, verdict in zip(to_judge, verdicts):
        rec["verdict"] = verdict
        agg = rec.get("agg", {})
        flat_rows.append({
            "rule_name": rec["rule_name"],
            "rule_id": rec["rule_id"],
            "score": verdict.get("score"),
            "label": verdict.get("label"),
            "n_queries_fired": agg.get("n_queries_fired"),
            "n_queries_beneficial": agg.get("n_queries_beneficial"),
            "median_percent_saved": _fmt(agg.get("median_percent_saved")),
            "max_percent_saved": _fmt(agg.get("max_percent_saved")),
            "justification": verdict.get("justification", ""),
            "rationale": " ".join(rec.get("rationales", [])),
        })

    (out_dir / "wk_eval.json").write_text(
        json.dumps({"experiment": args.experiment, "model": model, "rules": to_judge}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "wk_eval.csv", flat_rows)
    print(f"  Wrote {out_dir / 'wk_eval.json'} and {out_dir / 'wk_eval.csv'}")

    # Local static figures (PNG + PDF), alongside the JSON/CSV.
    try:
        from systematic_eval.world_knowledge_eval.plot import render_plots
        figs = render_plots(to_judge, out_dir, experiment=args.experiment, model=model)
        print("  Wrote plots: " + ", ".join(p.name for p in figs))
    except Exception as exc:  # plotting must never sink an otherwise-good run
        print(f"  WARNING: plot generation failed ({exc}); JSON/CSV are intact.")

    # 4. Console summary.
    scored = [r["score"] for r in flat_rows if isinstance(r["score"], int)]
    n_parse_fail = sum(1 for v in verdicts if v.get("parse_error"))
    print("\nScore distribution (1=DB-derivable ... 5=world-knowledge):")
    for s in range(1, 6):
        n = scored.count(s)
        bar = "#" * n
        print(f"  {s}: {n:3d} {bar}")
    if n_parse_fail:
        print(f"  ({n_parse_fail} verdicts failed to parse — see parse_error in wk_eval.json)")


if __name__ == "__main__":
    main()
