"""World-knowledge evaluation of generated rewrite rules.

Post-hoc analysis (not a pipeline stage): for each *validated* rule from an
oracle run, ask an OpenAI judge how strongly the rule depends on real-world
domain knowledge versus being derivable from the database instance alone
(e.g. a functional/statistical dependency a system could discover by scanning
the data). Each rule is annotated with its individual (single-rule) oracle
performance.

Data sources (all under ``systematic_eval/transfer_data/<experiment>/``):
  - ``rule_summary_result.json`` — validated, merged rules per query plus
    ``per_subset_results`` (the oracle's all-combinations run). The singleton
    subsets give each rule's individual speedup.
  - ``result.json`` / ``result2.json`` — generation/refinement records whose
    ``original_output`` holds the raw LLM JSON, including ``short_rationale``
    (the model's stated reasoning for each rule).
"""
