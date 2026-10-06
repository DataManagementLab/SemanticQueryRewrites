"""Loads prompt text files organized by dataset.

Prompt directory structure:
    prompts/<dataset_name>/
        schema.txt                  # Shared schema block
        system/<name>.txt           # System prompts
        generation/<name>.txt       # Generation prompts (may use {schema})
        refinement/<name>.txt       # Refinement prompts
        examples/<name>.json        # Rule format examples
"""

from __future__ import annotations

from pathlib import Path


PROMPTS_ROOT = Path(__file__).parent / "prompts"


class PromptLoader:
    """Loads prompts from text files for a given dataset."""

    def __init__(self, dataset_name: str) -> None:
        self.base = PROMPTS_ROOT / dataset_name
        if not self.base.exists():
            raise FileNotFoundError(f"Prompt directory not found: {self.base}")

        schema_path = self.base / "schema.txt"
        self.schema = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""

    def load_system_prompt(self, name: str) -> str:
        path = self.base / "system" / f"{name}.txt"
        return path.read_text(encoding="utf-8")

    def load_generation_prompt(self, name: str) -> str:
        """Load a generation prompt, substituting {schema} with the dataset schema."""
        path = self.base / "generation" / f"{name}.txt"
        text = path.read_text(encoding="utf-8")
        return text.replace("{schema}", self.schema)

    def load_refinement_prompt(self, name: str) -> str:
        path = self.base / "refinement" / f"{name}.txt"
        return path.read_text(encoding="utf-8")

    def load_rule_example(self, name: str = "rule_example") -> str:
        path = self.base / "examples" / f"{name}.json"
        return path.read_text(encoding="utf-8")
