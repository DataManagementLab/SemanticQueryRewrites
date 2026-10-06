"""Shared JSON/string parsing utilities for LLM output processing."""

from __future__ import annotations

import json


def cut_string(s: str) -> str:
    """Extract the outermost JSON object from a string.

    LLM responses may wrap JSON in markdown code blocks or other text.
    This finds the first '{' and last '}' and returns the substring.
    """
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and start < end:
        return s[start : end + 1]
    return s


def replace_backslash(s: str) -> str:
    """Fix escaped quotes that some LLMs produce in JSON output."""
    return s.replace('\\"', '"')


def parse_llm_json(raw: str) -> dict | None:
    """Best-effort parse of LLM JSON output.

    Applies cut_string and replace_backslash before parsing.
    Returns None on failure.
    """
    try:
        return json.loads(replace_backslash(cut_string(raw)))
    except json.JSONDecodeError:
        return None
