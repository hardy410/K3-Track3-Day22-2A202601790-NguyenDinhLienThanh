from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
    *,
    tie_score: float = 0.0,
) -> float:
    """Return fraction where chosen score is greater than rejected score."""
    if not 0.0 <= tie_score <= 1.0:
        raise ValueError("tie_score must be between 0 and 1")
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("examples, chosen_scores, and rejected_scores must have equal lengths")
    if any(not math.isfinite(score) for score in (*chosen_scores, *rejected_scores)):
        raise ValueError("scores must contain only finite values")
    if not examples:
        return 0.0

    credit = sum(
        1.0 if chosen > rejected else tie_score if chosen == rejected else 0.0
        for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True)
    )
    return credit / len(examples)


def deterministic_text_score(text: str) -> float:
    """Return a reproducible CPU-only proxy score for smoke testing.

    This is deliberately not a semantic judge. It rewards informative length,
    lexical variety, and sentence completion so that the evaluation pipeline can
    be tested before model-derived log-probabilities are available.
    """
    tokens = re.findall(r"\b\w+\b", text.casefold(), flags=re.UNICODE)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    informative_length = math.log1p(min(len(tokens), 128))
    sentence_completion = 0.2 if text.rstrip().endswith((".", "!", "?")) else 0.0
    return informative_length + unique_ratio + sentence_completion


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
