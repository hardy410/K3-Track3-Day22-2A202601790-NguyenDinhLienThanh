from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


class PreferenceDataError(ValueError):
    """Raised when a preference dataset cannot be validated safely."""


def load_jsonl(
    path: str | Path,
    *,
    reject_duplicate_prompts: bool = True,
    pii_guard: bool = False,
) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Errors include the source line number. Duplicate prompts are rejected by
    default because they can leak across a naive train/validation split.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreferenceDataError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg} (column {exc.colno})"
                ) from exc

            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                details = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors()
                )
                raise PreferenceDataError(
                    f"{source}:{line_number}: invalid preference example: {details}"
                ) from exc

            normalized_prompt = " ".join(example.prompt.casefold().split())
            previous_line = seen_prompts.get(normalized_prompt)
            if reject_duplicate_prompts and previous_line is not None:
                raise PreferenceDataError(
                    f"{source}:{line_number}: duplicate prompt; first seen on line {previous_line}"
                )
            seen_prompts.setdefault(normalized_prompt, line_number)

            if pii_guard:
                _raise_if_pii(example, source=source, line_number=line_number)
            examples.append(example)
    return examples


def _raise_if_pii(example: PreferenceExample, *, source: Path, line_number: int) -> None:
    """Apply conservative checks for common email and phone-number patterns."""
    import re

    combined = f"{example.prompt}\n{example.chosen}\n{example.rejected}"
    patterns = {
        "email address": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone number": r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, combined, flags=re.IGNORECASE):
            raise PreferenceDataError(f"{source}:{line_number}: possible PII ({label})")


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    *,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Rows sharing the same normalized prompt always remain in the same split.
    """
    import random

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        key = " ".join(example.prompt.casefold().split())
        groups.setdefault(key, []).append(example)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    if len(keys) == 1:
        return list(groups[keys[0]]), []

    validation_group_count = round(len(keys) * validation_ratio)
    validation_group_count = min(max(1, validation_group_count), len(keys) - 1)
    validation_keys = set(keys[:validation_group_count])

    train = [item for key in keys if key not in validation_keys for item in groups[key]]
    validation = [item for key in keys if key in validation_keys for item in groups[key]]
    return train, validation
