from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, field_validator


def normalize_for_comparison(value: str) -> str:
    """Normalize text before equality and near-duplicate checks."""
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            return rejected

        normalized_chosen = normalize_for_comparison(chosen)
        normalized_rejected = normalize_for_comparison(rejected)
        if normalized_chosen == normalized_rejected:
            raise ValueError("chosen and rejected must differ after normalization")

        # Only apply fuzzy matching to substantive responses. On very short text,
        # a one-character difference can represent a genuinely different answer.
        if min(len(normalized_chosen), len(normalized_rejected)) >= 20:
            similarity = SequenceMatcher(
                None, normalized_chosen, normalized_rejected, autojunk=False
            ).ratio()
            if similarity >= 0.98:
                raise ValueError("chosen and rejected are near duplicates")
        return rejected
