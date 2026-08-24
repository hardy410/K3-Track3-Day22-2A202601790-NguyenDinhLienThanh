import json
from pathlib import Path

import pytest

from preference_lab.data import PreferenceDataError, load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=7)
    assert len(train) + len(val) == len(examples)
    assert {example.prompt for example in train}.isdisjoint(example.prompt for example in val)


def test_split_is_reproducible_and_groups_duplicate_prompts() -> None:
    examples = [
        PreferenceExample(prompt="same", chosen="good 1", rejected="bad 1"),
        PreferenceExample(prompt=" SAME ", chosen="good 2", rejected="bad 2"),
        PreferenceExample(prompt="other", chosen="good 3", rejected="bad 3"),
        PreferenceExample(prompt="third", chosen="good 4", rejected="bad 4"),
    ]
    first = split_by_prompt(examples, validation_ratio=0.34, seed=11)
    second = split_by_prompt(examples, validation_ratio=0.34, seed=11)
    assert first == second

    locations = ["train" if example in first[0] else "validation" for example in examples[:2]]
    assert locations[0] == locations[1]


def test_load_jsonl_reports_line_number(tmp_path: Path) -> None:
    data = tmp_path / "invalid.jsonl"
    data.write_text('{"prompt": "ok", "chosen": "a", "rejected": "b"}\n{bad json}\n')
    with pytest.raises(PreferenceDataError, match=r"invalid\.jsonl:2: invalid JSON"):
        load_jsonl(data)


def test_load_jsonl_rejects_duplicate_prompt(tmp_path: Path) -> None:
    rows = [
        {"prompt": "Question", "chosen": "answer a", "rejected": "answer b"},
        {"prompt": " question ", "chosen": "answer c", "rejected": "answer d"},
    ]
    data = tmp_path / "duplicates.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(PreferenceDataError, match="duplicate prompt"):
        load_jsonl(data)


def test_pii_guard_is_optional(tmp_path: Path) -> None:
    row = {
        "prompt": "Email me at learner@example.com",
        "chosen": "I cannot send email.",
        "rejected": "Done.",
    }
    data = tmp_path / "pii.jsonl"
    data.write_text(json.dumps(row), encoding="utf-8")
    assert len(load_jsonl(data)) == 1
    with pytest.raises(PreferenceDataError, match="possible PII"):
        load_jsonl(data, pii_guard=True)
