import json

from scripts.generate_vi_pilot import (
    actual_length_relation,
    local_quality_report,
    parse_examples,
)

from preference_lab.schemas import PreferenceExample


def make_example(
    *, prompt: str = "Một câu hỏi tiếng Việt?", relation: str = "chosen_shorter"
) -> PreferenceExample:
    return PreferenceExample(
        prompt=prompt,
        chosen="Câu trả lời đúng và ngắn gọn.",
        rejected=("Câu trả lời nghe có vẻ hợp lý nhưng chứa một lỗi tinh tế và dài hơn đáng kể."),
        metadata={
            "domain": "test",
            "rubric": "accuracy",
            "language": "vi",
            "source": "synthetic",
            "difficulty": "hard",
            "length_relation": relation,
        },
    )


def test_parse_examples_accepts_fenced_json_array() -> None:
    payload = json.dumps([make_example().model_dump()], ensure_ascii=False)
    examples = parse_examples(f"```json\n{payload}\n```")
    assert len(examples) == 1
    assert examples[0].metadata["language"] == "vi"


def test_actual_length_relation() -> None:
    assert actual_length_relation(make_example()) == "chosen_shorter"


def test_local_report_detects_clean_pilot() -> None:
    pilot = [make_example(prompt=f"Câu hỏi số {index}?") for index in range(3)]
    report = local_quality_report(pilot, [])
    assert report["sample_count"] == 3
    assert report["unique_prompt_count"] == 3
    assert report["local_checks_passed"] is True
