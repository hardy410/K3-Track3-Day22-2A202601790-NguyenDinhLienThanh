from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from preference_lab.data import load_jsonl
from preference_lab.schemas import PreferenceExample

SYSTEM_PROMPT = """Bạn là chuyên gia xây dựng dữ liệu preference alignment tiếng Việt.
Hãy tạo dữ liệu để huấn luyện DPO, không phải dữ liệu hỏi đáp đơn giản.

Yêu cầu bắt buộc cho từng mẫu:
- prompt, chosen và rejected phải là tiếng Việt tự nhiên.
- chosen chính xác, hữu ích, an toàn và tuân thủ đúng yêu cầu.
- rejected phải hợp lý ở cái nhìn đầu tiên nhưng có lỗi tinh tế; không được sai ngớ ngẩn.
- Không dùng thông tin cá nhân thật, URL, API key hoặc bí mật.
- Không lặp lại prompt hoặc ý tưởng của ví dụ few-shot.
- metadata phải có domain, rubric, language="vi", source="synthetic",
  difficulty="hard" và length_relation.
- length_relation phải mô tả quan hệ độ dài thực tế giữa chosen và rejected:
  chosen_shorter, comparable hoặc chosen_longer.

Trả về DUY NHẤT một JSON array hợp lệ. Không markdown, không giải thích."""


FEW_SHOTS: list[dict[str, Any]] = [
    {
        "prompt": "Tóm tắt nguyên nhân chính gây ra hiện tượng thủy triều trong đúng 25 từ.",
        "chosen": "Lực hấp dẫn của Mặt Trăng, cùng ảnh hưởng nhỏ hơn từ Mặt Trời, làm nước biển dao động tuần hoàn, tạo nên thủy triều.",
        "rejected": "Thủy triều chủ yếu do gió mạnh đẩy nước biển về phía bờ, kết hợp với sự quay của Trái Đất và lực hấp dẫn từ Mặt Trăng.",
        "metadata": {
            "domain": "instruction_following",
            "rubric": "accuracy_and_word_limit",
            "language": "vi",
            "source": "synthetic",
            "difficulty": "hard",
            "length_relation": "chosen_shorter",
        },
    },
    {
        "prompt": "Ứng dụng Python của tôi báo lỗi kết nối cơ sở dữ liệu. Hãy chỉ cách sửa ngay.",
        "chosen": "Chưa đủ thông tin để xác định nguyên nhân. Hãy cung cấp loại cơ sở dữ liệu, thông báo lỗi đầy đủ, chuỗi kết nối đã che mật khẩu, phiên bản thư viện và bước tái hiện; trước mắt kiểm tra dịch vụ đang chạy, địa chỉ máy chủ, cổng và quyền truy cập.",
        "rejected": "Lỗi này thường do cổng 5432 bị đóng. Hãy mở cổng 5432 trên tường lửa và đổi mật khẩu tài khoản postgres thành postgres, sau đó khởi động lại máy.",
        "metadata": {
            "domain": "troubleshooting",
            "rubric": "uncertainty_and_safe_diagnostics",
            "language": "vi",
            "source": "synthetic",
            "difficulty": "hard",
            "length_relation": "chosen_longer",
        },
    },
]


PILOT_BLUEPRINT = [
    ("factual_reasoning", "accuracy", "chosen_shorter"),
    ("medical_safety", "safe_helpfulness", "comparable"),
    ("financial_safety", "calibrated_advice", "chosen_longer"),
    ("instruction_following", "strict_constraint", "chosen_shorter"),
    ("uncertainty", "calibration", "comparable"),
    ("troubleshooting", "missing_context", "chosen_longer"),
    ("structured_output", "format_compliance", "chosen_shorter"),
    ("vietnamese_context", "cultural_accuracy", "comparable"),
    ("coding", "subtle_bug", "chosen_longer"),
    ("hallucination", "source_uncertainty", "chosen_longer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 10-sample Vietnamese DPO pilot")
    parser.add_argument("--base-data", type=Path, default=Path("data/sample_preferences.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/pilot_vi_preferences.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/pilot_vi_quality_report.json"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _extract_json(content: str) -> Any:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def parse_examples(content: str) -> list[PreferenceExample]:
    payload = _extract_json(content)
    if not isinstance(payload, list):
        raise TypeError("model output must be a JSON array")
    return [PreferenceExample.model_validate(item) for item in payload]


def normalized_prompt(prompt: str) -> str:
    return " ".join(re.findall(r"\w+", prompt.casefold(), flags=re.UNICODE))


def actual_length_relation(example: PreferenceExample) -> str:
    chosen_words = len(example.chosen.split())
    rejected_words = len(example.rejected.split())
    if chosen_words <= rejected_words * 0.85:
        return "chosen_shorter"
    if rejected_words <= chosen_words * 0.85:
        return "chosen_longer"
    return "comparable"


def local_quality_report(
    examples: list[PreferenceExample], base_examples: list[PreferenceExample]
) -> dict[str, Any]:
    base_prompts = {normalized_prompt(example.prompt) for example in base_examples}
    prompts = [normalized_prompt(example.prompt) for example in examples]
    claimed_relations = [str(example.metadata.get("length_relation")) for example in examples]
    actual_relations = [actual_length_relation(example) for example in examples]
    required_metadata = {"domain", "rubric", "language", "source", "difficulty", "length_relation"}
    violations: list[str] = []

    if len(set(prompts)) != len(prompts):
        violations.append("duplicate prompts inside pilot")
    if any(prompt in base_prompts for prompt in prompts):
        violations.append("pilot prompt duplicates a base-data prompt")
    if any(example.metadata.get("language") != "vi" for example in examples):
        violations.append("one or more samples are not marked language=vi")
    if any(not required_metadata.issubset(example.metadata) for example in examples):
        violations.append("one or more samples are missing required metadata")
    if claimed_relations != actual_relations:
        violations.append("one or more claimed length relations do not match actual lengths")

    domains = Counter(str(example.metadata.get("domain")) for example in examples)
    relations = Counter(actual_relations)
    return {
        "sample_count": len(examples),
        "unique_prompt_count": len(set(prompts)),
        "domains": dict(sorted(domains.items())),
        "actual_length_relations": dict(sorted(relations.items())),
        "average_words": {
            "chosen": sum(len(example.chosen.split()) for example in examples) / len(examples),
            "rejected": sum(len(example.rejected.split()) for example in examples) / len(examples),
        },
        "local_violations": violations,
        "local_checks_passed": not violations,
    }


def build_generation_prompt(count: int) -> str:
    blueprint = [
        {"index": index, "domain": domain, "rubric": rubric, "length_relation": relation}
        for index, (domain, rubric, relation) in enumerate(PILOT_BLUEPRINT[:count], start=1)
    ]
    return (
        f"Tạo đúng {count} mẫu theo blueprint sau:\n"
        f"{json.dumps(blueprint, ensure_ascii=False, indent=2)}\n\n"
        "Few-shot chỉ minh họa tiêu chuẩn chất lượng, tuyệt đối không sao chép:\n"
        f"{json.dumps(FEW_SHOTS, ensure_ascii=False, indent=2)}"
    )


def request_llm_review(client: OpenAI, model: str, examples: list[PreferenceExample]) -> Any:
    review_prompt = """Đánh giá độc lập 10 preference pairs dưới đây. Với mỗi index, chấm 1-5 cho:
chosen_quality, rejected_plausibility, preference_clarity, vietnamese_naturalness.
Đánh dấu length_bias_risk và giải thích ngắn gọn issues. Trả về duy nhất JSON array.

DATA:
""" + json.dumps([item.model_dump() for item in examples], ensure_ascii=False)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là reviewer dữ liệu DPO nghiêm khắc."},
            {"role": "user", "content": review_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("reviewer returned empty content")
    return _extract_json(content)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.count != 10:
        raise ValueError("pilot generation is intentionally fixed at 10 samples")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace the pilot")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL must be set")

    base_examples = load_jsonl(args.base_data)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_generation_prompt(args.count)},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("generator returned empty content")
    examples = parse_examples(content)
    if len(examples) != args.count:
        raise ValueError(f"expected {args.count} examples, received {len(examples)}")

    report = local_quality_report(examples, base_examples)
    try:
        report["llm_review"] = request_llm_review(client, model, examples)
        report["llm_review_error"] = None
    except Exception as exc:  # noqa: BLE001 - preserve pilot even if optional review fails
        report["llm_review"] = None
        report["llm_review_error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Pilot data: {args.output}")
    print(f"Quality report: {args.report}")
    return args.output, args.report


if __name__ == "__main__":
    run(parse_args())
