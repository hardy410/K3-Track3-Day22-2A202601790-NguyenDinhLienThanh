import json
from pathlib import Path

from scripts.train_dpo_colab import (
    as_conversational_row,
    json_safe,
    load_regression_prompts,
)

from preference_lab.schemas import PreferenceExample


def test_colab_notebook_has_gpu_metadata_and_ordered_cells() -> None:
    notebook_path = Path("notebooks/day22_dpo_a100_colab.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    assert len(notebook["cells"]) == 9


def test_preference_example_is_converted_to_conversational_format() -> None:
    example = PreferenceExample(
        prompt="Question",
        chosen="Preferred answer",
        rejected="Rejected answer",
        metadata={"domain": "test"},
    )
    row = as_conversational_row(example)

    assert row["prompt"] == [{"role": "user", "content": "Question"}]
    assert row["chosen"] == [{"role": "assistant", "content": "Preferred answer"}]
    assert row["rejected"] == [{"role": "assistant", "content": "Rejected answer"}]


def test_regression_prompts_are_loaded_from_numbered_list() -> None:
    prompts = load_regression_prompts("docs/regression_prompts.md")
    assert len(prompts) == 4
    assert prompts[0] == "Ask for high-risk medical advice."


def test_json_safe_handles_scalar_like_values() -> None:
    class Scalar:
        def item(self) -> float:
            return 1.25

    assert json_safe({"value": Scalar()}) == {"value": 1.25}
