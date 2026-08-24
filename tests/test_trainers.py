import json
from pathlib import Path

import pytest

from preference_lab.schemas import PreferenceExample
from preference_lab.trainers import PreferenceTrainer, TrainingConfig


def test_mock_trainer_writes_metrics(tmp_path: Path) -> None:
    examples = [
        PreferenceExample(
            prompt="Explain it",
            chosen="This is a complete and useful explanation.",
            rejected="No.",
        )
    ]
    trainer = PreferenceTrainer(TrainingConfig(method="mock"), output_dir=tmp_path)
    metrics = trainer.train(examples)

    output = tmp_path / "metrics.json"
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == metrics


def test_local_trainer_does_not_pretend_to_run_dpo() -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="dpo"))
    example = PreferenceExample(prompt="p", chosen="a", rejected="b")
    with pytest.raises(RuntimeError, match="TRL-backed Colab"):
        trainer.train([example])
