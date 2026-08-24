import pytest

from preference_lab.evaluate import deterministic_text_score, pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_handles_ties_explicitly() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [1.0], [1.0]) == 0.0
    assert pairwise_accuracy(examples, [1.0], [1.0], tie_score=0.5) == 0.5


def test_pairwise_accuracy_validates_lengths() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="equal lengths"):
        pairwise_accuracy(examples, [], [1.0])


def test_deterministic_score_is_reproducible() -> None:
    text = "A complete and informative answer."
    assert deterministic_text_score(text) == deterministic_text_score(text)
    assert deterministic_text_score("") == 0.0
