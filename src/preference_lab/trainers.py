from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import deterministic_text_score, pairwise_accuracy, write_metrics
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2

    def __post_init__(self) -> None:
        if self.method not in {"dpo", "orpo", "mock"}:
            raise ValueError("method must be one of: dpo, orpo, mock")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.lambda_orpo < 0:
            raise ValueError("lambda_orpo must be non-negative")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")


class PreferenceTrainer:
    """CPU mock trainer used to validate the end-to-end training contract.

    Real DPO/ORPO parameter updates are intentionally delegated to the TRL-backed
    Colab stage. The mock mode produces deterministic metrics without pretending
    that model weights were updated.
    """

    def __init__(self, config: TrainingConfig, output_dir: str | Path = "outputs") -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    def train(self, examples: list[PreferenceExample]) -> dict[str, float]:
        """Run a deterministic CPU smoke-training pass and persist metrics."""
        if self.config.method != "mock":
            raise RuntimeError(
                "The local PreferenceTrainer supports method='mock' only; "
                "use the TRL-backed Colab workflow for DPO/ORPO weight updates"
            )
        if not examples:
            raise ValueError("at least one preference example is required")

        chosen_scores = [deterministic_text_score(example.chosen) for example in examples]
        rejected_scores = [deterministic_text_score(example.rejected) for example in examples]
        margins = np.asarray(chosen_scores) - np.asarray(rejected_scores)
        metrics = {
            "pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores),
            "mean_preference_margin": float(np.mean(margins)),
            "num_examples": float(len(examples)),
        }
        write_metrics(metrics, self.output_dir)
        return metrics
