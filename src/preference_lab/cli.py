from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import deterministic_text_score, pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(
    data: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    pii_guard: Annotated[
        bool, typer.Option("--pii-guard", help="Reject common email and phone-number patterns")
    ] = False,
) -> None:
    examples = load_jsonl(data, pii_guard=pii_guard)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="Experiment YAML"),
    ],
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    chosen_scores = [deterministic_text_score(example.chosen) for example in examples]
    rejected_scores = [deterministic_text_score(example.rejected) for example in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote deterministic CPU metrics to {out}[/green]")


@app.command("train-mock")
def train_mock(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="Experiment YAML"),
    ],
) -> None:
    """Run the local deterministic trainer without model weight updates."""
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    training = cfg["training"]
    trainer = PreferenceTrainer(
        TrainingConfig(
            method="mock",
            beta=float(training["beta"]),
            lambda_orpo=float(training["lambda_orpo"]),
            max_length=int(training["max_length"]),
            batch_size=int(training["batch_size"]),
        ),
        output_dir=cfg["paths"]["output_dir"],
    )
    metrics = trainer.train(examples)
    print(f"[green]Completed CPU mock pass: {metrics}[/green]")


if __name__ == "__main__":
    app()
