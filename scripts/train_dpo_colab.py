from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from pathlib import Path
from typing import Any

import yaml

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost-aware DPO training on Colab A100")
    parser.add_argument("--config", type=Path, default=Path("configs/colab_a100.yaml"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path}: configuration root must be a mapping")
    return loaded


def verify_gpu(runtime_config: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        if runtime_config["fail_if_cuda_unavailable"]:
            raise RuntimeError(
                "CUDA is unavailable; stop the Colab runtime before spending more units"
            )
        return {"name": "cpu", "vram_gb": 0.0}

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    name = torch.cuda.get_device_name(device_index)
    vram_gb = properties.total_memory / 1024**3
    expected = str(runtime_config["required_gpu_name_contains"])
    minimum_vram_gb = float(runtime_config["minimum_vram_gb"])

    if runtime_config["fail_if_wrong_gpu"] and expected.casefold() not in name.casefold():
        raise RuntimeError(f"Expected an {expected} GPU, but Colab assigned {name}")
    if vram_gb < minimum_vram_gb:
        raise RuntimeError(
            f"Expected at least {minimum_vram_gb:.1f} GiB VRAM, but only {vram_gb:.1f} GiB is visible"
        )
    return {"name": name, "vram_gb": round(vram_gb, 2)}


def validate_trl_api() -> None:
    from trl import DPOConfig

    available = set(inspect.signature(DPOConfig).parameters)
    required = {
        "eval_strategy",
        "max_length",
        "precompute_ref_log_probs",
        "precompute_ref_batch_size",
    }
    missing = required - available
    if missing:
        raise RuntimeError(
            "Installed TRL is incompatible with this notebook; missing DPOConfig fields: "
            + ", ".join(sorted(missing))
        )


def as_conversational_row(example: PreferenceExample) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": example.prompt}],
        "chosen": [{"role": "assistant", "content": example.chosen}],
        "rejected": [{"role": "assistant", "content": example.rejected}],
        "metadata": example.metadata,
    }


def build_datasets(config: dict[str, Any]) -> tuple[Any, Any, list[PreferenceExample]]:
    from datasets import Dataset

    examples = load_jsonl(config["paths"]["train_data"])
    train_examples, validation_examples = split_by_prompt(
        examples,
        validation_ratio=float(config["data"]["validation_ratio"]),
        seed=int(config["seed"]),
    )
    train_dataset = Dataset.from_list([as_conversational_row(item) for item in train_examples])
    validation_dataset = Dataset.from_list(
        [as_conversational_row(item) for item in validation_examples]
    )
    return train_dataset, validation_dataset, validation_examples


def _chat_text(tokenizer: Any, prompt: str, response: str | None = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    if response is None:
        return str(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    messages.append({"role": "assistant", "content": response})
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))


def score_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> list[float]:
    """Return mean response-token log probabilities in deterministic batches."""
    import torch

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal lengths")

    full_texts = [_chat_text(tokenizer, p, r) for p, r in zip(prompts, responses, strict=True)]
    prompt_texts = [_chat_text(tokenizer, prompt) for prompt in prompts]
    prompt_lengths = [
        min(len(tokenizer(text, add_special_tokens=False)["input_ids"]), max_length)
        for text in prompt_texts
    ]

    scores: list[float] = []
    model.eval()
    for start in range(0, len(full_texts), batch_size):
        stop = start + batch_size
        texts = full_texts[start:stop]
        lengths = prompt_lengths[start:stop]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            logits = model(**encoded).logits[:, :-1].float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            target_ids = encoded["input_ids"][:, 1:]
            token_logps = log_probabilities.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        response_mask = encoded["attention_mask"][:, 1:].bool()
        sequence_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        padded_length = encoded["input_ids"].shape[1]
        for row, (prompt_length, sequence_length) in enumerate(
            zip(lengths, sequence_lengths, strict=True)
        ):
            padding_length = padded_length - int(sequence_length)
            response_start = max(0, padding_length + prompt_length - 1)
            response_mask[row, :response_start] = False
            token_count = int(response_mask[row].sum().item())
            if token_count == 0:
                raise RuntimeError("A response was fully truncated; increase data.max_length")
            score = token_logps[row][response_mask[row]].mean().item()
            scores.append(float(score))
    return scores


def evaluate_pairs(
    model: Any,
    tokenizer: Any,
    examples: list[PreferenceExample],
    config: dict[str, Any],
) -> dict[str, float]:
    from preference_lab.evaluate import pairwise_accuracy

    prompts = [example.prompt for example in examples]
    batch_size = int(config["evaluation"]["pairwise_batch_size"])
    max_length = int(config["data"]["max_length"])
    chosen_scores = score_responses(
        model,
        tokenizer,
        prompts,
        [example.chosen for example in examples],
        batch_size=batch_size,
        max_length=max_length,
    )
    rejected_scores = score_responses(
        model,
        tokenizer,
        prompts,
        [example.rejected for example in examples],
        batch_size=batch_size,
        max_length=max_length,
    )
    margins = [chosen - rejected for chosen, rejected in zip(chosen_scores, rejected_scores)]
    return {
        "pairwise_accuracy": pairwise_accuracy(
            examples,
            chosen_scores,
            rejected_scores,
            tie_score=float(config["evaluation"]["tie_score"]),
        ),
        "mean_logprob_margin": sum(margins) / len(margins),
        "mean_chosen_logprob": sum(chosen_scores) / len(chosen_scores),
        "mean_rejected_logprob": sum(rejected_scores) / len(rejected_scores),
    }


def load_regression_prompts(path: str | Path) -> list[str]:
    prompts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit() or "." not in stripped:
            continue
        prompts.append(stripped.split(".", maxsplit=1)[1].strip())
    if not prompts:
        raise ValueError(f"No numbered regression prompts found in {path}")
    return prompts


def generate_regression_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    import torch

    rendered = [_chat_text(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(rendered, padding=True, return_tensors="pt").to(model.device)
    previous_use_cache = bool(model.config.use_cache)
    model.config.use_cache = True
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(config["evaluation"]["generation_max_new_tokens"]),
            do_sample=bool(config["evaluation"]["do_sample"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model.config.use_cache = previous_use_cache
    prompt_width = encoded["input_ids"].shape[1]
    responses = tokenizer.batch_decode(output_ids[:, prompt_width:], skip_special_tokens=True)
    return [
        {"prompt": prompt, "response": response.strip()}
        for prompt, response in zip(prompts, responses, strict=True)
    ]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def run(config_path: Path) -> Path:
    import datasets
    import peft
    import torch
    import transformers
    import trl
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    started_at = time.perf_counter()
    config = load_yaml(config_path)
    gpu = verify_gpu(config["runtime"])
    validate_trl_api()
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["training"]["tf32"])
    torch.cuda.reset_peak_memory_stats()

    output_dir = Path(config["paths"]["output_dir"])
    adapter_dir = Path(config["paths"]["adapter_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    dtype = torch.bfloat16 if config["model"]["dtype"] == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        dtype=dtype,
        attn_implementation=config["model"]["attn_implementation"],
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        low_cpu_mem_usage=True,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = bool(config["model"]["use_cache"])

    train_dataset, validation_dataset, validation_examples = build_datasets(config)
    regression_prompts = load_regression_prompts(config["evaluation"]["regression_prompts"])

    baseline_metrics = evaluate_pairs(model, tokenizer, validation_examples, config)
    baseline_regression = generate_regression_outputs(model, tokenizer, regression_prompts, config)

    training = config["training"]
    data = config["data"]
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        beta=float(training["beta"]),
        loss_type=str(training["loss_type"]),
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        warmup_steps=float(training["warmup_steps"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        optim=str(training["optim"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        bf16=bool(training["bf16"]),
        fp16=bool(training["fp16"]),
        tf32=bool(training["tf32"]),
        precompute_ref_log_probs=bool(training["precompute_ref_log_probs"]),
        precompute_ref_batch_size=int(training["precompute_ref_batch_size"]),
        auto_find_batch_size=bool(training["auto_find_batch_size"]),
        dataloader_num_workers=int(training["dataloader_num_workers"]),
        dataloader_pin_memory=bool(training["dataloader_pin_memory"]),
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        eval_strategy=str(training["eval_strategy"]),
        save_strategy=str(training["save_strategy"]),
        save_total_limit=int(training["save_total_limit"]),
        save_only_model=bool(training["save_only_model"]),
        load_best_model_at_end=bool(training["load_best_model_at_end"]),
        metric_for_best_model=str(training["metric_for_best_model"]),
        greater_is_better=bool(training["greater_is_better"]),
        report_to=str(training["report_to"]),
        torch_compile=bool(training["torch_compile"]),
        use_liger_kernel=bool(training["use_liger_kernel"]),
        dataset_num_proc=int(data["dataset_num_proc"]),
        max_length=int(data["max_length"]),
        truncation_mode=str(data["truncation_mode"]),
        pad_to_multiple_of=int(data["pad_to_multiple_of"]),
    )
    lora = config["lora"]
    lora_config = LoraConfig(
        task_type=str(lora["task_type"]),
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        target_modules=str(lora["target_modules"]),
        bias=str(lora["bias"]),
        use_rslora=bool(lora["use_rslora"]),
    )

    class WallClockLimitCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            elapsed_minutes = (time.perf_counter() - started_at) / 60
            if elapsed_minutes >= float(config["runtime"]["max_wall_minutes"]):
                control.should_training_stop = True
            return control

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[WallClockLimitCallback()],
    )
    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    evaluation_metrics = trainer.evaluate()

    post_metrics = evaluate_pairs(trainer.model, tokenizer, validation_examples, config)
    post_regression = generate_regression_outputs(
        trainer.model, tokenizer, regression_prompts, config
    )

    elapsed_minutes = (time.perf_counter() - started_at) / 60
    metrics = {
        "run": {
            "seed": int(config["seed"]),
            "model": config["model"]["name_or_path"],
            "gpu": gpu,
            "elapsed_minutes": round(elapsed_minutes, 3),
            "peak_allocated_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "platform": platform.platform(),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "datasets": datasets.__version__,
                "trl": trl.__version__,
                "peft": peft.__version__,
            },
        },
        "baseline": baseline_metrics,
        "post_dpo": post_metrics,
        "delta_pairwise_accuracy": post_metrics["pairwise_accuracy"]
        - baseline_metrics["pairwise_accuracy"],
        "trainer": {
            "train": json_safe(train_result.metrics),
            "evaluation": json_safe(evaluation_metrics),
        },
        "regression": {"baseline": baseline_regression, "post_dpo": post_regression},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(metrics), indent=2, ensure_ascii=False))
    print(f"Saved adapter to {adapter_dir}")
    print(f"Saved metrics to {metrics_path}")
    return metrics_path


if __name__ == "__main__":
    run(parse_args().config)
