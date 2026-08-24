from preference_lab.config import load_config


def test_colab_a100_config_is_cost_aware() -> None:
    config = load_config("configs/colab_a100.yaml")

    assert config["secrets"] == {
        "huggingface_token_name": "HF_TOKEN",
        "llm_api_key_name": "XAH_API_KEY",
    }
    assert config["llm"]["base_url"] == "https://api.xah.io/v1"
    assert config["llm"]["model"] == "levuphong2909/gpt-5.6-luna"
    assert config["llm"]["route"] == "/chat/completions"
    assert config["runtime"]["required_gpu_name_contains"] == "A100"
    assert config["model"]["dtype"] == "bfloat16"
    assert config["model"]["quantization"] == "none"
    assert config["training"]["gradient_checkpointing"] is False
    assert config["training"]["precompute_ref_log_probs"] is True
    assert config["training"]["save_total_limit"] == 1
    assert config["artifacts"]["save_merged_model"] is False
