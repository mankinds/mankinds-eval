"""Global configuration management for mankinds-eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Hardcoded default models
DEFAULT_MODELS: dict[str, str] = {
    "embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentiment": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "ner": "Isotonic/distilbert_finetuned_ai4privacy_v2",
}


def get_config_path() -> Path:
    """Returns the path to the config file.

    Returns:
        Path to ~/.mankinds_eval/config.yaml
    """
    return Path.home() / ".mankinds_eval" / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load configuration from file.

    Returns:
        Configuration dictionary, or empty dict if file doesn't exist.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return config if config is not None else {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file.

    Creates the config directory if it doesn't exist.

    Args:
        config: Configuration dictionary to save.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_model(model_type: str, override: str | None = None) -> str:
    """Get model name with priority: override > config file > hardcoded default.

    Args:
        model_type: Type of model (e.g., "embeddings", "sentiment", "ner").
        override: Optional override value that takes highest priority.

    Returns:
        Model name string.

    Raises:
        ValueError: If model_type is not recognized and no override is provided.
    """
    # Priority 1: Override
    if override is not None:
        return override

    # Priority 2: Config file
    config = load_config()
    models = config.get("models", {})
    if model_type in models:
        return str(models[model_type])

    # Priority 3: Hardcoded default
    if model_type in DEFAULT_MODELS:
        return DEFAULT_MODELS[model_type]

    raise ValueError(f"Unknown model type: {model_type}")
