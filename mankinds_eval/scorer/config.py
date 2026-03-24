"""Configuration loading and method resolution for scorers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from mankinds_eval.methods import Method


def load_scorer_config(path: str | Path) -> dict[str, Any]:
    """Load and validate scorer config from YAML or JSON.

    Args:
        path: Path to the configuration file (.yaml, .yml, or .json).

    Returns:
        Dictionary containing scorer configuration.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If file format is unsupported or config is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML config files. Install it with: pip install pyyaml"
            ) from e

        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            config = json.load(f)

    else:
        raise ValueError(
            f"Unsupported config format: '{suffix}'. Supported formats: .yaml, .yml, .json"
        )

    # Validate required fields
    _validate_config(config)

    assert isinstance(config, dict)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Validate scorer configuration.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    if "name" not in config:
        raise ValueError("Config must have 'name' field")

    if "methods" not in config:
        raise ValueError("Config must have 'methods' field")

    if not isinstance(config["methods"], list):
        raise ValueError("'methods' must be a list")

    if len(config["methods"]) == 0:
        raise ValueError("'methods' list cannot be empty")

    for i, method_config in enumerate(config["methods"]):
        if not isinstance(method_config, dict):
            raise ValueError(f"methods[{i}] must be a dictionary")
        if "type" not in method_config:
            raise ValueError(f"methods[{i}] must have 'type' field")


def resolve_method(method_config: dict[str, Any]) -> Method:
    """Resolve method from config dict.

    Config format:
    {
        "type": "heuristic.FuzzyMatch",  # Full path relative to mankinds_eval.methods
        "threshold": 0.8,
        ...
    }

    For composite methods:
    {
        "type": "composite",  # or "composite.CompositeMethod"
        "mode": "all",
        "methods": [
            {"type": "heuristic.TextLength", "min_length": 50},
            {"type": "heuristic.NoRefusal"}
        ]
    }

    Maps "type" to actual class and instantiates with remaining config.

    Args:
        method_config: Dictionary containing method type and configuration.

    Returns:
        Instantiated Method object.

    Raises:
        ValueError: If method type is invalid or class cannot be found.
        ImportError: If method module cannot be imported.
    """
    config = method_config.copy()
    method_type = config.pop("type")

    # Handle composite method types with recursive resolution
    if method_type == "composite" or method_type.startswith("composite."):
        from mankinds_eval.methods.composite import CompositeMethod

        # Extract and recursively resolve nested methods
        nested_method_configs = config.pop("methods", [])
        if not isinstance(nested_method_configs, list):
            raise ValueError("Composite method 'methods' must be a list")

        resolved_methods = resolve_methods(nested_method_configs)

        # Instantiate CompositeMethod with resolved methods and remaining config
        return CompositeMethod(methods=resolved_methods, **config)

    # Parse the method type path
    # Format: "submodule.ClassName" or just "ClassName"
    parts = method_type.rsplit(".", 1)

    if len(parts) == 1:
        # Just class name, look in methods base
        class_name = parts[0]
        module_path = "mankinds_eval.methods"
    else:
        # submodule.ClassName format
        submodule, class_name = parts
        module_path = f"mankinds_eval.methods.{submodule}"

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import method module '{module_path}': {e}") from e

    if not hasattr(module, class_name):
        raise ValueError(f"Method class '{class_name}' not found in module '{module_path}'")

    method_class = getattr(module, class_name)

    if not isinstance(method_class, type) or not issubclass(method_class, Method):
        raise ValueError(f"'{class_name}' is not a valid Method subclass")

    # Instantiate with remaining config
    return method_class(**config)


def resolve_methods(method_configs: list[dict[str, Any]]) -> list[Method]:
    """Resolve multiple methods from config list.

    Args:
        method_configs: List of method configuration dictionaries.

    Returns:
        List of instantiated Method objects.
    """
    return [resolve_method(config) for config in method_configs]
