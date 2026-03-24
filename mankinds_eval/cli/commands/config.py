"""Configuration management commands."""

from typing import Annotated

import typer

from mankinds_eval.config import (
    DEFAULT_MODELS,
    get_config_path,
    load_config,
    save_config,
)

app = typer.Typer(help="Configuration management commands.")


def init() -> None:
    """Interactive setup for mankinds-eval configuration."""
    typer.echo("mankinds-eval Configuration Setup")
    typer.echo("=" * 40)
    typer.echo()
    typer.echo("Current default models:")
    for model_type, model_name in DEFAULT_MODELS.items():
        typer.echo(f"  {model_type}: {model_name}")
    typer.echo()

    customize = typer.confirm("Do you want to customize the model settings?", default=False)

    if not customize:
        typer.echo("Using default configuration.")
        return

    typer.echo()
    typer.echo("Enter custom model names (press Enter to keep default):")
    typer.echo()

    models: dict[str, str] = {}
    for model_type, default_model in DEFAULT_MODELS.items():
        prompt = f"{model_type} [{default_model}]"
        custom_value = typer.prompt(prompt, default="", show_default=False)
        if custom_value.strip():
            models[model_type] = custom_value.strip()
        else:
            models[model_type] = default_model

    config = load_config()
    config["models"] = models
    save_config(config)

    config_path = get_config_path()
    typer.echo()
    typer.echo(f"Configuration saved to: {config_path}")


@app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Config key (e.g., 'models.embeddings')")],
) -> None:
    """Get a configuration value by key."""
    config = load_config()

    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            typer.echo(f"Key not found: {key}")
            raise typer.Exit(1)

    typer.echo(value)


@app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g., 'models.embeddings')")],
    value: Annotated[str, typer.Argument(help="Value to set")],
) -> None:
    """Set a configuration value."""
    config = load_config()

    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value
    save_config(config)

    typer.echo(f"Set {key} = {value}")


@app.command("show")
def config_show() -> None:
    """Show the full configuration."""
    import yaml

    config = load_config()
    config_path = get_config_path()

    if not config:
        typer.echo(f"No configuration file found at: {config_path}")
        typer.echo()
        typer.echo("Default models:")
        for model_type, model_name in DEFAULT_MODELS.items():
            typer.echo(f"  {model_type}: {model_name}")
        return

    typer.echo(f"Configuration file: {config_path}")
    typer.echo()
    typer.echo(yaml.dump(config, default_flow_style=False, allow_unicode=True))
