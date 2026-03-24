"""Command line interface for mankinds-eval."""

import typer

from mankinds_eval.cli.commands import cache, config, evaluate, methods

app = typer.Typer(
    name="mankinds-eval",
    help="AI evaluation toolkit - assemble methods to create custom scorers",
)

app.add_typer(config.app, name="config")
app.command(name="run")(evaluate.run)
app.command(name="dry-run")(evaluate.dry_run)
app.command(name="init")(config.init)
app.command(name="version")(evaluate.version)
app.command(name="list-methods")(methods.list_methods)
app.command(name="cache")(cache.cache_cmd)


def main() -> None:
    """Entry point for the CLI."""
    app()
