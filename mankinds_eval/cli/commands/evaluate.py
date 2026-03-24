"""Evaluation commands."""

from typing import Annotated, Optional

import typer
from rich.console import Console

console = Console()


def run(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to scorer config file"),
    ],
    data: Annotated[
        str,
        typer.Option("--data", "-d", help="Path to data file"),
    ],
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Path to output JSON file"),
    ] = None,
    html: Annotated[
        Optional[str],
        typer.Option("--html", help="Path to output HTML scorecard"),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Max concurrent evaluations"),
    ] = 10,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """
    Run evaluation using a scorer config on a data file.

    Example:
        mankinds-eval run -c scorer.yaml -d data.jsonl -o results.json --html scorecard.html
    """
    import asyncio
    from pathlib import Path

    from mankinds_eval import Scorer, load_samples

    if not Path(config).exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    if not Path(data).exists():
        typer.echo(f"Error: Data file not found: {data}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Loading scorer from {config}...")

    try:
        scorer = Scorer.from_config(config)
    except Exception as e:
        typer.echo(f"Error loading scorer config: {e}", err=True)
        raise typer.Exit(1) from e

    if verbose:
        typer.echo(f"Scorer '{scorer.name}' loaded with {len(scorer.methods)} methods")
        typer.echo(f"Loading data from {data}...")

    try:
        samples = load_samples(data)
    except Exception as e:
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(1) from e

    if verbose:
        typer.echo(f"Loaded {len(samples)} samples")
        typer.echo("Running evaluation...")

    try:
        result = asyncio.run(scorer.run(samples, concurrency=concurrency))
    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        raise typer.Exit(1) from e

    if output:
        result.to_json(output)
        typer.echo(f"Results written to {output}")

    if html:
        result.to_html(html)
        typer.echo(f"Scorecard written to {html}")

    typer.echo("\nEvaluation complete:")
    typer.echo(f"  Samples: {result.meta['sample_count']}")
    typer.echo(f"  Duration: {result.meta['duration_seconds']:.2f}s")
    typer.echo("\nSummary by method:")
    for method_name, stats in result.summary.items():
        mean = stats.get("mean_score", "N/A")
        pass_rate = stats.get("pass_rate")
        if isinstance(mean, float):
            mean = f"{mean:.3f}"
        if pass_rate is not None:
            typer.echo(f"  {method_name}: mean={mean}, pass_rate={pass_rate:.1%}")
        else:
            typer.echo(f"  {method_name}: mean={mean}")

    if not output and not html:
        typer.echo("\nTip: Use --output or --html to save results")


def dry_run(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to scorer config file"),
    ],
    data: Annotated[
        str,
        typer.Option("--data", "-d", help="Path to data file"),
    ],
) -> None:
    """Validate config and data without running evaluation."""
    from pathlib import Path

    from mankinds_eval import Scorer, load_samples

    errors = []

    config_path = Path(config)
    if not config_path.exists():
        errors.append(f"Config file not found: {config}")
    else:
        try:
            scorer = Scorer.from_config(config)
            console.print(f"[green]Config valid:[/green] {config}")
            console.print(f"  Scorer name: {scorer.name}")
            console.print(f"  Methods: {len(scorer.methods)}")
            for method in scorer.methods:
                console.print(f"    - {method.name}")
        except Exception as e:
            errors.append(f"Config error: {e}")

    data_path = Path(data)
    if not data_path.exists():
        errors.append(f"Data file not found: {data}")
    else:
        try:
            samples = load_samples(data)
            console.print(f"[green]Data valid:[/green] {data}")
            console.print(f"  Samples: {len(samples)}")
            if samples:
                sample = samples[0]
                console.print("  Fields: input, output", end="")
                if sample.expected:
                    console.print(", expected", end="")
                if sample.context:
                    console.print(", context", end="")
                console.print()
        except Exception as e:
            errors.append(f"Data error: {e}")

    if errors:
        console.print()
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)

    console.print()
    console.print("[green]Validation passed. Ready to run evaluation.[/green]")


def version() -> None:
    """Show mankinds-eval version."""
    from mankinds_eval import __version__

    typer.echo(f"mankinds-eval {__version__}")
