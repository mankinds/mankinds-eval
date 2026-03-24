"""Cache management commands."""

import typer
from rich.console import Console

console = Console()


def cache_cmd(
    action: str = typer.Argument(..., help="Action: clear, size, path"),
) -> None:
    """Manage evaluation cache."""
    from mankinds_eval.cache import ResultCache

    cache = ResultCache(enabled=True)

    if action == "clear":
        count = cache.clear()
        console.print(f"Cleared {count} cache entries")
    elif action == "size":
        size = cache.size()
        console.print(f"Cache contains {size} entries")
    elif action == "path":
        console.print(f"Cache directory: {cache.cache_dir}")
    else:
        console.print(f"[red]Unknown action:[/red] {action}")
        console.print("Available actions: clear, size, path")
        raise typer.Exit(1)
