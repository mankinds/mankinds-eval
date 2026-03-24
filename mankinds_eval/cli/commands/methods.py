"""Method listing commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def list_methods(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category (heuristic, ml, llm)"
    ),
) -> None:
    """List all available evaluation methods."""
    methods = {
        "heuristic": [
            ("ExactMatch", "Exact string comparison"),
            ("FuzzyMatch", "Fuzzy string similarity"),
            ("RegexMatch", "Regular expression matching"),
            ("ContainsAll", "Check if output contains all keywords"),
            ("ContainsAny", "Check if output contains any keyword"),
            ("TextLength", "Validate text length"),
            ("WordCount", "Validate word count"),
            ("SentenceCount", "Validate sentence count"),
            ("BLEU", "BLEU score for translation quality"),
            ("ROUGE", "ROUGE score for summarization"),
            ("JSONValid", "Validate JSON syntax"),
            ("JSONSchema", "Validate against JSON Schema"),
            ("NoRefusal", "Detect LLM refusals"),
        ],
        "ml": [
            ("EmbeddingsSimilarity", "Semantic similarity via embeddings"),
            ("SentimentAnalysis", "Analyze sentiment"),
            ("Toxicity", "Detect toxic content"),
            ("PIIDetection", "Detect PII/NER entities"),
            ("LanguageDetection", "Detect/verify language"),
            ("ZeroShotClassification", "Zero-shot text classification"),
        ],
        "llm": [
            ("Faithfulness", "Check if response is grounded in context"),
            ("AnswerRelevancy", "Check if response addresses the question"),
            ("Coherence", "Evaluate logical flow and clarity"),
            ("Helpfulness", "Evaluate practical utility"),
            ("Correctness", "Compare against expected answer"),
            ("SingleCriterionJudge", "Custom single criterion evaluation"),
            ("MultiCriteriaJudge", "Multi-criteria weighted scoring"),
            ("PairwiseJudge", "Compare two responses"),
            ("ConsensusJudge", "Multi-LLM consensus evaluation"),
        ],
    }

    categories = [category] if category else ["heuristic", "ml", "llm"]

    table = Table(title="Available Evaluation Methods")
    table.add_column("Category", style="cyan")
    table.add_column("Method", style="green")
    table.add_column("Description")

    for cat in categories:
        if cat not in methods:
            typer.echo(f"Unknown category: {cat}", err=True)
            continue
        for name, desc in methods[cat]:
            table.add_row(cat, name, desc)

    console.print(table)
