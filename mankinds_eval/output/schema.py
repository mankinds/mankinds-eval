"""Schema for evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    """Container for complete evaluation results.

    Attributes:
        meta: Metadata about the evaluation run including scorer_name,
            created_at, version, methods, sample_count, duration.
        summary: Aggregated statistics per method.
        results: Per-sample results with method outputs.
    """

    meta: dict[str, Any]
    summary: dict[str, Any]
    results: list[dict[str, Any]]

    def to_json(self, path: str | Path) -> None:
        """Write results to JSON file.

        Args:
            path: Path to write the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the evaluation result.
        """
        return {
            "meta": self.meta,
            "summary": self.summary,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Create an EvaluationResult from a dictionary.

        Args:
            data: Dictionary containing meta, summary, and results.

        Returns:
            A new EvaluationResult instance.

        Raises:
            KeyError: If required fields are missing.
        """
        return cls(
            meta=data["meta"],
            summary=data["summary"],
            results=data["results"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationResult:
        """Load an EvaluationResult from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A new EvaluationResult instance.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_html(self, path: str | Path) -> None:
        """Generate HTML scorecard.

        Args:
            path: Path to write the HTML file.
        """
        from mankinds_eval.output.html import generate_scorecard

        generate_scorecard(self, path)


def compute_summary(
    results: list[dict[str, Any]],
    method_names: list[str],
) -> dict[str, Any]:
    """Compute summary statistics for evaluation results.

    Args:
        results: List of per-sample result dictionaries.
        method_names: List of method names to compute stats for.

    Returns:
        Dictionary with per-method statistics including:
        - mean_score: Average score across samples
        - pass_rate: Proportion of samples that passed (if applicable)
        - min_score: Minimum score
        - max_score: Maximum score
        - error_count: Number of errors
        - sample_count: Total samples evaluated
        - score_distribution: Distribution for discrete scores (if applicable)
    """
    summary: dict[str, Any] = {}

    for method_name in method_names:
        scores: list[float] = []
        passed_count = 0
        failed_count = 0
        error_count = 0
        has_passed_field = False
        score_counts: dict[float, int] = {}

        for result in results:
            method_result = result.get("methods", {}).get(method_name)
            if method_result is None:
                continue

            # Handle errors
            if method_result.get("error") is not None:
                error_count += 1
                continue

            # Collect scores
            score = method_result.get("score")
            if score is not None:
                scores.append(score)
                # Track distribution for discrete scores
                score_counts[score] = score_counts.get(score, 0) + 1

            # Track pass/fail
            passed = method_result.get("passed")
            if passed is not None:
                has_passed_field = True
                if passed:
                    passed_count += 1
                else:
                    failed_count += 1

        method_summary: dict[str, Any] = {
            "sample_count": len(results),
            "error_count": error_count,
        }

        if scores:
            method_summary["mean_score"] = sum(scores) / len(scores)
            method_summary["min_score"] = min(scores)
            method_summary["max_score"] = max(scores)
            method_summary["scored_count"] = len(scores)

            # Check if scores are discrete (limited unique values)
            unique_scores = set(scores)
            if len(unique_scores) <= 10:  # Threshold for discrete distribution
                method_summary["score_distribution"] = {
                    str(k): v for k, v in sorted(score_counts.items())
                }

        if has_passed_field:
            total_judged = passed_count + failed_count
            if total_judged > 0:
                method_summary["pass_rate"] = passed_count / total_judged
                method_summary["passed_count"] = passed_count
                method_summary["failed_count"] = failed_count

        summary[method_name] = method_summary

    return summary
