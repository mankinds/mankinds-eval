"""Output module for evaluation results."""

from mankinds_eval.output.html import generate_scorecard, render_scorecard
from mankinds_eval.output.schema import EvaluationResult, compute_summary

__all__ = [
    "EvaluationResult",
    "compute_summary",
    "generate_scorecard",
    "render_scorecard",
]
