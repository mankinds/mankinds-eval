"""Pre-configured scorer combinations for common evaluation scenarios.

This module provides factory functions that return pre-configured sets of
evaluation methods for common use cases like RAG evaluation and safety checks.

Example:
    >>> from mankinds_eval.methods.presets import RAGScorer, SafetyScorer
    >>> rag_methods = RAGScorer.create(provider="openai")
    >>> safety_methods = SafetyScorer.create(check_pii=True, check_toxicity=True)
"""

from mankinds_eval.methods.presets.rag import RAGScorer
from mankinds_eval.methods.presets.safety import SafetyScorer

__all__ = [
    "RAGScorer",
    "SafetyScorer",
]
