"""Machine learning evaluation methods.

This module provides ML-based evaluation methods that use pre-trained models
for semantic analysis, sentiment detection, entity recognition, and classification.

Note: Most methods require the [ml] extras to be installed:
    pip install mankinds-eval[ml]

LanguageDetection uses langdetect which is included in base dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# LanguageDetection uses langdetect (base dependency), always available
from mankinds_eval.methods.ml.language import LanguageDetection

# Lazy imports to handle missing dependencies gracefully for ML-heavy methods
_IMPORTS_AVAILABLE = True
_IMPORT_ERROR: str | None = None

try:
    from mankinds_eval.methods.ml.classification import ZeroShotClassification
    from mankinds_eval.methods.ml.embeddings import EmbeddingsSimilarity
    from mankinds_eval.methods.ml.ner import PIIDetection
    from mankinds_eval.methods.ml.sentiment import SentimentAnalysis
    from mankinds_eval.methods.ml.toxicity import Toxicity
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    # Define placeholder classes for type checking
    if TYPE_CHECKING:
        from mankinds_eval.methods.ml.classification import ZeroShotClassification
        from mankinds_eval.methods.ml.embeddings import EmbeddingsSimilarity
        from mankinds_eval.methods.ml.ner import PIIDetection
        from mankinds_eval.methods.ml.sentiment import SentimentAnalysis
        from mankinds_eval.methods.ml.toxicity import Toxicity


def _check_imports() -> None:
    """Check if ML dependencies are available.

    Raises:
        ImportError: If ML dependencies are not installed.
    """
    if not _IMPORTS_AVAILABLE:
        raise ImportError(
            f"ML methods require additional dependencies. "
            f"Install them with: pip install mankinds-eval[ml]\n"
            f"Original error: {_IMPORT_ERROR}"
        )


def get_embeddings_similarity(*args: Any, **kwargs: Any) -> EmbeddingsSimilarity:
    """Factory function for EmbeddingsSimilarity with dependency check."""
    _check_imports()
    return EmbeddingsSimilarity(*args, **kwargs)


def get_sentiment_analysis(*args: Any, **kwargs: Any) -> SentimentAnalysis:
    """Factory function for SentimentAnalysis with dependency check."""
    _check_imports()
    return SentimentAnalysis(*args, **kwargs)


def get_pii_detection(*args: Any, **kwargs: Any) -> PIIDetection:
    """Factory function for PIIDetection with dependency check."""
    _check_imports()
    return PIIDetection(*args, **kwargs)


def get_zero_shot_classification(*args: Any, **kwargs: Any) -> ZeroShotClassification:
    """Factory function for ZeroShotClassification with dependency check."""
    _check_imports()
    return ZeroShotClassification(*args, **kwargs)


def get_toxicity(*args: Any, **kwargs: Any) -> Toxicity:
    """Factory function for Toxicity with dependency check."""
    _check_imports()
    return Toxicity(*args, **kwargs)


__all__ = [
    # Classes (ML-heavy, require [ml] extras)
    "EmbeddingsSimilarity",
    "PIIDetection",
    "SentimentAnalysis",
    "Toxicity",
    "ZeroShotClassification",
    # Classes (lightweight, base dependencies only)
    "LanguageDetection",
    # Factory functions
    "get_embeddings_similarity",
    "get_pii_detection",
    "get_sentiment_analysis",
    "get_toxicity",
    "get_zero_shot_classification",
]
