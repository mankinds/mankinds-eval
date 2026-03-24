"""Evaluation methods.

This module provides access to all evaluation method classes. Methods can be
imported directly from this module or from their respective submodules.

Example:
    >>> from mankinds_eval.methods import Method, ExactMatch, FuzzyMatch
    >>> from mankinds_eval.methods.heuristic import BLEU, ROUGE, JSONValid
    >>> from mankinds_eval.methods.llm import SingleCriterionJudge, Faithfulness
    >>> from mankinds_eval.methods.presets import RAGScorer, SafetyScorer
"""

from mankinds_eval.methods.base import Method

# Composite methods
from mankinds_eval.methods.composite import (
    CompositeMethod,
    ConditionalMethod,
)

# Re-export heuristic methods
from mankinds_eval.methods.heuristic import (
    BLEU,
    ROUGE,
    ContainsAll,
    ContainsAny,
    ExactMatch,
    FuzzyMatch,
    JSONSchema,
    JSONValid,
    NoRefusal,
    RegexMatch,
    SentenceCount,
    TextLength,
    WordCount,
)

# Re-export LLM methods
from mankinds_eval.methods.llm import (
    LITELLM_AVAILABLE,
    AnswerRelevancy,
    Coherence,
    ConsensusJudge,
    Correctness,
    Faithfulness,
    Helpfulness,
    LLMMethod,
    LLMProvider,
    MultiCriteriaJudge,
    PairwiseJudge,
    SingleCriterionJudge,
)

# Re-export ML methods (may not be available without [ml] extras)
from mankinds_eval.methods.ml import (
    EmbeddingsSimilarity,
    LanguageDetection,
    PIIDetection,
    SentimentAnalysis,
    Toxicity,
    ZeroShotClassification,
    get_embeddings_similarity,
    get_pii_detection,
    get_sentiment_analysis,
    get_toxicity,
    get_zero_shot_classification,
)

# Re-export presets
from mankinds_eval.methods.presets import (
    RAGScorer,
    SafetyScorer,
)

__all__ = [
    # Base
    "Method",
    # Heuristic methods
    "BLEU",
    "ContainsAll",
    "ContainsAny",
    "ExactMatch",
    "FuzzyMatch",
    "JSONSchema",
    "JSONValid",
    "NoRefusal",
    "RegexMatch",
    "ROUGE",
    "SentenceCount",
    "TextLength",
    "WordCount",
    # LLM methods - Pre-built judges
    "AnswerRelevancy",
    "Coherence",
    "Correctness",
    "Faithfulness",
    "Helpfulness",
    # LLM methods - Configurable judges
    "ConsensusJudge",
    "MultiCriteriaJudge",
    "PairwiseJudge",
    "SingleCriterionJudge",
    # LLM utilities
    "LITELLM_AVAILABLE",
    "LLMMethod",
    "LLMProvider",
    # ML methods
    "EmbeddingsSimilarity",
    "LanguageDetection",
    "PIIDetection",
    "SentimentAnalysis",
    "Toxicity",
    "ZeroShotClassification",
    # ML factory functions
    "get_embeddings_similarity",
    "get_pii_detection",
    "get_sentiment_analysis",
    "get_toxicity",
    "get_zero_shot_classification",
    # Presets
    "RAGScorer",
    "SafetyScorer",
    # Composite methods
    "CompositeMethod",
    "ConditionalMethod",
]
