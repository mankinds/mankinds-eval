"""LLM-based evaluation methods."""

from mankinds_eval.methods.llm.base import LLMMethod
from mankinds_eval.methods.llm.coherence import Coherence
from mankinds_eval.methods.llm.consensus import ConsensusJudge
from mankinds_eval.methods.llm.correctness import Correctness
from mankinds_eval.methods.llm.faithfulness import Faithfulness
from mankinds_eval.methods.llm.helpfulness import Helpfulness
from mankinds_eval.methods.llm.multi import MultiCriteriaJudge
from mankinds_eval.methods.llm.pairwise import PairwiseJudge
from mankinds_eval.methods.llm.providers import LITELLM_AVAILABLE, LLMProvider
from mankinds_eval.methods.llm.relevancy import AnswerRelevancy
from mankinds_eval.methods.llm.single import SingleCriterionJudge

__all__ = [
    # Pre-built judges
    "AnswerRelevancy",
    "Coherence",
    "Correctness",
    "Faithfulness",
    "Helpfulness",
    # Configurable judges
    "ConsensusJudge",
    "MultiCriteriaJudge",
    "PairwiseJudge",
    "SingleCriterionJudge",
    # Base classes and utilities
    "LITELLM_AVAILABLE",
    "LLMMethod",
    "LLMProvider",
]
