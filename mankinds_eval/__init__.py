"""mankinds-eval: Open source Python library providing evaluation methods for AI systems."""

from importlib.metadata import PackageNotFoundError, version

# Define __version__ BEFORE other imports to avoid circular import issues
try:
    __version__ = version("mankinds-eval")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for development

from mankinds_eval.cache import ResultCache, get_cache, set_cache
from mankinds_eval.callbacks import (
    Callback,
    CallbackManager,
    EvaluationEvent,
    LoggingCallback,
    ProgressCallback,
)
from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.data import load_samples
from mankinds_eval.methods import Method
from mankinds_eval.output import EvaluationResult
from mankinds_eval.scorer import Scorer

__all__ = [
    # Core
    "EvaluationResult",
    "Method",
    "MethodResult",
    "Sample",
    "Scorer",
    "load_samples",
    # Callbacks
    "Callback",
    "CallbackManager",
    "EvaluationEvent",
    "LoggingCallback",
    "ProgressCallback",
    # Cache
    "ResultCache",
    "get_cache",
    "set_cache",
    # Version
    "__version__",
]
