"""Base class for LLM-based evaluation methods."""

from __future__ import annotations

from typing import Any

from mankinds_eval.methods.base import Method
from mankinds_eval.methods.llm.providers import LLMProvider


class LLMMethod(Method):
    """Base class for LLM-based evaluation methods.

    This class extends Method to provide LLM capabilities through the LLMProvider.
    Subclasses should implement the evaluate() method using self.llm for LLM calls.

    Example:
        class MyLLMMethod(LLMMethod):
            name = "my_llm_method"

            async def evaluate(self, sample: Sample) -> MethodResult:
                response = await self.llm.complete([
                    {"role": "user", "content": f"Evaluate: {sample.output}"}
                ])
                # Parse response and return MethodResult
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_concurrent: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM method.

        Args:
            provider: LLM provider name (e.g., "openai", "anthropic", "mistral").
            model: Model name. Defaults based on provider if not specified.
            api_key: API key. Falls back to environment variables if not provided.
            api_base: Custom API base URL (useful for ollama, local deployments).
            temperature: Sampling temperature. Defaults to 0.0 for deterministic output.
            max_tokens: Maximum tokens in response. Defaults to 1024.
            max_concurrent: Method-specific concurrency limit for batch evaluation.
            **kwargs: Additional parameters passed to parent Method.
        """
        super().__init__(**kwargs)
        self.llm = LLMProvider(
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.max_concurrent = max_concurrent
