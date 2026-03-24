"""LLM provider abstraction using litellm."""

from __future__ import annotations

import os
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# litellm import with graceful handling
try:
    import litellm
    from litellm.exceptions import (
        APIConnectionError,
        APIError,
        RateLimitError,
        ServiceUnavailableError,
    )

    LITELLM_AVAILABLE = True
    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RateLimitError,
        APIConnectionError,
        APIError,
        ServiceUnavailableError,
    )
except ImportError:
    litellm = None  # type: ignore[assignment]
    LITELLM_AVAILABLE = False
    RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError)


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_WAIT = 1
DEFAULT_MAX_WAIT = 60


class LLMProvider:
    """Wrapper around litellm for LLM calls.

    This class provides a unified interface to various LLM providers through litellm.
    It handles API key management, model string formatting, and provides both
    async and sync completion methods.

    Example:
        provider = LLMProvider(provider="openai", model="gpt-4o-mini")
        response = await provider.complete([
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_min_wait: float = DEFAULT_MIN_WAIT,
        retry_max_wait: float = DEFAULT_MAX_WAIT,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic", "mistral", "ollama").
            model: Model name. Defaults based on provider if not specified.
            api_key: API key. Falls back to environment variables if not provided.
            api_base: Custom API base URL (useful for ollama, local deployments).
            temperature: Sampling temperature. Defaults to 0.0 for deterministic output.
            max_tokens: Maximum tokens in response. Defaults to 1024.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_min_wait: Minimum wait time between retries in seconds.
            retry_max_wait: Maximum wait time between retries in seconds.
            **kwargs: Additional parameters passed to litellm.
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LLM methods. Install with: pip install mankinds-eval[llm]"
            )
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.extra_params: dict[str, Any] = kwargs

        # Set API key in environment if provided explicitly
        self._setup_api_key()

    def _default_model(self, provider: str) -> str:
        """Return default model for each provider.

        Args:
            provider: The provider name.

        Returns:
            Default model name for the provider.
        """
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "mistral": "mistral-small-latest",
            "ollama": "llama3.2",
            "groq": "llama-3.1-8b-instant",
        }
        return defaults.get(provider, provider)

    def _setup_api_key(self) -> None:
        """Set API key in environment if provided."""
        if self.api_key:
            key_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "mistral": "MISTRAL_API_KEY",
            }
            env_var = key_map.get(self.provider)
            if env_var:
                os.environ[env_var] = self.api_key

    def _get_model_string(self) -> str:
        """Get litellm model string.

        litellm uses format: provider/model for some providers.

        Returns:
            Model string formatted for litellm.
        """
        if self.provider == "ollama":
            return f"ollama/{self.model}"
        if self.provider == "anthropic":
            return f"anthropic/{self.model}"
        if self.provider == "mistral":
            return f"mistral/{self.model}"
        # OpenAI doesn't need prefix
        return self.model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send completion request and return response text.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                Role can be "user", "assistant", or "system".
            temperature: Override default temperature for this call.
            max_tokens: Override default max_tokens for this call.
            **kwargs: Additional parameters passed to litellm.

        Returns:
            Response content as string.

        Raises:
            Exception: If all retry attempts fail.
        """
        # Create retry decorator with instance settings
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                min=self.retry_min_wait,
                max=self.retry_max_wait,
            ),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        )

        @retry_decorator
        async def _complete_with_retry() -> str:
            response = await litellm.acompletion(
                model=self._get_model_string(),
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                api_base=self.api_base,
                **self.extra_params,
                **kwargs,
            )
            return str(response.choices[0].message.content)

        return await _complete_with_retry()

    def complete_sync(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Synchronous version of complete.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Additional parameters passed to complete().

        Returns:
            Response content as string.
        """
        import asyncio

        return asyncio.run(self.complete(messages, **kwargs))
