"""LLM provider abstraction using litellm."""

from __future__ import annotations

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
    The implementation stays generic: model strings follow the litellm convention
    ``<provider>/<model>`` (``openai`` being the implicit default), the API key is
    forwarded directly to ``litellm.acompletion``, and any provider-specific auth
    parameter (eg. ``vertex_credentials`` for Google Vertex AI, ``aws_region_name``
    for Bedrock, ``api_version`` for Azure) is passed through ``provider_kwargs``.

    Examples:
        Basic OpenAI / Anthropic / Mistral usage:

            provider = LLMProvider(provider="openai", model="gpt-4o-mini",
                                   api_key="sk-...")
            await provider.complete([{"role": "user", "content": "Hello!"}])

        Google Vertex AI (service-account auth):

            provider = LLMProvider(
                provider="vertex_ai",
                model="gemini-2.5-flash",
                provider_kwargs={
                    "vertex_credentials": '{"type":"service_account",...}',
                    "vertex_project": "my-gcp-project",
                    "vertex_location": "global",
                },
            )

        AWS Bedrock:

            provider = LLMProvider(
                provider="bedrock",
                model="anthropic.claude-3-opus-20240229-v1:0",
                provider_kwargs={"aws_region_name": "us-east-1"},
            )
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
        provider_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM provider.

        Args:
            provider: Provider name (any litellm-supported provider:
                "openai", "anthropic", "mistral", "ollama", "groq", "gemini",
                "vertex_ai", "bedrock", "azure", "cohere", ...).
            model: Model name. Defaults based on provider if not specified.
                May be passed pre-prefixed (``"vertex_ai/gemini-2.5-flash"``) or
                bare (``"gemini-2.5-flash"``) — the prefix is added automatically
                when missing, following the litellm convention.
            api_key: API key for providers that authenticate via a single token
                (openai, anthropic, mistral, groq, cohere, gemini AI Studio…).
                Falls back to the provider-specific environment variable that
                litellm reads natively when ``None``. For cloud providers that
                use service accounts / IAM (vertex_ai, bedrock), leave this
                ``None`` and pass the auth payload via ``provider_kwargs``.
            api_base: Custom API base URL (useful for ollama, on-prem deployments,
                Azure, OpenAI-compatible endpoints).
            temperature: Sampling temperature. Defaults to 0.0 for deterministic output.
            max_tokens: Maximum tokens in response. Defaults to 1024.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_min_wait: Minimum wait time between retries in seconds.
            retry_max_wait: Maximum wait time between retries in seconds.
            provider_kwargs: Opaque dict forwarded verbatim to ``litellm.acompletion``.
                Use it for provider-specific auth or routing parameters that don't fit
                into ``api_key`` / ``api_base`` (eg. ``vertex_credentials``,
                ``vertex_project``, ``vertex_location``, ``aws_region_name``,
                ``api_version`` for Azure). LiteLLM is the source of truth for the
                accepted keys.
            **kwargs: Additional parameters passed to litellm at call time. Kept
                for backward compatibility — new code should prefer ``provider_kwargs``
                for provider-specific arguments.
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
        self.provider_kwargs: dict[str, Any] = provider_kwargs or {}
        self.extra_params: dict[str, Any] = kwargs

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

    def _get_model_string(self) -> str:
        """Return the litellm-compatible model string.

        LiteLLM uses ``<provider>/<model>`` for every provider except ``openai``
        (which is the implicit default and works without prefix). When the model
        already contains ``/`` we assume the caller has pre-formatted it and we
        pass it through untouched. This keeps the wrapper generic: any future
        litellm provider works without code change.
        """
        if self.model and "/" in self.model:
            return self.model
        if self.provider == "openai":
            return self.model
        return f"{self.provider}/{self.model}"

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
                api_key=self.api_key,
                api_base=self.api_base,
                **self.provider_kwargs,
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
