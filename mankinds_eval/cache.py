"""Simple caching for evaluation results."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mankinds_eval.core import MethodResult, Sample


def _compute_sample_hash(sample: Sample, method_config: dict[str, Any]) -> str:
    """Compute a hash for a sample and method configuration.

    Args:
        sample: The sample to hash.
        method_config: The method configuration dict.

    Returns:
        SHA256 hash string.
    """
    data = {
        "input": sample.input,
        "output": sample.output,
        "expected": sample.expected,
        "context": sample.context,
        "method_config": method_config,
    }
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


class ResultCache:
    """Simple file-based cache for evaluation results.

    Caches method results based on sample content and method configuration.
    This avoids re-running expensive evaluations (especially LLM calls) for
    unchanged samples.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to .mankinds_cache
            enabled: If False, cache is disabled (always miss).
        """
        self.enabled = enabled
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / ".mankinds_cache"

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            cache_key: The cache key (hash).

        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        sample: Sample,
        method_config: dict[str, Any],
    ) -> MethodResult | None:
        """Get a cached result if available.

        Args:
            sample: The sample being evaluated.
            method_config: The method configuration dict.

        Returns:
            Cached MethodResult or None if not found.
        """
        if not self.enabled:
            return None

        cache_key = _compute_sample_hash(sample, method_config)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return MethodResult(
                method_name=data["method_name"],
                score=data.get("score"),
                passed=data.get("passed"),
                reason=data.get("reason"),
                metadata=data.get("metadata"),
                error=data.get("error"),
            )
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(
        self,
        sample: Sample,
        method_config: dict[str, Any],
        result: MethodResult,
    ) -> None:
        """Store a result in the cache.

        Args:
            sample: The sample that was evaluated.
            method_config: The method configuration dict.
            result: The result to cache.
        """
        if not self.enabled:
            return

        cache_key = _compute_sample_hash(sample, method_config)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f)
        except OSError:
            pass  # Silently ignore cache write failures

    def clear(self) -> int:
        """Clear all cached results.

        Returns:
            Number of cache entries removed.
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        return count

    def size(self) -> int:
        """Get the number of cached entries.

        Returns:
            Number of cache files.
        """
        if not self.cache_dir.exists():
            return 0
        return len(list(self.cache_dir.glob("*.json")))


# Global cache instance
_default_cache: ResultCache | None = None


def get_cache() -> ResultCache:
    """Get the default cache instance.

    Returns:
        The default ResultCache instance.
    """
    global _default_cache
    if _default_cache is None:
        # Check environment variable to enable/disable
        enabled = os.environ.get("MANKINDS_CACHE_ENABLED", "0") == "1"
        _default_cache = ResultCache(enabled=enabled)
    return _default_cache


def set_cache(cache: ResultCache) -> None:
    """Set the default cache instance.

    Args:
        cache: The cache instance to use as default.
    """
    global _default_cache
    _default_cache = cache
