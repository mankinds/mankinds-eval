"""Base generator class."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from mankinds_eval.core import Sample


class Generator(ABC):
    """Base class for test generators."""

    name: str = "base_generator"

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    async def generate(self, base_prompt: str, n_variants: int = 10) -> list[Sample]:
        """Generate sample variants from a base prompt."""
        pass

    def generate_sync(self, base_prompt: str, n_variants: int = 10) -> list[Sample]:
        """Synchronous wrapper."""
        return asyncio.run(self.generate(base_prompt, n_variants))
