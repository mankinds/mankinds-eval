"""Pairwise comparison LLM-as-Judge evaluation method."""

from __future__ import annotations

from typing import Any, Literal

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod
from mankinds_eval.utils import extract_json_object

WinnerType = Literal["A", "B", "tie"]


class PairwiseJudge(LLMMethod):
    """Compare two outputs and judge which is better.

    This method uses an LLM to compare two responses and determine which one
    is better according to a specified criterion.

    Example:
        # Compare output against expected
        judge = PairwiseJudge(
            criterion="overall quality",
            compare="output_vs_expected",
            provider="openai",
        )
        result = await judge.evaluate(sample)

        # Compare two outputs in sample metadata
        judge = PairwiseJudge(
            criterion="helpfulness",
            compare="output_a_vs_output_b",
            provider="openai",
        )
        result = await judge.evaluate(sample)  # sample needs output_a, output_b in metadata
    """

    name = "PairwiseJudge"
    required_fields = ["input", "output"]

    def __init__(
        self,
        criterion: str,
        compare: str = "output_vs_expected",
        allow_tie: bool = True,
        include_conversation: bool = True,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the pairwise judge.

        Args:
            criterion: The criterion to compare on.
            compare: Comparison mode - "output_vs_expected" or "output_a_vs_output_b".
            allow_tie: Whether to allow tie judgments.
            include_conversation: Whether to include conversation history if present.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.criterion = criterion
        self.compare = compare
        self.allow_tie = allow_tie
        self.include_conversation = include_conversation

        # Validate compare mode
        if compare not in ("output_vs_expected", "output_a_vs_output_b"):
            raise ValueError(
                f"Invalid compare mode: {compare}. "
                f"Must be 'output_vs_expected' or 'output_a_vs_output_b'"
            )

    def _get_texts_to_compare(self, sample: Sample) -> tuple[str, str]:
        """Get the two texts to compare based on comparison mode.

        Args:
            sample: The sample containing the texts.

        Returns:
            Tuple of (text_a, text_b).

        Raises:
            ValueError: If required texts are not available.
        """
        if self.compare == "output_vs_expected":
            if sample.expected is None:
                raise ValueError(
                    "Sample must have 'expected' field for output_vs_expected comparison"
                )
            return sample.output, sample.expected
        else:  # output_a_vs_output_b
            metadata = sample.metadata or {}
            output_a = metadata.get("output_a")
            output_b = metadata.get("output_b")
            if output_a is None or output_b is None:
                raise ValueError(
                    "Sample metadata must have 'output_a' and 'output_b' "
                    "for output_a_vs_output_b comparison"
                )
            return str(output_a), str(output_b)

    def _build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build messages for LLM comparison.

        Args:
            sample: The sample to evaluate.

        Returns:
            List of message dicts for the LLM.
        """
        text_a, text_b = self._get_texts_to_compare(sample)

        # Build choice instructions
        if self.allow_tie:
            choice_instruction = 'Choose "A", "B", or "tie"'
            json_format = '{"winner": "A" or "B" or "tie", "reason": "your explanation"}'
        else:
            choice_instruction = 'Choose "A" or "B" (you must pick one)'
            json_format = '{"winner": "A" or "B", "reason": "your explanation"}'

        # System prompt
        system_content = (
            f"You are an expert evaluator. Your task is to compare two AI responses "
            f"and determine which is better based on the criterion: {self.criterion}.\n\n"
            f"{choice_instruction}.\n\n"
            f"You MUST respond with valid JSON in this exact format:\n"
            f"{json_format}\n\n"
            f"Do not include any text outside the JSON object."
        )

        # Build user message content
        user_parts: list[str] = []

        # Include conversation history if present and enabled
        if self.include_conversation and sample.conversation:
            user_parts.append("## Conversation History")
            for msg in sample.conversation:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                user_parts.append(f"{role.capitalize()}: {content}")
            user_parts.append("")

        # Include context if present
        if sample.context:
            user_parts.append("## Context")
            user_parts.append(sample.context)
            user_parts.append("")

        # Include input
        user_parts.append("## User Input")
        user_parts.append(sample.input)
        user_parts.append("")

        # Include the two responses
        user_parts.append("## Response A")
        user_parts.append(text_a)
        user_parts.append("")
        user_parts.append("## Response B")
        user_parts.append(text_b)
        user_parts.append("")

        # Add comparison instruction
        user_parts.append("## Comparison Task")
        user_parts.append(
            f"Compare Response A and Response B based on: {self.criterion}\n"
            f"{choice_instruction}.\n"
            f"Respond with JSON: {json_format}"
        )

        user_content = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> tuple[WinnerType, str]:
        """Parse LLM response to extract winner and reason.

        Args:
            response: Raw LLM response string.

        Returns:
            Tuple of (winner, reason).

        Raises:
            ValueError: If response cannot be parsed.
        """
        data = extract_json_object(response)

        # Extract winner
        raw_winner = data.get("winner")
        if raw_winner is None:
            raise ValueError(f"No winner found in response: {data}")

        winner_str = str(raw_winner).upper().strip()

        if winner_str == "A":
            winner: WinnerType = "A"
        elif winner_str == "B":
            winner = "B"
        elif winner_str == "TIE" and self.allow_tie:
            winner = "tie"
        else:
            if not self.allow_tie and winner_str == "TIE":
                raise ValueError("Tie not allowed in this comparison")
            raise ValueError(f"Invalid winner value: {raw_winner}")

        # Extract reason
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)

        return winner, reason

    def _winner_to_score(self, winner: WinnerType) -> float:
        """Convert winner to numeric score.

        For output_vs_expected:
            - A (output) wins: 1.0 (output is better than expected)
            - tie: 0.5
            - B (expected) wins: 0.0 (output is worse than expected)

        For output_a_vs_output_b:
            - A wins: 1.0
            - tie: 0.5
            - B wins: 0.0

        Args:
            winner: The winner determination.

        Returns:
            Numeric score.
        """
        if winner == "A":
            return 1.0
        elif winner == "tie":
            return 0.5
        else:  # B
            return 0.0

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate a sample using pairwise comparison.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with comparison results.
        """
        messages = self._build_prompt(sample)
        response = await self.llm.complete(messages)
        winner, reason = self._parse_response(response)

        score = self._winner_to_score(winner)

        # Build metadata
        text_a, text_b = self._get_texts_to_compare(sample)
        metadata: dict[str, Any] = {
            "winner": winner,
            "criterion": self.criterion,
            "compare_mode": self.compare,
            "allow_tie": self.allow_tie,
            "provider": self.llm.provider,
            "model": self.llm.model,
        }

        # Add text labels based on comparison mode
        if self.compare == "output_vs_expected":
            metadata["text_a_label"] = "output"
            metadata["text_b_label"] = "expected"
        else:
            metadata["text_a_label"] = "output_a"
            metadata["text_b_label"] = "output_b"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=None,  # Pairwise comparison doesn't have a natural pass/fail
            reason=reason,
            metadata=metadata,
        )
