"""Structured output evaluation methods."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method

# Check for optional jsonschema dependency
try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None  # type: ignore[assignment]


class JSONValid(Method):
    """Check if output is valid JSON.

    This method validates that the output string can be parsed as valid JSON.
    Optionally extracts JSON from markdown code blocks.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "JSONValid"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    # Pattern to extract JSON from markdown code blocks
    _MARKDOWN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL
    )

    def __init__(
        self,
        extract_from_markdown: bool = True,
        allow_trailing_comma: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize JSONValid method.

        Args:
            extract_from_markdown: Whether to extract JSON from markdown code blocks.
                Defaults to True.
            allow_trailing_comma: Whether to allow trailing commas (non-standard JSON).
                Defaults to False.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.extract_from_markdown = extract_from_markdown
        self.allow_trailing_comma = allow_trailing_comma
        self.config.update(
            {
                "extract_from_markdown": extract_from_markdown,
                "allow_trailing_comma": allow_trailing_comma,
            }
        )

    def _extract_json(self, text: str) -> tuple[str, bool]:
        """Extract JSON content from text.

        Args:
            text: The text to extract JSON from.

        Returns:
            Tuple of (extracted_text, was_extracted_from_markdown).
        """
        if self.extract_from_markdown:
            match = self._MARKDOWN_PATTERN.search(text)
            if match:
                return match.group(1).strip(), True
        return text.strip(), False

    def _remove_trailing_commas(self, text: str) -> str:
        """Remove trailing commas from JSON-like text.

        Args:
            text: The JSON text with potential trailing commas.

        Returns:
            Text with trailing commas removed.
        """
        # Remove trailing commas before ] or }
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output is valid JSON.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if valid JSON, 0.0 otherwise.
        """
        text, extracted_from_markdown = self._extract_json(sample.output)

        if self.allow_trailing_comma:
            text = self._remove_trailing_commas(text)

        try:
            parsed = json.loads(text)
            return MethodResult(
                method_name=self.name,
                score=1.0,
                passed=True,
                reason="Valid JSON",
                metadata={
                    "parsed": True,
                    "extracted_from_markdown": extracted_from_markdown,
                    "json_type": type(parsed).__name__,
                },
            )
        except json.JSONDecodeError as e:
            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=False,
                reason=f"Invalid JSON: {e.msg}",
                metadata={
                    "parsed": False,
                    "error_position": e.pos,
                    "error_line": e.lineno,
                    "error_column": e.colno,
                    "extracted_from_markdown": extracted_from_markdown,
                },
            )


class JSONSchema(Method):
    """Validate output against a JSON Schema.

    This method validates that the output JSON conforms to a specified
    JSON Schema definition.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "JSONSchema"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    # Pattern to extract JSON from markdown code blocks
    _MARKDOWN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL
    )

    def __init__(
        self,
        schema: dict[str, Any],
        extract_from_markdown: bool = True,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize JSONSchema method.

        Args:
            schema: The JSON Schema to validate against.
            extract_from_markdown: Whether to extract JSON from markdown code blocks.
                Defaults to True.
            strict: Whether to fail on additional properties not in schema.
                Defaults to True.
            **kwargs: Additional configuration options.

        Raises:
            ImportError: If jsonschema is not installed.
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema is required for JSONSchema. Install it with: pip install jsonschema"
            )

        super().__init__(**kwargs)
        self.schema = schema
        self.extract_from_markdown = extract_from_markdown
        self.strict = strict

        # Apply strict mode by adding additionalProperties: false
        if strict and schema.get("type") == "object":
            self._effective_schema = {**schema, "additionalProperties": False}
        else:
            self._effective_schema = schema

        self.config.update(
            {
                "schema": schema,
                "extract_from_markdown": extract_from_markdown,
                "strict": strict,
            }
        )

    def _extract_json(self, text: str) -> tuple[str, bool]:
        """Extract JSON content from text.

        Args:
            text: The text to extract JSON from.

        Returns:
            Tuple of (extracted_text, was_extracted_from_markdown).
        """
        if self.extract_from_markdown:
            match = self._MARKDOWN_PATTERN.search(text)
            if match:
                return match.group(1).strip(), True
        return text.strip(), False

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output conforms to JSON Schema.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if valid, 0.0 otherwise.
        """
        text, extracted_from_markdown = self._extract_json(sample.output)

        # First, try to parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=False,
                reason=f"Invalid JSON: {e.msg}",
                metadata={
                    "valid": False,
                    "errors": [f"JSON parse error: {e.msg}"],
                    "error_paths": [],
                    "extracted_from_markdown": extracted_from_markdown,
                },
            )

        # Validate against schema
        validator = jsonschema.Draft7Validator(self._effective_schema)
        errors = list(validator.iter_errors(data))

        if not errors:
            return MethodResult(
                method_name=self.name,
                score=1.0,
                passed=True,
                reason="Schema valid",
                metadata={
                    "valid": True,
                    "errors": [],
                    "error_paths": [],
                    "extracted_from_markdown": extracted_from_markdown,
                },
            )
        else:
            error_messages = [e.message for e in errors[:5]]  # Limit to 5 errors
            error_paths = [".".join(str(p) for p in e.absolute_path) for e in errors[:5]]

            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=False,
                reason=f"Schema validation failed: {len(errors)} error(s)",
                metadata={
                    "valid": False,
                    "errors": error_messages,
                    "error_paths": error_paths,
                    "total_errors": len(errors),
                    "extracted_from_markdown": extracted_from_markdown,
                },
            )


class NoRefusal(Method):
    """Detect if LLM refused to answer.

    This method checks if the output contains common refusal patterns,
    indicating the LLM declined to respond to the request.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "NoRefusal"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    # Default refusal patterns (multi-language)
    DEFAULT_PATTERNS: ClassVar[list[str]] = [
        # English patterns
        r"(?i)\bi('m| am) (sorry|afraid|unable)",
        r"(?i)\bi can('t|not) (help|assist|provide|do that|comply)",
        r"(?i)\bas an ai",
        r"(?i)\bi('m| am) not able to",
        r"(?i)\bi('m| am) designed to",
        r"(?i)\bi must (decline|refuse)",
        r"(?i)\bi will not",
        r"(?i)\bthis (is|goes) (beyond|against)",
        r"(?i)\bmy (programming|guidelines|policies)",
        r"(?i)\bethical (guidelines|concerns|boundaries)",
        # French patterns
        r"(?i)\bje (ne peux pas|suis dans l'impossibilit)",
        r"(?i)\bje suis d[eé]sol[eé]",
        r"(?i)\bil m'est impossible",
        r"(?i)\bje ne suis pas en mesure",
        # German patterns
        r"(?i)\bes tut mir leid",
        r"(?i)\bich kann (nicht|leider nicht)",
        # Spanish patterns
        r"(?i)\blo siento",
        r"(?i)\bno puedo (ayudar|proporcionar)",
    ]

    def __init__(
        self,
        patterns: list[str] | None = None,
        extend_default: bool = True,
        case_sensitive: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize NoRefusal method.

        Args:
            patterns: Custom regex patterns to detect refusals.
                If None and extend_default is True, uses only default patterns.
            extend_default: Whether to extend default patterns with custom ones.
                If False, only custom patterns are used. Defaults to True.
            case_sensitive: Whether pattern matching is case sensitive.
                Defaults to False (case insensitive).
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.case_sensitive = case_sensitive

        # Build pattern list
        if patterns is None:
            self._patterns = self.DEFAULT_PATTERNS.copy()
        elif extend_default:
            self._patterns = self.DEFAULT_PATTERNS + patterns
        else:
            self._patterns = patterns

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self._compiled_patterns = [re.compile(p, flags) for p in self._patterns]

        self.config.update(
            {
                "patterns": patterns,
                "extend_default": extend_default,
                "case_sensitive": case_sensitive,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output contains refusal patterns.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if no refusal, 0.0 if refusal detected.
        """
        text = sample.output

        matched_patterns: list[str] = []
        matched_texts: list[str] = []

        for pattern, compiled in zip(self._patterns, self._compiled_patterns):
            match = compiled.search(text)
            if match:
                matched_patterns.append(pattern)
                matched_texts.append(match.group(0))

        is_refusal = len(matched_patterns) > 0

        if is_refusal:
            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=False,
                reason=f"Refusal detected: '{matched_texts[0]}'",
                metadata={
                    "is_refusal": True,
                    "matched_patterns": matched_patterns,
                    "matched_texts": matched_texts,
                },
            )
        else:
            return MethodResult(
                method_name=self.name,
                score=1.0,
                passed=True,
                reason="No refusal detected",
                metadata={
                    "is_refusal": False,
                    "matched_patterns": [],
                    "matched_texts": [],
                },
            )
