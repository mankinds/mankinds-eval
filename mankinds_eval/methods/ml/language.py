"""Language detection evaluation method."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method

# Check for optional dependency
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None  # type: ignore[assignment]
    detect_langs = None  # type: ignore[assignment]
    LangDetectException = Exception  # type: ignore[misc, assignment]


# ISO 639-1 language code to name mapping (common languages)
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "fa": "Persian",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "sw": "Swahili",
    "af": "Afrikaans",
    "cy": "Welsh",
    "tl": "Tagalog",
}


class LanguageDetection(Method):
    """Detect and verify the language of text.

    This method uses statistical language detection to identify the language
    of the output text and optionally verify it matches an expected language.

    Attributes:
        name: Method identifier.
        required_fields: Fields required in the Sample.
    """

    name: str = "LanguageDetection"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        expected_language: str | None = None,
        match_input_language: bool = False,
        confidence_threshold: float = 0.8,
        target: str = "output",
        **kwargs: Any,
    ) -> None:
        """Initialize the language detection method.

        Args:
            expected_language: Expected language code (ISO 639-1, e.g., "en", "fr").
                If None and match_input_language is False, method just reports
                detected language without pass/fail.
            match_input_language: If True, verify output language matches input language.
                Takes precedence over expected_language. Defaults to False.
            confidence_threshold: Minimum confidence for detection to be considered valid.
                Defaults to 0.8.
            target: Which field to analyze for language ("output" or "input").
                Defaults to "output".
            **kwargs: Additional configuration passed to parent.

        Raises:
            ImportError: If langdetect is not installed.
        """
        if not LANGDETECT_AVAILABLE:
            raise ImportError(
                "langdetect is required for LanguageDetection. "
                "Install it with: pip install langdetect"
            )

        super().__init__(**kwargs)

        if target not in ("output", "input"):
            raise ValueError(f"Invalid target: {target}. Must be 'output' or 'input'")

        self.expected_language = expected_language.lower() if expected_language else None
        self.match_input_language = match_input_language
        self.confidence_threshold = confidence_threshold
        self.target = target

        self.config = {
            "expected_language": expected_language,
            "match_input_language": match_input_language,
            "confidence_threshold": confidence_threshold,
            "target": target,
            **kwargs,
        }

    def _get_language_name(self, code: str) -> str:
        """Get human-readable language name from code.

        Args:
            code: ISO 639-1 language code.

        Returns:
            Human-readable language name or the code if not found.
        """
        return LANGUAGE_NAMES.get(code.lower(), code.upper())

    def _detect_language(self, text: str) -> tuple[str, float, list[dict[str, Any]]]:
        """Detect language of text.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (detected_language, confidence, top_languages).

        Raises:
            ValueError: If detection fails.
        """
        try:
            # Get all detected languages with probabilities
            langs = detect_langs(text)

            if not langs:
                raise ValueError("No language detected")

            # Top language
            top_lang = langs[0]
            detected = top_lang.lang
            confidence = top_lang.prob

            # Build top languages list
            top_languages = [
                {
                    "code": lang.lang,
                    "name": self._get_language_name(lang.lang),
                    "confidence": lang.prob,
                }
                for lang in langs[:3]
            ]

            return detected, confidence, top_languages

        except LangDetectException as e:
            raise ValueError(f"Language detection failed: {e}") from e

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate language of the target text.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with language detection results.
        """
        # Get target text
        text = sample.output if self.target == "output" else sample.input

        # Handle empty or very short text
        if not text or len(text.strip()) < 3:
            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=None,
                reason="Text too short for language detection",
                metadata={
                    "detected_language": None,
                    "detected_language_name": None,
                    "confidence": 0.0,
                    "expected_language": self.expected_language,
                    "top_languages": [],
                },
            )

        try:
            detected, confidence, top_languages = self._detect_language(text)
        except ValueError as e:
            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=None,
                reason=str(e),
                metadata={
                    "detected_language": None,
                    "detected_language_name": None,
                    "confidence": 0.0,
                    "expected_language": self.expected_language,
                    "top_languages": [],
                    "error": str(e),
                },
            )

        detected_name = self._get_language_name(detected)

        # Determine expected language
        expected = None
        if self.match_input_language:
            # Detect input language
            try:
                input_lang, _, _ = self._detect_language(sample.input)
                expected = input_lang
            except ValueError:
                expected = None
        elif self.expected_language:
            expected = self.expected_language

        # Determine passed status
        passed: bool | None = None
        if expected is not None:
            language_matches = detected.lower() == expected.lower()
            confidence_ok = confidence >= self.confidence_threshold
            passed = language_matches and confidence_ok

        # Build reason
        confidence_pct = confidence * 100
        reason = f"Detected {detected_name} ({detected}) with {confidence_pct:.1f}% confidence"

        if expected is not None:
            expected_name = self._get_language_name(expected)
            if passed:
                reason += f" (matches expected: {expected_name})"
            else:
                if detected.lower() != expected.lower():
                    reason += f" (expected: {expected_name})"
                else:
                    reason += f" (confidence below threshold: {self.confidence_threshold:.0%})"

        return MethodResult(
            method_name=self.name,
            score=confidence,
            passed=passed,
            reason=reason,
            metadata={
                "detected_language": detected,
                "detected_language_name": detected_name,
                "confidence": confidence,
                "expected_language": expected,
                "match_input_language": self.match_input_language,
                "top_languages": top_languages,
            },
        )
