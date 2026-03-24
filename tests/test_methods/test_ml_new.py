"""Tests for new ML evaluation methods (Toxicity, LanguageDetection)."""

from __future__ import annotations

import pytest

from mankinds_eval.core import Sample

# LanguageDetection uses langdetect (base dependency), should always be available
from mankinds_eval.methods.ml.language import LanguageDetection

# Toxicity requires transformers, may not be available
try:
    from mankinds_eval.methods.ml.toxicity import TRANSFORMERS_AVAILABLE, Toxicity
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    Toxicity = None  # type: ignore[misc, assignment]


# =============================================================================
# LanguageDetection Tests
# =============================================================================


class TestLanguageDetection:
    """Tests for the LanguageDetection method."""

    @pytest.mark.asyncio
    async def test_detect_english(self) -> None:
        """Test detection of English text."""
        method = LanguageDetection()
        sample = Sample(
            input="test",
            output="This is a sample text in English for testing purposes.",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert result.score > 0.5
        assert result.metadata is not None
        assert result.metadata["detected_language"] == "en"
        assert result.metadata["detected_language_name"] == "English"

    @pytest.mark.asyncio
    async def test_detect_french(self) -> None:
        """Test detection of French text."""
        method = LanguageDetection()
        sample = Sample(
            input="test",
            output="Ceci est un texte en francais pour tester la detection.",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert result.metadata["detected_language"] == "fr"

    @pytest.mark.asyncio
    async def test_expected_language_match(self) -> None:
        """Test with expected language that matches."""
        method = LanguageDetection(expected_language="en")
        sample = Sample(
            input="test",
            output="This is English text that should match the expected language.",
        )
        result = await method.evaluate(sample)

        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["detected_language"] == "en"

    @pytest.mark.asyncio
    async def test_expected_language_mismatch(self) -> None:
        """Test with expected language that doesn't match."""
        method = LanguageDetection(expected_language="fr")
        sample = Sample(
            input="test",
            output="This is clearly English text, not French.",
        )
        result = await method.evaluate(sample)

        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["detected_language"] == "en"
        assert result.metadata["expected_language"] == "fr"

    @pytest.mark.asyncio
    async def test_match_input_language(self) -> None:
        """Test matching output language to input language."""
        method = LanguageDetection(match_input_language=True)
        # Use longer text for more reliable language detection
        sample = Sample(
            input=(
                "Bonjour, comment allez-vous aujourd'hui?"
                " J'espère que vous passez une bonne journée."
            ),
            output="Je vais très bien, merci beaucoup de demander. C'est une journée magnifique.",
        )
        result = await method.evaluate(sample)

        assert result.passed is True
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_match_input_language_mismatch(self) -> None:
        """Test when output language doesn't match input."""
        method = LanguageDetection(match_input_language=True)
        sample = Sample(
            input="Bonjour, comment allez-vous?",
            output="I am doing well, thank you for asking.",
        )
        result = await method.evaluate(sample)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_confidence_threshold(self) -> None:
        """Test confidence threshold filtering."""
        method = LanguageDetection(expected_language="en", confidence_threshold=0.99)
        sample = Sample(
            input="test",
            output="Hello",  # Very short, may have lower confidence
        )
        result = await method.evaluate(sample)

        # May or may not pass depending on confidence
        assert result.score is not None
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_short_text_handling(self) -> None:
        """Test handling of very short text."""
        method = LanguageDetection()
        sample = Sample(
            input="test",
            output="Hi",  # Very short
        )
        result = await method.evaluate(sample)

        # Should handle gracefully
        assert result.score is not None or result.passed is None

    @pytest.mark.asyncio
    async def test_very_short_text_handling(self) -> None:
        """Test handling of very short text (1-2 chars)."""
        method = LanguageDetection()
        # Sample requires non-empty output, so test with minimal valid text
        sample = Sample(
            input="test",
            output="ab",  # Very short - may fail detection
        )
        result = await method.evaluate(sample)

        # Should handle gracefully - either detect with low confidence or report too short
        assert result.score is not None or result.passed is None

    @pytest.mark.asyncio
    async def test_top_languages_in_metadata(self) -> None:
        """Test that top languages are included in metadata."""
        method = LanguageDetection()
        sample = Sample(
            input="test",
            output="This is a longer English text to ensure confident detection.",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert "top_languages" in result.metadata
        assert isinstance(result.metadata["top_languages"], list)

    def test_invalid_target(self) -> None:
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError):
            LanguageDetection(target="invalid")


# =============================================================================
# Toxicity Tests (requires [ml] extras)
# =============================================================================


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestToxicity:
    """Tests for the Toxicity method.

    Note: These tests require the [ml] extras to be installed and will be skipped
    if transformers is not available. The tests use the actual model which may be
    slow on first run due to model download.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_non_toxic_text(self) -> None:
        """Test non-toxic text returns high score."""
        method = Toxicity()
        sample = Sample(
            input="test",
            output="This is a friendly and helpful response about programming.",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        # Score is inverted: higher = less toxic
        assert result.score > 0.5
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["is_toxic"] is False

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_toxic_text(self) -> None:
        """Test toxic text returns low score."""
        method = Toxicity(threshold=0.5)
        sample = Sample(
            input="test",
            output="You are an idiot and I hate you!",  # Obviously toxic
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        # Score is inverted: lower = more toxic
        assert result.metadata is not None
        # The model should detect this as toxic
        assert result.metadata["toxicity_score"] > 0.3

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_threshold_configuration(self) -> None:
        """Test toxicity threshold configuration."""
        method = Toxicity(threshold=0.9)  # Very strict
        sample = Sample(
            input="test",
            output="This is mildly negative feedback.",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert result.metadata["threshold"] == 0.9

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_target_input(self) -> None:
        """Test analyzing input instead of output."""
        method = Toxicity(target="input")
        sample = Sample(
            input="This is a normal question about coding.",
            output="Some response",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert result.metadata["target"] == "input"

    def test_invalid_target(self) -> None:
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError):
            Toxicity(target="invalid")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_metadata_contains_scores(self) -> None:
        """Test that metadata contains all expected fields."""
        method = Toxicity()
        sample = Sample(
            input="test",
            output="A normal helpful response.",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert "toxicity_score" in result.metadata
        assert "is_toxic" in result.metadata
        assert "raw_label" in result.metadata
        assert "model" in result.metadata
