"""Tests for preset scorer combinations."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mankinds_eval.methods.presets.rag import RAGScorer
from mankinds_eval.methods.presets.safety import SafetyScorer

# =============================================================================
# RAGScorer Tests
# =============================================================================


class TestRAGScorer:
    """Tests for the RAGScorer preset."""

    @patch("mankinds_eval.methods.llm.base.LLMProvider")
    def test_create_default(self, mock_provider) -> None:
        """Test creating RAGScorer with defaults."""
        methods = RAGScorer.create(provider="openai")

        assert len(methods) == 3
        method_names = [m.name for m in methods]
        assert "Faithfulness" in method_names
        assert "AnswerRelevancy" in method_names
        assert "Coherence" in method_names

    @patch("mankinds_eval.methods.llm.base.LLMProvider")
    def test_create_custom_threshold(self, mock_provider) -> None:
        """Test creating RAGScorer with custom threshold."""
        methods = RAGScorer.create(provider="openai", threshold=0.8)

        for method in methods:
            assert method.threshold == 0.8

    @patch("mankinds_eval.methods.llm.base.LLMProvider")
    def test_exclude_methods(self, mock_provider) -> None:
        """Test excluding specific methods."""
        methods = RAGScorer.create(
            provider="openai",
            include_faithfulness=True,
            include_relevancy=False,
            include_coherence=False,
        )

        assert len(methods) == 1
        assert methods[0].name == "Faithfulness"

    @patch("mankinds_eval.methods.llm.base.LLMProvider")
    def test_no_methods_raises(self, mock_provider) -> None:
        """Test that excluding all methods raises ValueError."""
        with pytest.raises(ValueError, match="(?i)at least one method"):
            RAGScorer.create(
                provider="openai",
                include_faithfulness=False,
                include_relevancy=False,
                include_coherence=False,
            )

    def test_get_weights(self) -> None:
        """Test getting method weights."""
        weights = RAGScorer.get_weights()

        assert weights["Faithfulness"] == 0.4
        assert weights["AnswerRelevancy"] == 0.3
        assert weights["Coherence"] == 0.3

    def test_get_weights_custom(self) -> None:
        """Test getting custom weights."""
        weights = RAGScorer.get_weights(
            faithfulness_weight=0.5,
            relevancy_weight=0.3,
            coherence_weight=0.2,
        )

        assert weights["Faithfulness"] == 0.5
        assert weights["AnswerRelevancy"] == 0.3
        assert weights["Coherence"] == 0.2


# =============================================================================
# SafetyScorer Tests
# =============================================================================


class TestSafetyScorer:
    """Tests for the SafetyScorer preset."""

    def test_create_refusal_only(self) -> None:
        """Test creating SafetyScorer with only refusal check (no ML deps)."""
        methods = SafetyScorer.create(
            check_pii=False,
            check_toxicity=False,
            check_refusal=True,
        )

        assert len(methods) == 1
        assert methods[0].name == "NoRefusal"

    def test_no_methods_raises(self) -> None:
        """Test that excluding all methods raises ValueError."""
        with pytest.raises(ValueError, match="(?i)at least one"):
            SafetyScorer.create(
                check_pii=False,
                check_toxicity=False,
                check_refusal=False,
            )

    def test_get_check_names(self) -> None:
        """Test getting check names."""
        names = SafetyScorer.get_check_names()

        assert "PIIDetection" in names
        assert "Toxicity" in names
        assert "NoRefusal" in names

    def test_get_check_names_subset(self) -> None:
        """Test getting check names with subset."""
        names = SafetyScorer.get_check_names(
            check_pii=True,
            check_toxicity=False,
            check_refusal=True,
        )

        assert "PIIDetection" in names
        assert "Toxicity" not in names
        assert "NoRefusal" in names


# =============================================================================
# Integration Tests (requires ML dependencies)
# =============================================================================


try:
    from mankinds_eval.methods.ml.ner import PIIDetection  # noqa: F401
    from mankinds_eval.methods.ml.toxicity import Toxicity  # noqa: F401

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
class TestSafetyScorerWithML:
    """Tests for SafetyScorer with ML dependencies."""

    def test_create_with_pii(self) -> None:
        """Test creating SafetyScorer with PII detection."""
        methods = SafetyScorer.create(
            check_pii=True,
            check_toxicity=False,
            check_refusal=False,
        )

        assert len(methods) == 1
        assert methods[0].name == "PIIDetection"

    def test_create_with_toxicity(self) -> None:
        """Test creating SafetyScorer with toxicity detection."""
        methods = SafetyScorer.create(
            check_pii=False,
            check_toxicity=True,
            check_refusal=False,
        )

        assert len(methods) == 1
        assert methods[0].name == "Toxicity"

    def test_create_full(self) -> None:
        """Test creating SafetyScorer with all checks."""
        methods = SafetyScorer.create(
            check_pii=True,
            check_toxicity=True,
            check_refusal=True,
        )

        assert len(methods) == 3
        method_names = [m.name for m in methods]
        assert "PIIDetection" in method_names
        assert "Toxicity" in method_names
        assert "NoRefusal" in method_names

    def test_threshold_configuration(self) -> None:
        """Test threshold configuration is passed to methods."""
        methods = SafetyScorer.create(
            check_pii=True,
            check_toxicity=True,
            check_refusal=False,
            pii_threshold=0.7,
            toxicity_threshold=0.6,
        )

        pii_method = next(m for m in methods if m.name == "PIIDetection")
        toxicity_method = next(m for m in methods if m.name == "Toxicity")

        # PIIDetection doesn't use threshold, just verify it was created
        assert pii_method is not None
        assert toxicity_method.threshold == 0.6
