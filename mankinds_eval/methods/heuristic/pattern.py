"""Pattern matching evaluation methods."""

from __future__ import annotations

import re
from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method


class ExactMatch(Method):
    """Check if output exactly matches expected.

    This method compares the output string against the expected string,
    with options for case sensitivity and whitespace handling.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "ExactMatch"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output", "expected"]

    def __init__(
        self,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize ExactMatch method.

        Args:
            case_sensitive: Whether comparison is case sensitive. Defaults to True.
            strip_whitespace: Whether to strip leading/trailing whitespace before
                comparison. Defaults to True.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.config.update(
            {
                "case_sensitive": case_sensitive,
                "strip_whitespace": strip_whitespace,
            }
        )

    def _normalize(self, text: str) -> str:
        """Normalize text based on configuration.

        Args:
            text: The text to normalize.

        Returns:
            Normalized text.
        """
        if self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output exactly matches expected.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if match, 0.0 otherwise.
        """
        output = self._normalize(sample.output)
        expected = self._normalize(sample.expected)  # type: ignore[arg-type]

        is_match = output == expected
        score = 1.0 if is_match else 0.0
        reason = "Exact match" if is_match else "No match"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=is_match,
            reason=reason,
        )


class RegexMatch(Method):
    """Check if output matches a regex pattern.

    This method uses regular expressions to evaluate whether the output
    contains or fully matches a specified pattern.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "RegexMatch"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        full_match: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RegexMatch method.

        Args:
            pattern: The regex pattern to match against.
            flags: Regex flags (e.g., re.IGNORECASE). Defaults to 0.
            full_match: If True, use fullmatch; otherwise use search.
                Defaults to False.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.pattern = pattern
        self.flags = flags
        self.full_match = full_match
        self._compiled_pattern = re.compile(pattern, flags)
        self.config.update(
            {
                "pattern": pattern,
                "flags": flags,
                "full_match": full_match,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output matches the regex pattern.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if match found, 0.0 otherwise.
            Metadata includes list of matched groups.
        """
        if self.full_match:
            match = self._compiled_pattern.fullmatch(sample.output)
        else:
            match = self._compiled_pattern.search(sample.output)

        if match:
            # Collect all matched groups
            matches: list[str] = []
            if match.group(0):
                matches.append(match.group(0))
            # Add any named or numbered groups
            groups = match.groups()
            if groups:
                matches.extend([g for g in groups if g is not None])

            return MethodResult(
                method_name=self.name,
                score=1.0,
                passed=True,
                reason="Pattern matched",
                metadata={"matches": matches},
            )
        else:
            return MethodResult(
                method_name=self.name,
                score=0.0,
                passed=False,
                reason="Pattern not matched",
                metadata={"matches": []},
            )


class ContainsAll(Method):
    """Check if output contains all specified keywords.

    This method verifies that every keyword in the provided list
    appears in the output string.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "ContainsAll"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        keywords: list[str],
        case_sensitive: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ContainsAll method.

        Args:
            keywords: List of keywords that must all be present.
            case_sensitive: Whether keyword matching is case sensitive.
                Defaults to False.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.config.update(
            {
                "keywords": keywords,
                "case_sensitive": case_sensitive,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output contains all keywords.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score as fraction of keywords found (0.0 to 1.0).
            Passed is True only if all keywords are found.
            Metadata includes lists of found and missing keywords.
        """
        output = sample.output if self.case_sensitive else sample.output.lower()

        found: list[str] = []
        missing: list[str] = []

        for keyword in self.keywords:
            check_keyword = keyword if self.case_sensitive else keyword.lower()
            if check_keyword in output:
                found.append(keyword)
            else:
                missing.append(keyword)

        total_keywords = len(self.keywords)
        score = 1.0 if total_keywords == 0 else len(found) / total_keywords

        all_found = len(missing) == 0
        reason = "All keywords found" if all_found else f"Missing {len(missing)} keyword(s)"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=all_found,
            reason=reason,
            metadata={"found": found, "missing": missing},
        )


class ContainsAny(Method):
    """Check if output contains any of the specified keywords.

    This method verifies that at least one keyword from the provided list
    appears in the output string.

    Attributes:
        name: Method identifier.
        required_fields: Fields required from the Sample.
    """

    name: str = "ContainsAny"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        keywords: list[str],
        case_sensitive: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ContainsAny method.

        Args:
            keywords: List of keywords where at least one must be present.
            case_sensitive: Whether keyword matching is case sensitive.
                Defaults to False.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.config.update(
            {
                "keywords": keywords,
                "case_sensitive": case_sensitive,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate if output contains any of the keywords.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if any keyword found, 0.0 otherwise.
            Passed is True if any keyword is found.
            Metadata includes list of found keywords.
        """
        output = sample.output if self.case_sensitive else sample.output.lower()

        found: list[str] = []

        for keyword in self.keywords:
            check_keyword = keyword if self.case_sensitive else keyword.lower()
            if check_keyword in output:
                found.append(keyword)

        any_found = len(found) > 0
        score = 1.0 if any_found else 0.0
        reason = f"Found {len(found)} keyword(s)" if any_found else "No keywords found"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=any_found,
            reason=reason,
            metadata={"found": found},
        )
