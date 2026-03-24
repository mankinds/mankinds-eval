"""HTML scorecard generator for evaluation results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from mankinds_eval.output.schema import EvaluationResult

# CSS styles for the scorecard HTML output
SCORECARD_CSS = """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont,
                "Segoe UI", Roboto, "Helvetica Neue",
                Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            background: #fff;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .title {
            font-size: 24px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 20px;
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }

        .meta-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .meta-label {
            font-size: 12px;
            font-weight: 500;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .meta-value {
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
        }

        /* Sections */
        .section {
            background: #fff;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }

        .no-data {
            color: #666;
            font-style: italic;
        }

        /* Summary Table */
        .table-wrapper {
            overflow-x: auto;
        }

        .summary-table {
            width: 100%;
            border-collapse: collapse;
        }

        .summary-table th,
        .summary-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        .summary-table th {
            font-weight: 600;
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: #fafafa;
        }

        .summary-table tbody tr:hover {
            background: #fafafa;
        }

        .summary-table .method-name {
            font-weight: 500;
        }

        .summary-table .score {
            font-family: "SF Mono", Monaco, "Consolas", monospace;
        }

        .summary-table .error-count.has-errors {
            color: #dc3545;
            font-weight: 600;
        }

        /* Pass Rate Bar */
        .rate-container {
            position: relative;
            width: 120px;
            height: 20px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }

        .rate-bar {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: #28a745;
            transition: width 0.3s ease;
        }

        .rate-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 12px;
            font-weight: 600;
            color: #333;
            z-index: 1;
        }

        /* Results List */
        .results-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        /* Sample Card */
        .sample-card {
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }

        .sample-card.passed {
            border-left: 4px solid #28a745;
        }

        .sample-card.failed {
            border-left: 4px solid #dc3545;
        }

        .sample-card.neutral {
            border-left: 4px solid #6c757d;
        }

        .sample-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: #fafafa;
            cursor: pointer;
            user-select: none;
        }

        .sample-header:hover {
            background: #f0f0f0;
        }

        .sample-index {
            font-weight: 600;
            color: #333;
        }

        .sample-status {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }

        .sample-content {
            padding: 16px;
            border-top: 1px solid #eee;
        }

        /* Fields */
        .sample-data {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 16px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .field-label {
            font-size: 12px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
        }

        .field-details {
            cursor: pointer;
        }

        .field-details .field-label {
            display: inline;
        }

        .expand-hint {
            font-size: 11px;
            font-weight: normal;
            color: #999;
            text-transform: none;
        }

        .field-value {
            font-family: "SF Mono", Monaco, "Consolas", monospace;
            font-size: 13px;
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-word;
            border: 1px solid #eee;
            margin: 0;
        }

        .field-details[open] .field-value.preview {
            display: none;
        }

        .field-details:not([open]) .field-value.full {
            display: none;
        }

        /* Method Results */
        .method-results-title {
            font-size: 14px;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
        }

        .method-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .method-item {
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }

        .method-item.passed {
            border-left: 3px solid #28a745;
        }

        .method-item.failed {
            border-left: 3px solid #dc3545;
        }

        .method-item.error {
            border-left: 3px solid #fd7e14;
        }

        .method-item.neutral {
            border-left: 3px solid #6c757d;
        }

        .method-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: #fafafa;
        }

        .method-item .method-name {
            font-weight: 500;
            color: #333;
        }

        .method-body {
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        /* Badges */
        .badge {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .badge-passed {
            background: #d4edda;
            color: #155724;
        }

        .badge-failed {
            background: #f8d7da;
            color: #721c24;
        }

        .badge-error {
            background: #fff3cd;
            color: #856404;
        }

        .badge-neutral {
            background: #e9ecef;
            color: #495057;
        }

        /* Score Display */
        .score-display {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .score-label,
        .reason-label,
        .error-label {
            font-size: 12px;
            font-weight: 500;
            color: #666;
        }

        .score-value {
            font-family: "SF Mono", Monaco, "Consolas", monospace;
            font-weight: 600;
            font-size: 14px;
        }

        .score-bar-container {
            flex: 1;
            max-width: 150px;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }

        .score-bar {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .score-bar.passed {
            background: #28a745;
        }

        .score-bar.failed {
            background: #dc3545;
        }

        .score-bar.error {
            background: #fd7e14;
        }

        .score-bar.neutral {
            background: #6c757d;
        }

        /* Reason */
        .reason {
            display: flex;
            gap: 8px;
        }

        .reason-text {
            color: #555;
        }

        /* Error */
        .error-message {
            display: flex;
            gap: 8px;
            padding: 8px;
            background: #fff3cd;
            border-radius: 4px;
        }

        .error-text {
            color: #856404;
            font-family: "SF Mono", Monaco, "Consolas", monospace;
            font-size: 13px;
        }

        /* Metadata */
        .metadata-details {
            margin-top: 4px;
        }

        .metadata-label {
            font-size: 12px;
            font-weight: 500;
            color: #666;
            cursor: pointer;
        }

        .metadata-content {
            font-family: "SF Mono", Monaco, "Consolas", monospace;
            font-size: 12px;
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-word;
            margin-top: 8px;
            border: 1px solid #eee;
        }

        /* Composite Badge */
        .badge-composite {
            background: #e0e7ff;
            color: #3730a3;
            font-size: 10px;
            padding: 2px 6px;
            margin-left: 8px;
        }

        /* Nested Method Results */
        .method-list.nested {
            margin-left: 20px;
            margin-top: 12px;
            padding-left: 12px;
            border-left: 2px solid #ddd;
        }

        .method-list.nested-1 .method-item {
            background: #fafafa;
        }

        .method-list.nested-1 .method-header {
            background: #f5f5f5;
        }

        .method-list.nested-2 .method-item {
            background: #f5f5f5;
        }

        .method-list.nested-2 .method-header {
            background: #f0f0f0;
        }

        .method-list.nested-3 .method-item {
            background: #f0f0f0;
        }

        .method-list.nested-3 .method-header {
            background: #ebebeb;
        }

        .sub-results-title {
            font-size: 12px;
            font-weight: 500;
            color: #666;
            margin-bottom: 8px;
            margin-top: 8px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 12px;
            }

            .header,
            .section {
                padding: 16px;
            }

            .meta-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .rate-container {
                width: 80px;
            }
        }
"""


def generate_scorecard(result: EvaluationResult, output_path: str | Path) -> None:
    """Generate HTML scorecard from evaluation results.

    Args:
        result: EvaluationResult object containing meta, summary, and results.
        output_path: Path to write HTML file.
    """
    html_content = render_scorecard(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")


def render_scorecard(result: EvaluationResult) -> str:
    """Render evaluation results to HTML string.

    Args:
        result: EvaluationResult object containing meta, summary, and results.

    Returns:
        Complete HTML document as string.
    """
    meta = result.meta
    summary = result.summary
    results = result.results

    header_html = _render_header(meta)
    summary_html = _render_summary(summary)
    results_html = _render_results(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Scorecard - {_escape(meta.get("scorer_name", "Unknown"))}</title>
    <style>
{_get_styles()}
    </style>
</head>
<body>
    <div class="container">
        {header_html}
        {summary_html}
        {results_html}
    </div>
</body>
</html>"""


def _escape(value: Any) -> str:
    """Escape HTML special characters."""
    return html.escape(str(value)) if value is not None else ""


def _format_duration(duration: Any) -> str:
    """Format duration value for display."""
    if duration is None:
        return "N/A"
    try:
        seconds = float(duration)
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
    except (ValueError, TypeError):
        return str(duration)


def _format_score(score: Any) -> str:
    """Format score for display."""
    if score is None:
        return "N/A"
    try:
        score_val = float(score)
        if score_val == int(score_val):
            return str(int(score_val))
        return f"{score_val:.3f}"
    except (ValueError, TypeError):
        return str(score)


def _format_percentage(value: Any) -> str:
    """Format value as percentage."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(value)


def _truncate_text(text: str, max_length: int = 200) -> tuple[str, bool]:
    """Truncate text if too long.

    Returns:
        Tuple of (truncated_text, was_truncated).
    """
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


def _render_header(meta: dict[str, Any]) -> str:
    """Render the header section."""
    scorer_name = _escape(meta.get("scorer_name", "Unknown Scorer"))
    created_at = _escape(meta.get("created_at", "N/A"))
    version = _escape(meta.get("version", "N/A"))
    sample_count = meta.get("sample_count", "N/A")
    duration = _format_duration(meta.get("duration"))
    methods = meta.get("methods", [])
    method_count = len(methods) if isinstance(methods, list) else "N/A"

    return f"""
        <header class="header">
            <h1 class="title">Evaluation Scorecard</h1>
            <div class="meta-grid">
                <div class="meta-item">
                    <span class="meta-label">Scorer</span>
                    <span class="meta-value">{scorer_name}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Timestamp</span>
                    <span class="meta-value">{created_at}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Version</span>
                    <span class="meta-value">{version}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Samples</span>
                    <span class="meta-value">{sample_count}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Methods</span>
                    <span class="meta-value">{method_count}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Duration</span>
                    <span class="meta-value">{duration}</span>
                </div>
            </div>
        </header>"""


def _render_summary(summary: dict[str, Any]) -> str:
    """Render the summary section with per-method statistics."""
    if not summary:
        return """
        <section class="section">
            <h2 class="section-title">Summary</h2>
            <p class="no-data">No summary data available.</p>
        </section>"""

    rows = []
    for method_name, stats in summary.items():
        mean_score = _format_score(stats.get("mean_score"))
        pass_rate = _format_percentage(stats.get("pass_rate"))
        min_score = _format_score(stats.get("min_score"))
        max_score = _format_score(stats.get("max_score"))
        error_count = stats.get("error_count", 0)
        sample_count = stats.get("sample_count", "N/A")

        # Calculate pass rate bar width
        pass_rate_val = stats.get("pass_rate")
        bar_width = f"{float(pass_rate_val) * 100:.1f}" if pass_rate_val is not None else "0"
        err_cls = " has-errors" if error_count > 0 else ""

        rows.append(f"""
                <tr>
                    <td class="method-name">{_escape(method_name)}</td>
                    <td class="score">{mean_score}</td>
                    <td class="pass-rate">
                        <div class="rate-container">
                            <div class="rate-bar" style="width: {bar_width}%;"></div>
                            <span class="rate-text">{pass_rate}</span>
                        </div>
                    </td>
                    <td class="score">{min_score}</td>
                    <td class="score">{max_score}</td>
                    <td class="error-count{err_cls}">{error_count}</td>
                    <td class="sample-count">{sample_count}</td>
                </tr>""")

    return f"""
        <section class="section">
            <h2 class="section-title">Summary</h2>
            <div class="table-wrapper">
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Mean Score</th>
                            <th>Pass Rate</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Errors</th>
                            <th>Samples</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
        </section>"""


def _render_results(results: list[dict[str, Any]]) -> str:
    """Render the results section with per-sample details."""
    if not results:
        return """
        <section class="section">
            <h2 class="section-title">Results</h2>
            <p class="no-data">No results available.</p>
        </section>"""

    sample_cards = []
    for idx, sample in enumerate(results):
        sample_html = _render_sample(idx, sample)
        sample_cards.append(sample_html)

    return f"""
        <section class="section">
            <h2 class="section-title">Results</h2>
            <div class="results-list">
                {"".join(sample_cards)}
            </div>
        </section>"""


def _render_sample(idx: int, sample: dict[str, Any]) -> str:
    """Render a single sample result."""
    input_val = sample.get("input", "")
    output_val = sample.get("output", "")
    expected_val = sample.get("expected")
    methods = sample.get("methods", {})

    # Format input/output for display
    input_str = _format_sample_value(input_val)
    output_str = _format_sample_value(output_val)
    expected_str = _format_sample_value(expected_val) if expected_val is not None else None

    # Truncate for preview
    input_preview, input_truncated = _truncate_text(input_str)
    output_preview, output_truncated = _truncate_text(output_str)
    expected_preview = None
    expected_truncated = False
    if expected_str:
        expected_preview, expected_truncated = _truncate_text(expected_str)

    # Determine overall status
    overall_status = _get_overall_status(methods)
    status_class = (
        "passed"
        if overall_status == "passed"
        else "failed"
        if overall_status == "failed"
        else "neutral"
    )

    # Build input/output sections
    input_html = _render_expandable_field("Input", input_preview, input_str, input_truncated)
    output_html = _render_expandable_field("Output", output_preview, output_str, output_truncated)
    expected_html = ""
    if expected_str and expected_preview is not None:
        expected_html = _render_expandable_field(
            "Expected", expected_preview, expected_str, expected_truncated
        )

    # Build method results
    method_results_html = _render_method_results(methods)

    return f"""
            <details class="sample-card {status_class}">
                <summary class="sample-header">
                    <span class="sample-index">Sample {idx}</span>
                    <span class="sample-status badge-{status_class}">{overall_status.upper()}</span>
                </summary>
                <div class="sample-content">
                    <div class="sample-data">
                        {input_html}
                        {output_html}
                        {expected_html}
                    </div>
                    <div class="method-results">
                        <h4 class="method-results-title">Method Results</h4>
                        {method_results_html}
                    </div>
                </div>
            </details>"""


def _format_sample_value(value: Any) -> str:
    """Format a sample value (input/output/expected) for display."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        import json

        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _render_expandable_field(
    label: str,
    preview: str,
    full_content: str,
    is_truncated: bool,
) -> str:
    """Render an expandable field with preview and full content."""
    escaped_preview = _escape(preview)
    escaped_full = _escape(full_content)

    if is_truncated:
        return f"""
                        <div class="field">
                            <details class="field-details">
                                <summary class="field-label">{label}
                                    <span class="expand-hint">(click to expand)</span></summary>
                                <pre class="field-value full">{escaped_full}</pre>
                            </details>
                            <pre class="field-value preview">{escaped_preview}</pre>
                        </div>"""
    else:
        return f"""
                        <div class="field">
                            <span class="field-label">{label}</span>
                            <pre class="field-value">{escaped_preview}</pre>
                        </div>"""


def _get_overall_status(methods: dict[str, Any]) -> str:
    """Determine overall status from method results."""
    if not methods:
        return "neutral"

    has_passed = False
    has_failed = False

    for method_result in methods.values():
        if method_result.get("error"):
            has_failed = True
        elif method_result.get("passed") is True:
            has_passed = True
        elif method_result.get("passed") is False:
            has_failed = True

    if has_failed:
        return "failed"
    if has_passed:
        return "passed"
    return "neutral"


def _render_method_results(methods: dict[str, Any], depth: int = 0) -> str:
    """Render method results for a sample.

    Args:
        methods: Dictionary mapping method names to their results.
        depth: Nesting depth for visual indentation (0 = top level).

    Returns:
        HTML string for the method results.
    """
    if not methods:
        return '<p class="no-data">No method results.</p>'

    result_items = []
    for method_name, result in methods.items():
        result_html = _render_single_method_result(method_name, result, depth)
        result_items.append(result_html)

    # Add nested class for indentation styling
    nested_class = f" nested nested-{depth}" if depth > 0 else ""

    return f"""
                        <div class="method-list{nested_class}">
                            {"".join(result_items)}
                        </div>"""


def _render_single_method_result(
    method_name: str,
    result: dict[str, Any],
    depth: int = 0,
) -> str:
    """Render a single method result.

    Args:
        method_name: Name of the method.
        result: Dictionary containing score, passed, reason, error, and metadata.
        depth: Nesting depth for visual indentation (0 = top level).

    Returns:
        HTML string for the method result.
    """
    score = result.get("score")
    passed = result.get("passed")
    reason = result.get("reason", "")
    error = result.get("error")
    metadata = result.get("metadata")

    # Check for sub_results in metadata (composite method)
    sub_results: dict[str, Any] | None = None
    if metadata and isinstance(metadata, dict):
        sub_results = metadata.get("sub_results")

    is_composite = sub_results is not None and isinstance(sub_results, dict)

    # Determine status
    if error:
        status_class = "error"
        status_text = "ERROR"
    elif passed is True:
        status_class = "passed"
        status_text = "PASSED"
    elif passed is False:
        status_class = "failed"
        status_text = "FAILED"
    else:
        status_class = "neutral"
        status_text = "N/A"

    # Composite badge
    composite_badge = ""
    if is_composite:
        composite_badge = '<span class="badge badge-composite">composite</span>'

    # Score display
    score_html = ""
    if score is not None:
        score_formatted = _format_score(score)
        # Create score bar (assuming 0-1 scale, but handle other ranges)
        try:
            score_val = float(score)
            # Normalize to 0-100 for display
            if 0 <= score_val <= 1:
                bar_width = score_val * 100
            elif score_val > 1:
                bar_width = min(score_val, 100)
            else:
                bar_width = 0
        except (ValueError, TypeError):
            bar_width = 0

        score_html = f"""
                                <div class="score-display">
                                    <span class="score-label">Score:</span>
                                    <span class="score-value">{score_formatted}</span>
                                    <div class="score-bar-container">
                                        <div class="score-bar {status_class}"
                                            style="width: {bar_width:.1f}%;"></div>
                                    </div>
                                </div>"""

    # Reason display
    reason_html = ""
    if reason:
        reason_escaped = _escape(reason)
        reason_html = f"""
                                <div class="reason">
                                    <span class="reason-label">Reason:</span>
                                    <span class="reason-text">{reason_escaped}</span>
                                </div>"""

    # Error display
    error_html = ""
    if error:
        error_escaped = _escape(str(error))
        error_html = f"""
                                <div class="error-message">
                                    <span class="error-label">Error:</span>
                                    <span class="error-text">{error_escaped}</span>
                                </div>"""

    # Metadata display (filter out sub_results for separate rendering)
    metadata_html = ""
    if metadata:
        import json

        # Filter out sub_results from metadata display
        display_metadata = metadata
        if is_composite and isinstance(metadata, dict):
            display_metadata = {k: v for k, v in metadata.items() if k != "sub_results"}

        if display_metadata:
            try:
                metadata_str = json.dumps(display_metadata, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                metadata_str = str(display_metadata)
            metadata_escaped = _escape(metadata_str)
            metadata_html = f"""
                                <details class="metadata-details">
                                    <summary class="metadata-label">Metadata</summary>
                                    <pre class="metadata-content">{metadata_escaped}</pre>
                                </details>"""

    # Sub-results display (recursive rendering for composite methods)
    sub_results_html = ""
    if is_composite and sub_results:
        sub_results_html = f"""
                                <div class="sub-results">
                                    <span class="sub-results-title">Sub-Results:</span>
                                    {_render_method_results(sub_results, depth + 1)}
                                </div>"""

    return f"""
                            <div class="method-item {status_class}">
                                <div class="method-header">
                                    <span class="method-name">{_escape(method_name)}
                                        {composite_badge}</span>
                                    <span class="badge badge-{status_class}">{status_text}</span>
                                </div>
                                <div class="method-body">
                                    {score_html}
                                    {reason_html}
                                    {error_html}
                                    {metadata_html}
                                    {sub_results_html}
                                </div>
                            </div>"""


def _get_styles() -> str:
    """Return the CSS styles for the scorecard."""
    return SCORECARD_CSS
