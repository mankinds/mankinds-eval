# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2026-06-08

### Changed
- LLM judge methods now extract their JSON verdict through a shared, robust
  `extract_json_object` utility instead of a naive regex. It strips markdown
  code fences, skips leading prose, ignores braces nested inside string values,
  and repairs responses truncated by the model's `max_tokens` ceiling (closing
  unterminated strings and unclosed objects/arrays) so a verdict cut off mid
  `reason` field can still be parsed.

## [1.1.0] - 2026-05-18

### Added
- `LLMProvider` and `LLMMethod` now accept a `provider_kwargs: dict | None`
  parameter that is forwarded verbatim to `litellm.acompletion`. Use it to pass
  provider-specific authentication or routing arguments (eg. `vertex_credentials`,
  `vertex_project`, `vertex_location` for Google Vertex AI; `aws_region_name`
  for Bedrock; `api_version` for Azure). The wrapper itself stays generic and
  does not need to know about each provider's auth model.

### Changed
- `_get_model_string` is now generic: any litellm-supported provider works
  without code change (model gets prefixed with `<provider>/` when the bare
  name is given, untouched when already prefixed; `openai` stays unprefixed
  as the litellm default).
- The API key is now passed directly to `litellm.acompletion` via the
  `api_key` parameter instead of being injected into a per-provider environment
  variable. LiteLLM's environment fallback still applies when no key is provided,
  so existing code that relies on `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
  `MISTRAL_API_KEY` env vars keeps working.

### Removed
- Internal `_setup_api_key` helper that set hardcoded environment variables
  for a closed list of providers. Replaced by direct `api_key` pass-through,
  which works for every provider supported by litellm.

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- Heuristic evaluation methods (ExactMatch, FuzzyMatch, TextLength, ROUGE, BLEU, pattern matching)
- ML-based evaluation methods (embeddings similarity, sentiment analysis, NER/PII detection, zero-shot classification)
- LLM-as-Judge methods (single criterion, multi-criteria, pairwise comparison, consensus)
- Data loaders for CSV, JSON, JSONL, and HuggingFace Datasets
- HTML scorecard generation
- JSON export for programmatic analysis
- CLI for running evaluations
- YAML/JSON configuration file support
