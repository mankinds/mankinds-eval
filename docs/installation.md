# Installation

## Requirements

- Python 3.9 or higher
- pip package manager

## Basic Installation

Install the base package with heuristic methods:

```bash
pip install mankinds-eval
```

This includes all heuristic evaluation methods (exact match, fuzzy match, regex, text metrics like BLEU and ROUGE).

## Optional Dependencies

mankinds-eval supports optional dependencies for ML and LLM-based evaluation methods.

### ML Methods

Install with machine learning support for embeddings, sentiment analysis, and NER:

```bash
pip install mankinds-eval[ml]
```

This adds:

- `sentence-transformers` - For embedding similarity comparisons
- `transformers` - For sentiment analysis and zero-shot classification
- `torch` - PyTorch for model inference

### LLM-as-Judge Methods

Install with LLM support for AI judge evaluations:

```bash
pip install mankinds-eval[llm]
```

This adds:

- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `litellm` - Unified interface for multiple LLM providers

### Full Installation

Install all optional dependencies:

```bash
pip install mankinds-eval[all]
```

## Verifying Installation

Verify your installation:

```bash
mankinds-eval version
```

Or in Python:

```python
import mankinds_eval
print(mankinds_eval.__version__)
```

## API Keys

For LLM-as-Judge methods, set the appropriate environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

## Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/mankinds-ai/mankinds-eval.git
cd mankinds-eval
pip install -e ".[all,dev]"
```
