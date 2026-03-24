# Config Files

mankinds-eval supports YAML and JSON configuration files for defining scorers, methods, and global settings.

## Scorer Configuration

Define scorers in YAML or JSON:

### YAML Format

```yaml
name: qa_scorer
methods:
  - type: heuristic.FuzzyMatch
    threshold: 0.8
    algorithm: ratio
    target: expected
  - type: heuristic.TextLength
    min_length: 20
    max_length: 500
  - type: llm.SingleCriterionJudge
    provider: openai
    model: gpt-4o
    criterion: "Is the response accurate and helpful?"
    scale: "1-5"
    threshold: 3
```

### JSON Format

```json
{
  "name": "qa_scorer",
  "methods": [
    {
      "type": "heuristic.FuzzyMatch",
      "threshold": 0.8,
      "algorithm": "ratio",
      "target": "expected"
    },
    {
      "type": "heuristic.TextLength",
      "min_length": 20,
      "max_length": 500
    },
    {
      "type": "llm.SingleCriterionJudge",
      "provider": "openai",
      "model": "gpt-4o",
      "criterion": "Is the response accurate and helpful?",
      "scale": "1-5",
      "threshold": 3
    }
  ]
}
```

## Method Type Paths

The `type` field specifies the method class using dot notation:

| Type Path | Class |
|-----------|-------|
| `heuristic.ExactMatch` | `ExactMatch` |
| `heuristic.RegexMatch` | `RegexMatch` |
| `heuristic.ContainsAll` | `ContainsAll` |
| `heuristic.ContainsAny` | `ContainsAny` |
| `heuristic.FuzzyMatch` | `FuzzyMatch` |
| `heuristic.BLEU` | `BLEU` |
| `heuristic.ROUGE` | `ROUGE` |
| `heuristic.TextLength` | `TextLength` |
| `heuristic.WordCount` | `WordCount` |
| `heuristic.SentenceCount` | `SentenceCount` |
| `ml.EmbeddingsSimilarity` | `EmbeddingsSimilarity` |
| `ml.SentimentAnalysis` | `SentimentAnalysis` |
| `ml.PIIDetection` | `PIIDetection` |
| `ml.ZeroShotClassification` | `ZeroShotClassification` |
| `llm.SingleCriterionJudge` | `SingleCriterionJudge` |
| `llm.MultiCriteriaJudge` | `MultiCriteriaJudge` |
| `llm.PairwiseJudge` | `PairwiseJudge` |
| `llm.ConsensusJudge` | `ConsensusJudge` |

## Loading Config Files

### In Python

```python
from mankinds_eval import Scorer

scorer = Scorer.from_config("scorer_config.yaml")
results = scorer.run_sync(data)
```

### From CLI

```bash
mankinds-eval run --config scorer_config.yaml --data evaluations.jsonl
```

## Scorer Options

Additional scorer options in config:

```yaml
name: advanced_scorer
max_concurrent: 5
fail_fast: false
timeout: 30.0
methods:
  - type: heuristic.FuzzyMatch
    threshold: 0.8
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | `str` | Required | Scorer name |
| `methods` | `list` | Required | Method configurations |
| `max_concurrent` | `int` | `10` | Max concurrent evaluations |
| `fail_fast` | `bool` | `false` | Stop on first failure |
| `timeout` | `float` | `null` | Per-sample timeout (seconds) |

## Global Configuration

Set global defaults in `~/.mankinds_eval/config.yaml`:

```yaml
# Default LLM settings
llm:
  default_provider: openai
  default_model: gpt-4o
  temperature: 0.0

# Default ML settings
ml:
  device: cpu
  cache_dir: ~/.cache/mankinds_eval/models

# Scoring defaults
scoring:
  max_concurrent: 10
  timeout: 60.0

# Output settings
output:
  default_format: json
  html_theme: default
```

### Creating Global Config

```bash
mankinds-eval config init
```

This creates `~/.mankinds_eval/config.yaml` with default values.

### Viewing Current Config

```bash
mankinds-eval config show
```

### Setting Values

```bash
mankinds-eval config set llm.default_model gpt-4o-mini
mankinds-eval config set ml.device cuda
```

## Environment Variables

Environment variables override config file values:

| Variable | Description |
|----------|-------------|
| `MANKINDS_EVAL_CONFIG` | Path to global config file |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |

## Config Inheritance

Scorer configs can extend a base config:

```yaml
# base_scorer.yaml
name: base
methods:
  - type: heuristic.TextLength
    min_length: 10
    max_length: 1000
```

```yaml
# extended_scorer.yaml
extends: base_scorer.yaml
name: extended
methods:
  - type: heuristic.FuzzyMatch
    threshold: 0.8
```

The extended config inherits and can override methods from the base.

## Validation

Validate config files before use:

```bash
mankinds-eval config validate scorer_config.yaml
```

In Python:

```python
from mankinds_eval.config import validate_config

errors = validate_config("scorer_config.yaml")
if errors:
    print("Config errors:", errors)
```

## Example Configurations

### Basic QA Evaluation

```yaml
name: qa_basic
methods:
  - type: heuristic.ExactMatch
  - type: heuristic.FuzzyMatch
    threshold: 0.7
```

### Content Safety

```yaml
name: content_safety
methods:
  - type: ml.PIIDetection
    fail_on_detection: true
  - type: ml.SentimentAnalysis
    expected_sentiment: positive
    threshold: 0.3
```

### LLM Quality Assessment

```yaml
name: llm_quality
max_concurrent: 5
methods:
  - type: llm.MultiCriteriaJudge
    provider: openai
    model: gpt-4o
    criteria:
      accuracy:
        description: "Is the information correct?"
        weight: 0.4
      helpfulness:
        description: "Is the response helpful?"
        weight: 0.3
      clarity:
        description: "Is it clear and well-written?"
        weight: 0.3
    scale: "1-5"
    threshold: 3
```

### Comprehensive Evaluation

```yaml
name: comprehensive
max_concurrent: 10
methods:
  # Heuristic checks
  - type: heuristic.TextLength
    min_length: 50
    max_length: 2000
  - type: heuristic.FuzzyMatch
    threshold: 0.6
    target: expected
  
  # ML checks
  - type: ml.PIIDetection
    fail_on_detection: true
  
  # LLM evaluation
  - type: llm.SingleCriterionJudge
    provider: openai
    model: gpt-4o-mini
    criterion: "Is the response accurate and helpful?"
    scale: "1-5"
    threshold: 3
```
