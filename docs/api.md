# API Reference

Quick reference for main classes and functions in mankinds-eval.

## Core Classes

### Scorer

```python
from mankinds_eval import Scorer

scorer = Scorer(
    name: str,                    # Required: Scorer name
    methods: list[Method],        # Required: Evaluation methods
    max_concurrent: int = 10,     # Max concurrent evaluations
    fail_fast: bool = False,      # Stop on first failure
    timeout: float | None = None  # Per-sample timeout
)

# Methods
results = scorer.run_sync(data)           # Synchronous evaluation
results = await scorer.run(data)          # Async evaluation
scorer = Scorer.from_config("config.yaml") # Load from config
```

### Sample

```python
from mankinds_eval.core import Sample

sample = Sample(
    input: str,                   # Required: Input prompt
    output: str,                  # Required: Model output
    expected: str | None = None,  # Expected/reference answer
    metadata: dict | None = None, # Additional metadata
    messages: list[dict] | None = None  # Multi-turn conversation
)
```

### EvaluationResult

```python
from mankinds_eval.core import EvaluationResult

# Properties
result.meta          # Metadata dict
result.results       # List of sample results
result.summary       # Summary statistics

# Methods
result.to_json("output.json")     # Export to JSON
result.to_html("scorecard.html")  # Export to HTML
result.to_dict()                  # Convert to dict
result.merge(other_result)        # Merge results
```

### MethodResult

```python
from mankinds_eval.core import MethodResult

# Properties
result.passed: bool        # Whether evaluation passed
result.score: float        # Normalized score (0.0-1.0)
result.error: str | None   # Error message if failed
result.details: dict       # Additional details
```

## Data Loaders

```python
from mankinds_eval.data import load_csv, load_jsonl, load_json, load_hf_dataset

# CSV
data = load_csv(
    path: str,
    input_col: str = "input",
    output_col: str = "output",
    expected_col: str = "expected"
)

# JSONL
data = load_jsonl(path: str)

# JSON
data = load_json(path: str, key: str | None = None)

# HuggingFace Datasets
data = load_hf_dataset(
    dataset_name: str,
    split: str = "test",
    input_col: str = "input",
    output_col: str = "output",
    expected_col: str | None = None
)
```

## Heuristic Methods

```python
from mankinds_eval.methods.heuristic import (
    ExactMatch,
    RegexMatch,
    ContainsAll,
    ContainsAny,
    FuzzyMatch,
    BLEU,
    ROUGE,
    TextLength,
    WordCount,
    SentenceCount,
)

# ExactMatch
ExactMatch(case_sensitive=True, strip_whitespace=True, normalize_unicode=False)

# RegexMatch
RegexMatch(pattern: str, flags: int = 0, full_match: bool = False)

# ContainsAll
ContainsAll(substrings: list[str], case_sensitive: bool = True)

# ContainsAny
ContainsAny(substrings: list[str], case_sensitive: bool = True)

# FuzzyMatch
FuzzyMatch(threshold: float = 0.8, algorithm: str = "ratio", target: str = "expected")

# BLEU
BLEU(threshold: float = 0.0, max_ngram: int = 4, smoothing: bool = True)

# ROUGE
ROUGE(threshold: float = 0.0, rouge_type: str = "rouge-l", metric: str = "f")

# TextLength
TextLength(min_length: int = 0, max_length: int | None = None, unit: str = "characters")

# WordCount
WordCount(min_words: int = 0, max_words: int | None = None)

# SentenceCount
SentenceCount(min_sentences: int = 0, max_sentences: int | None = None)
```

## ML Methods

```python
from mankinds_eval.methods.ml import (
    EmbeddingsSimilarity,
    SentimentAnalysis,
    PIIDetection,
    ZeroShotClassification,
)

# EmbeddingsSimilarity
EmbeddingsSimilarity(
    threshold: float = 0.7,
    compare: str = "output_vs_expected",
    model: str = "all-MiniLM-L6-v2",
    device: str = "cpu"
)

# SentimentAnalysis
SentimentAnalysis(
    expected_sentiment: str,  # "positive", "negative", "neutral"
    threshold: float = 0.5,
    model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cpu"
)

# PIIDetection
PIIDetection(
    fail_on_detection: bool = True,
    entity_types: list[str] | None = None,
    threshold: float = 0.5,
    model: str = "dslim/bert-base-NER"
)

# ZeroShotClassification
ZeroShotClassification(
    labels: list[str],
    expected_label: str | None = None,
    threshold: float = 0.5,
    model: str = "facebook/bart-large-mnli",
    multi_label: bool = False,
    device: str = "cpu"
)
```

## LLM Methods

```python
from mankinds_eval.methods.llm import (
    SingleCriterionJudge,
    MultiCriteriaJudge,
    PairwiseJudge,
    ConsensusJudge,
)

# SingleCriterionJudge
SingleCriterionJudge(
    provider: str = "openai",
    model: str = "gpt-4o",
    criterion: str,             # Required
    scale: str = "1-5",
    threshold: float = 3,
    temperature: float = 0.0,
    include_reasoning: bool = True
)

# MultiCriteriaJudge
MultiCriteriaJudge(
    provider: str = "openai",
    model: str = "gpt-4o",
    criteria: dict,             # Required: {name: {description, weight}}
    scale: str = "1-5",
    threshold: float = 3,
    aggregation: str = "weighted_mean"
)

# PairwiseJudge
PairwiseJudge(
    provider: str = "openai",
    model: str = "gpt-4o",
    criterion: str,             # Required
    compare_to: str = "expected",
    swap_order: bool = True
)

# ConsensusJudge
ConsensusJudge(
    judges: list[dict],         # Required: [{provider, model}]
    criterion: str,             # Required
    scale: str = "1-5",
    threshold: float = 3,
    consensus_method: str = "majority"
)
```

## Generators

```python
from mankinds_eval.generators import Generator
from mankinds_eval.core import Sample

class MyGenerator(Generator):
    name: str = "my_generator"
    
    async def generate(self, base_prompt: str, n_variants: int = 10) -> list[Sample]:
        # Implementation
        pass

# Usage
generator = MyGenerator()
samples = generator.generate_sync(base_prompt, n_variants)
```

## Configuration

```python
from mankinds_eval.config import load_config, validate_config

# Load config
config = load_config("config.yaml")

# Validate config
errors = validate_config("config.yaml")

# Create scorer from config
scorer = Scorer.from_config("config.yaml")
```

## Import Paths

```python
# Main imports
from mankinds_eval import Scorer
from mankinds_eval.core import Sample, MethodResult, EvaluationResult

# Data
from mankinds_eval.data import load_csv, load_jsonl, load_json, load_hf_dataset

# Heuristic methods
from mankinds_eval.methods.heuristic import ExactMatch, FuzzyMatch, BLEU, ROUGE, ...

# ML methods
from mankinds_eval.methods.ml import EmbeddingsSimilarity, SentimentAnalysis, ...

# LLM methods
from mankinds_eval.methods.llm import SingleCriterionJudge, MultiCriteriaJudge, ...

# Generators
from mankinds_eval.generators import Generator

# Config
from mankinds_eval.config import load_config, validate_config
```
