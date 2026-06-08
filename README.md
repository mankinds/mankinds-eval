<p align="center">
  <a href="https://github.com/mankinds-io/mankinds-eval">
    <img src="assets/logo.svg" alt="Mankinds" width="320" />
  </a>
</p>

<p align="center">
  <strong>Open source Python library for AI evaluation</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/mankinds-eval/"><img src="https://img.shields.io/pypi/v/mankinds-eval?style=flat-square&color=637AB9" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/mankinds-eval/"><img src="https://img.shields.io/pypi/pyversions/mankinds-eval?style=flat-square&color=637AB9" alt="Python versions" /></a>
  <a href="https://github.com/mankinds-io/mankinds-eval/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-637AB9?style=flat-square" alt="License" /></a>
  <a href="https://github.com/mankinds-io/mankinds-eval/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mankinds-io/mankinds-eval/ci.yml?branch=main&style=flat-square&label=CI" alt="CI" /></a>
  <a href="https://pypi.org/project/mankinds-eval/"><img src="https://img.shields.io/pypi/dm/mankinds-eval?style=flat-square&color=637AB9" alt="Downloads" /></a>
</p>

<p align="center">
  <a href="https://mankinds-io.github.io/mankinds-eval">Documentation</a> |
  <a href="#metrics-and-features">Metrics and Features</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="https://github.com/mankinds-io/mankinds-eval/tree/main/examples">Examples</a>
</p>

---

**mankinds-eval** is a modular, open-source evaluation framework for LLM applications. It provides a library of evaluation methods that you assemble to build custom scorers tailored to your specific use cases.

Whether you're building RAG pipelines, chatbots, AI agents, or any LLM-based application, mankinds-eval lets you combine heuristic checks (fast, free, deterministic), ML-based analysis, and LLM-as-Judge evaluations. Run everything locally or use external providers. You control the trade-offs.



---

## Metrics and Features

- Large variety of evaluation methods powered by heuristics, ML models (running locally), or any LLM provider:
  - **Heuristic methods:**
    - ExactMatch, FuzzyMatch, RegexMatch
    - ContainsAll, ContainsAny
    - BLEU, ROUGE
    - TextLength, WordCount, SentenceCount
    - JSONValid, JSONSchema
    - NoRefusal
  - **ML methods** (local models via transformers):
    - EmbeddingsSimilarity
    - SentimentAnalysis
    - Toxicity
    - PIIDetection
    - LanguageDetection
    - ZeroShotClassification
  - **LLM-as-Judge methods:**
    - Faithfulness
    - AnswerRelevancy
    - Coherence
    - Helpfulness
    - Correctness
    - SingleCriterionJudge (custom criteria)
    - MultiCriteriaJudge (weighted scoring)
    - PairwiseJudge (A vs B comparison)
    - ConsensusJudge (multi-LLM voting)
- Compose methods into pipelines with aggregation modes (all, any, weighted, sequential).
- Pre-built presets for common scenarios: RAGScorer, SafetyScorer.
- Load evaluation data from CSV, JSONL, JSON, or HuggingFace Datasets.
- Export results to JSON or HTML scorecards.
- CLI for CI/CD integration.
- Define scorers in Python or YAML configuration files.

---

## Quick Start

### Installation

```bash
pip install mankinds-eval
```

### Writing your first evaluation

Create a file `evaluate.py`:

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import FuzzyMatch, ROUGE

scorer = Scorer(
    name="qa_scorer",
    methods=[
        FuzzyMatch(threshold=0.7),
        ROUGE(threshold=0.4),
    ]
)

test_case = {
    "input": "What if these shoes don't fit?",
    "output": "You have 30 days to get a full refund at no extra cost.",
    "expected": "We offer a 30-day full refund at no extra costs.",
}

results = scorer.run_sync([test_case])
print(results.summary)
```

Run it:

```bash
python evaluate.py
```

Let's break down what happened:

- The `input` field contains the user query, `output` is the LLM response you want to evaluate.
- The `expected` field is the reference answer used by methods like FuzzyMatch and ROUGE.
- `FuzzyMatch` computes string similarity using token-based matching. A score >= 0.7 passes.
- `ROUGE` measures overlap between the output and expected text. A score >= 0.4 passes.
- All method scores range from 0 to 1. The `threshold` determines if the evaluation passes.

---

## Evaluating with LLM-as-Judge

For semantic evaluation that goes beyond string matching, use LLM-as-Judge methods. These use an LLM to evaluate outputs based on criteria you define.

```bash
pip install mankinds-eval[llm]
export OPENAI_API_KEY="your-api-key"
```

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.llm import Faithfulness, AnswerRelevancy

scorer = Scorer(
    name="rag_evaluator",
    methods=[
        Faithfulness(provider="openai", threshold=0.7),
        AnswerRelevancy(provider="openai", threshold=0.7),
    ]
)

test_case = {
    "input": "What is the refund policy?",
    "output": "You can get a full refund within 30 days.",
    "context": "All customers are eligible for a 30-day full refund at no extra cost.",
}

results = scorer.run_sync([test_case])
```

- `Faithfulness` checks if the output is grounded in the provided context (no hallucinations).
- `AnswerRelevancy` checks if the output actually addresses the input question.
- The `provider` parameter specifies which LLM to use (openai, anthropic, etc.).

---

## Evaluating with ML Models

For local evaluation without API calls, use ML methods that run transformer models on your machine.

```bash
pip install mankinds-eval[ml]
```

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.ml import Toxicity, EmbeddingsSimilarity

scorer = Scorer(
    name="safety_check",
    methods=[
        Toxicity(threshold=0.5),
        EmbeddingsSimilarity(threshold=0.8),
    ]
)

results = scorer.run_sync([test_case])
```

- `Toxicity` uses a local model to detect harmful content. Lower scores are better.
- `EmbeddingsSimilarity` computes semantic similarity using sentence embeddings.

---

## Evaluating a Dataset

Evaluate multiple samples at once by passing a list or loading from a file:

```python
from mankinds_eval import Scorer, load_samples
from mankinds_eval.methods.heuristic import ExactMatch

scorer = Scorer(name="batch_eval", methods=[ExactMatch()])

# From a list
samples = [
    {"input": "Q1", "output": "A", "expected": "A"},
    {"input": "Q2", "output": "B", "expected": "C"},
]
results = scorer.run_sync(samples)

# Or from a file
samples = load_samples("data.jsonl")
results = scorer.run_sync(samples)

# Export results
results.to_json("results.json")
results.to_html("scorecard.html")
```

Supported formats: CSV, JSONL, JSON, HuggingFace Datasets.

---

## Using Config Files

Define scorers in YAML for reproducibility and CI/CD integration:

```yaml
# scorer.yaml
name: qa_scorer
methods:
  - type: heuristic.FuzzyMatch
    threshold: 0.7
    algorithm: token_set_ratio
  - type: heuristic.ROUGE
    threshold: 0.4
  - type: llm.Faithfulness
    provider: openai
    threshold: 0.7
```

Load and run:

```python
from mankinds_eval import Scorer

scorer = Scorer.from_config("scorer.yaml")
results = scorer.run_sync("data.jsonl")
```

Or use the CLI:

```bash
mankinds-eval run -c scorer.yaml -d data.jsonl -o results.json --html scorecard.html
```

---

## Composite Pipelines

Combine methods with different aggregation logic:

```python
from mankinds_eval import Scorer
from mankinds_eval.methods import CompositeMethod
from mankinds_eval.methods.heuristic import TextLength, NoRefusal, FuzzyMatch

quality_gate = CompositeMethod(
    name="quality_gate",
    methods=[
        TextLength(min_length=50, max_length=500),
        NoRefusal(),
        FuzzyMatch(threshold=0.6),
    ],
    mode="all",  # all checks must pass
)

scorer = Scorer(name="eval", methods=[quality_gate])
results = scorer.run_sync(data)
```

Aggregation modes:

- `all` - all methods must pass (AND logic)
- `any` - at least one method must pass (OR logic)
- `weighted` - weighted average of scores
- `sequential` - methods run in order, sharing results

---

## Pre-built Presets

Use presets for common evaluation scenarios:

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.presets import RAGScorer, SafetyScorer

# RAG evaluation: Faithfulness + Relevancy + Coherence
rag_methods = RAGScorer.create(provider="openai", threshold=0.7)

# Safety evaluation: Toxicity + PII + Refusal detection
safety_methods = SafetyScorer.create(check_toxicity=True, check_pii=True)

scorer = Scorer(name="full_eval", methods=rag_methods + safety_methods)
```

---

## Available Methods


| Category  | Method                   | Description                                                      |
| --------- | ------------------------ | ---------------------------------------------------------------- |
| Heuristic | `ExactMatch`             | Exact string comparison                                          |
| Heuristic | `FuzzyMatch`             | Fuzzy string similarity (Levenshtein, Jaro-Winkler, token-based) |
| Heuristic | `RegexMatch`             | Regular expression matching                                      |
| Heuristic | `ContainsAll`            | Check if output contains all keywords                            |
| Heuristic | `ContainsAny`            | Check if output contains any keyword                             |
| Heuristic | `BLEU`                   | BLEU score for translation quality                               |
| Heuristic | `ROUGE`                  | ROUGE score for summarization                                    |
| Heuristic | `TextLength`             | Validate text length                                             |
| Heuristic | `JSONValid`              | Validate JSON syntax                                             |
| Heuristic | `JSONSchema`             | Validate against JSON Schema                                     |
| Heuristic | `NoRefusal`              | Detect LLM refusals                                              |
| ML        | `EmbeddingsSimilarity`   | Semantic similarity via embeddings                               |
| ML        | `Toxicity`               | Detect toxic content                                             |
| ML        | `SentimentAnalysis`      | Analyze sentiment                                                |
| ML        | `PIIDetection`           | Detect PII/NER entities                                          |
| ML        | `LanguageDetection`      | Detect/verify language                                           |
| ML        | `ZeroShotClassification` | Zero-shot text classification                                    |
| LLM       | `Faithfulness`           | Check if response is grounded in context                         |
| LLM       | `AnswerRelevancy`        | Check if response addresses the question                         |
| LLM       | `Coherence`              | Evaluate logical flow and clarity                                |
| LLM       | `Helpfulness`            | Evaluate practical utility                                       |
| LLM       | `Correctness`            | Compare against expected answer                                  |
| LLM       | `SingleCriterionJudge`   | Custom single criterion evaluation                               |
| LLM       | `MultiCriteriaJudge`     | Multi-criteria weighted scoring                                  |
| LLM       | `PairwiseJudge`          | Compare two responses                                            |
| LLM       | `ConsensusJudge`         | Multi-LLM consensus evaluation                                   |


---

## Data Format

Sample structure for evaluation:

```python
{
    "input": "User question or prompt",
    "output": "AI response to evaluate",
    "expected": "Optional expected response",
    "context": "Optional RAG context",
    "conversation": [{"role": "user", "content": "..."}],  # Optional
    "metadata": {}  # Optional
}
```

---

## Development

```bash
git clone https://github.com/mankinds-io/mankinds-eval.git
cd mankinds-eval
pip install -e ".[dev]"

pytest              # Run tests
ruff check .        # Lint
mypy mankinds_eval  # Type check
```

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.