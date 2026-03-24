# LLM-as-Judge Methods

LLM-as-Judge methods use large language models to evaluate outputs. These methods require the `[llm]` extra.

```bash
pip install mankinds-eval[llm]
```

## Pre-built Evaluators

mankinds-eval provides pre-built LLM judges for common evaluation scenarios, inspired by RAGAS and DeepEval.

### Faithfulness

Evaluates if the response is grounded in the provided context (no hallucinations).

```python
from mankinds_eval.methods.llm import Faithfulness

method = Faithfulness(
    provider="openai",
    model="gpt-4o-mini",
    threshold=0.7,
    scale="1-5",
    strict=False,  # If True, any unsupported claim fails
)
```

**Required fields:** `input`, `output`, `context`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum normalized score to pass |
| `scale` | `str` | `"1-5"` | Rating scale: `"1-5"`, `"1-10"`, `"binary"` |
| `strict` | `bool` | `False` | Strict mode: any unsupported claim fails |

**Result metadata:**
- `unsupported_claims`: List of claims not supported by context
- `num_unsupported_claims`: Count of unsupported claims

### AnswerRelevancy

Evaluates if the response directly addresses the question.

```python
from mankinds_eval.methods.llm import AnswerRelevancy

method = AnswerRelevancy(
    provider="openai",
    threshold=0.7,
    scale="1-5",
    check_completeness=True,
)
```

**Required fields:** `input`, `output`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum normalized score to pass |
| `scale` | `str` | `"1-5"` | Rating scale |
| `check_completeness` | `bool` | `True` | Also check if all parts of question are addressed |

**Result metadata:**
- `addresses_question`: Whether response addresses the question
- `is_complete`: Whether response is complete
- `off_topic_elements`: List of off-topic content

### Coherence

Evaluates the logical flow, clarity, and consistency of the response.

```python
from mankinds_eval.methods.llm import Coherence

method = Coherence(
    provider="openai",
    threshold=0.7,
    scale="1-5",
    aspects=["logical_flow", "clarity", "consistency"],
)
```

**Required fields:** `input`, `output`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum normalized score to pass |
| `scale` | `str` | `"1-5"` | Rating scale |
| `aspects` | `list[str]` | `["logical_flow", "clarity", "consistency"]` | Aspects to evaluate |

**Available aspects:** `logical_flow`, `clarity`, `consistency`, `organization`, `conciseness`

**Result metadata:**
- `aspect_scores`: Individual scores for each aspect
- `issues`: List of coherence issues found

### Helpfulness

Evaluates the practical utility and actionability of the response.

```python
from mankinds_eval.methods.llm import Helpfulness

method = Helpfulness(
    provider="openai",
    threshold=0.7,
    scale="1-5",
    consider_context=True,
)
```

**Required fields:** `input`, `output`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum normalized score to pass |
| `scale` | `str` | `"1-5"` | Rating scale |
| `consider_context` | `bool` | `True` | Consider context when evaluating |

**Result metadata:**
- `is_actionable`: Whether response provides actionable guidance
- `is_informative`: Whether response is informative
- `practical_value`: Practical value rating (0.0-1.0)

### Correctness

Compares the response against an expected answer for factual correctness.

```python
from mankinds_eval.methods.llm import Correctness

method = Correctness(
    provider="openai",
    threshold=0.7,
    scale="1-5",
    partial_credit=True,
)
```

**Required fields:** `input`, `output`, `expected`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum normalized score to pass |
| `scale` | `str` | `"1-5"` | Rating scale |
| `partial_credit` | `bool` | `True` | Allow partial credit for partially correct answers |

**Result metadata:**
- `correct_elements`: List of correct elements
- `incorrect_elements`: List of incorrect elements
- `missing_elements`: List of missing elements

---

## Configurable Judges

## Setup

Set API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

## SingleCriterionJudge

Evaluates output against a single criterion.

```python
from mankinds_eval.methods.llm import SingleCriterionJudge

method = SingleCriterionJudge(
    provider="openai",
    model="gpt-4o",
    criterion="Is the response helpful, accurate, and well-structured?",
    scale="1-5",
    threshold=3,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider (see providers below) |
| `model` | `str` | `"gpt-4o"` | Model name |
| `criterion` | `str` | Required | Evaluation criterion/question |
| `scale` | `str` | `"1-5"` | Rating scale: `"1-5"`, `"1-10"`, `"pass/fail"` |
| `threshold` | `float` | `3` | Minimum score to pass |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `include_reasoning` | `bool` | `True` | Include explanation in result |

### Result

- `passed`: `True` if score >= threshold
- `score`: Normalized score (0.0 to 1.0)
- `raw_score`: Raw score on the specified scale
- `reasoning`: Explanation from the judge (if enabled)

## MultiCriteriaJudge

Evaluates output against multiple weighted criteria.

```python
from mankinds_eval.methods.llm import MultiCriteriaJudge

method = MultiCriteriaJudge(
    provider="openai",
    model="gpt-4o",
    criteria={
        "accuracy": {
            "description": "Is the information factually correct?",
            "weight": 0.4
        },
        "clarity": {
            "description": "Is the response clear and easy to understand?",
            "weight": 0.3
        },
        "completeness": {
            "description": "Does it fully answer the question?",
            "weight": 0.2
        },
        "tone": {
            "description": "Is the tone appropriate and professional?",
            "weight": 0.1
        },
    },
    scale="1-5",
    threshold=3,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider |
| `model` | `str` | `"gpt-4o"` | Model name |
| `criteria` | `dict` | Required | Criteria with descriptions and weights |
| `scale` | `str` | `"1-5"` | Rating scale |
| `threshold` | `float` | `3` | Minimum weighted score to pass |
| `aggregation` | `str` | `"weighted_mean"` | How to aggregate: `"weighted_mean"`, `"min"`, `"max"` |

### Result

- `passed`: `True` if aggregated score >= threshold
- `score`: Aggregated normalized score
- `criteria_scores`: Individual scores for each criterion
- `criteria_reasoning`: Explanations for each criterion

## PairwiseJudge

Compares two outputs and determines which is better.

```python
from mankinds_eval.methods.llm import PairwiseJudge

method = PairwiseJudge(
    provider="openai",
    model="gpt-4o",
    criterion="Which response is more helpful and accurate?",
    compare_to="expected",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider |
| `model` | `str` | `"gpt-4o"` | Model name |
| `criterion` | `str` | Required | Comparison criterion |
| `compare_to` | `str` | `"expected"` | Compare output to: `"expected"` or field name |
| `swap_order` | `bool` | `True` | Run comparison in both orders to reduce bias |

### Result

- `passed`: `True` if output is preferred or equal
- `score`: `1.0` (output wins), `0.5` (tie), `0.0` (output loses)
- `preference`: `"output"`, `"reference"`, or `"tie"`
- `reasoning`: Explanation for preference

## ConsensusJudge

Uses multiple judge instances and aggregates their decisions.

```python
from mankinds_eval.methods.llm import ConsensusJudge

method = ConsensusJudge(
    judges=[
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    ],
    criterion="Is the response accurate and helpful?",
    scale="1-5",
    threshold=3,
    consensus_method="majority",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `judges` | `list[dict]` | Required | List of judge configurations |
| `criterion` | `str` | Required | Evaluation criterion |
| `scale` | `str` | `"1-5"` | Rating scale |
| `threshold` | `float` | `3` | Threshold for pass/fail |
| `consensus_method` | `str` | `"majority"` | Method: `"majority"`, `"unanimous"`, `"average"` |

### Consensus Methods

- `majority`: Pass if majority of judges pass
- `unanimous`: Pass only if all judges pass
- `average`: Pass if average score meets threshold

### Result

- `passed`: `True` based on consensus method
- `score`: Average normalized score across judges
- `judge_scores`: Individual scores from each judge
- `judge_reasoning`: Explanations from each judge

## Providers

### OpenAI

```python
method = SingleCriterionJudge(
    provider="openai",
    model="gpt-4o",  # or "gpt-4o-mini", "gpt-4-turbo"
)
```

### Anthropic

```python
method = SingleCriterionJudge(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",  # or "claude-3-opus-20240229"
)
```

### Azure OpenAI

```python
method = SingleCriterionJudge(
    provider="azure",
    model="your-deployment-name",
    api_base="https://your-resource.openai.azure.com",
    api_version="2024-02-15-preview",
)
```

### Custom/Local

Using litellm for other providers:

```python
method = SingleCriterionJudge(
    provider="litellm",
    model="ollama/llama2",
    api_base="http://localhost:11434",
)
```

## Example: Comprehensive LLM Evaluation

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.llm import SingleCriterionJudge, MultiCriteriaJudge

scorer = Scorer(
    name="llm_evaluation",
    methods=[
        SingleCriterionJudge(
            provider="openai",
            criterion="Is the response factually accurate?",
            scale="pass/fail",
        ),
        MultiCriteriaJudge(
            provider="openai",
            criteria={
                "helpfulness": {"description": "Is it helpful?", "weight": 0.4},
                "safety": {"description": "Is it safe and appropriate?", "weight": 0.3},
                "relevance": {"description": "Is it relevant to the question?", "weight": 0.3},
            },
            scale="1-5",
            threshold=3,
        ),
    ],
    max_concurrent=5,  # Limit concurrent API calls
)
```

## Cost Optimization

Tips for reducing API costs:

1. **Use smaller models** for initial filtering:
   ```python
   # Use gpt-4o-mini for pre-screening
   SingleCriterionJudge(model="gpt-4o-mini", ...)
   ```

2. **Batch evaluations** with appropriate concurrency:
   ```python
   scorer = Scorer(methods=[...], max_concurrent=10)
   ```

3. **Disable reasoning** when not needed:
   ```python
   SingleCriterionJudge(include_reasoning=False, ...)
   ```

4. **Use pass/fail scale** for binary decisions:
   ```python
   SingleCriterionJudge(scale="pass/fail", ...)
   ```
