# Heuristic Methods

Heuristic methods provide fast, deterministic evaluation based on text matching and metrics. These methods require no external APIs or ML models.

## ExactMatch

Checks if the output exactly matches the expected value.

```python
from mankinds_eval.methods.heuristic import ExactMatch

method = ExactMatch(
    case_sensitive=True,      # Default: True
    strip_whitespace=True,    # Default: True
    normalize_unicode=False,  # Default: False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `case_sensitive` | `bool` | `True` | Whether comparison is case-sensitive |
| `strip_whitespace` | `bool` | `True` | Strip leading/trailing whitespace |
| `normalize_unicode` | `bool` | `False` | Normalize Unicode characters |

### Result

- `passed`: `True` if output matches expected exactly
- `score`: `1.0` if passed, `0.0` otherwise

## RegexMatch

Matches output against a regular expression pattern.

```python
from mankinds_eval.methods.heuristic import RegexMatch

method = RegexMatch(
    pattern=r"\d{4}-\d{2}-\d{2}",  # Match date format
    flags=0,                        # Regex flags (re.IGNORECASE, etc.)
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | Required | Regular expression pattern |
| `flags` | `int` | `0` | Regex flags (e.g., `re.IGNORECASE`) |
| `full_match` | `bool` | `False` | Require full string match vs partial |

### Result

- `passed`: `True` if pattern matches
- `score`: `1.0` if passed, `0.0` otherwise
- `matches`: List of matched strings

## ContainsAll

Checks if output contains all specified substrings.

```python
from mankinds_eval.methods.heuristic import ContainsAll

method = ContainsAll(
    substrings=["error", "failed"],
    case_sensitive=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substrings` | `list[str]` | Required | Substrings that must all be present |
| `case_sensitive` | `bool` | `True` | Whether comparison is case-sensitive |

### Result

- `passed`: `True` if all substrings are found
- `score`: Proportion of substrings found (0.0 to 1.0)
- `found`: List of found substrings
- `missing`: List of missing substrings

## ContainsAny

Checks if output contains at least one of the specified substrings.

```python
from mankinds_eval.methods.heuristic import ContainsAny

method = ContainsAny(
    substrings=["yes", "correct", "right"],
    case_sensitive=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `substrings` | `list[str]` | Required | Substrings to search for |
| `case_sensitive` | `bool` | `True` | Whether comparison is case-sensitive |

### Result

- `passed`: `True` if any substring is found
- `score`: `1.0` if passed, `0.0` otherwise
- `found`: List of found substrings

## FuzzyMatch

Compares output to expected using fuzzy string matching algorithms.

```python
from mankinds_eval.methods.heuristic import FuzzyMatch

method = FuzzyMatch(
    threshold=0.8,
    algorithm="ratio",
    target="expected",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.8` | Minimum similarity score to pass (0.0-1.0) |
| `algorithm` | `str` | `"ratio"` | Algorithm: `"ratio"`, `"partial_ratio"`, `"token_sort_ratio"`, `"token_set_ratio"` |
| `target` | `str` | `"expected"` | Field to compare against: `"expected"` or `"input"` |

### Algorithms

- `ratio`: Standard Levenshtein distance ratio
- `partial_ratio`: Best partial match ratio
- `token_sort_ratio`: Sort tokens before comparing
- `token_set_ratio`: Compare unique token sets

### Result

- `passed`: `True` if similarity >= threshold
- `score`: Similarity score (0.0 to 1.0)

## BLEU

Calculates BLEU (Bilingual Evaluation Understudy) score.

```python
from mankinds_eval.methods.heuristic import BLEU

method = BLEU(
    threshold=0.5,
    max_ngram=4,
    smoothing=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.0` | Minimum BLEU score to pass |
| `max_ngram` | `int` | `4` | Maximum n-gram order |
| `smoothing` | `bool` | `True` | Apply smoothing for short texts |

### Result

- `passed`: `True` if BLEU score >= threshold
- `score`: BLEU score (0.0 to 1.0)

## ROUGE

Calculates ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.

```python
from mankinds_eval.methods.heuristic import ROUGE

method = ROUGE(
    threshold=0.5,
    rouge_type="rouge-l",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.0` | Minimum ROUGE score to pass |
| `rouge_type` | `str` | `"rouge-l"` | Type: `"rouge-1"`, `"rouge-2"`, `"rouge-l"` |
| `metric` | `str` | `"f"` | Metric: `"f"` (F1), `"p"` (precision), `"r"` (recall) |

### Result

- `passed`: `True` if ROUGE score >= threshold
- `score`: ROUGE score (0.0 to 1.0)

## TextLength

Validates that output length is within specified bounds.

```python
from mankinds_eval.methods.heuristic import TextLength

method = TextLength(
    min_length=10,
    max_length=500,
    unit="characters",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_length` | `int` | `0` | Minimum length |
| `max_length` | `int` | `None` | Maximum length (None = no limit) |
| `unit` | `str` | `"characters"` | Unit: `"characters"` or `"tokens"` |

### Result

- `passed`: `True` if length is within bounds
- `score`: `1.0` if passed, `0.0` otherwise
- `length`: Actual length of output

## WordCount

Validates word count of output.

```python
from mankinds_eval.methods.heuristic import WordCount

method = WordCount(
    min_words=5,
    max_words=100,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_words` | `int` | `0` | Minimum word count |
| `max_words` | `int` | `None` | Maximum word count (None = no limit) |

### Result

- `passed`: `True` if word count is within bounds
- `score`: `1.0` if passed, `0.0` otherwise
- `word_count`: Actual word count

## SentenceCount

Validates sentence count of output.

```python
from mankinds_eval.methods.heuristic import SentenceCount

method = SentenceCount(
    min_sentences=1,
    max_sentences=10,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_sentences` | `int` | `0` | Minimum sentence count |
| `max_sentences` | `int` | `None` | Maximum sentence count |

### Result

- `passed`: `True` if sentence count is within bounds
- `score`: `1.0` if passed, `0.0` otherwise
- `sentence_count`: Actual sentence count

## JSONValid

Validates that output is valid JSON syntax.

```python
from mankinds_eval.methods.heuristic import JSONValid

method = JSONValid(
    extract_from_markdown=True,
    allow_trailing_comma=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_from_markdown` | `bool` | `True` | Extract JSON from markdown code blocks |
| `allow_trailing_comma` | `bool` | `False` | Allow trailing commas (non-standard JSON) |

### Result

- `passed`: `True` if output is valid JSON
- `score`: `1.0` if valid, `0.0` otherwise
- `json_type`: Type of parsed JSON (`dict`, `list`, etc.)
- `extracted_from_markdown`: Whether JSON was extracted from markdown

### Example

```python
# This will pass - extracts JSON from markdown
output = '''Here is the result:
```json
{"name": "test", "value": 42}
```
'''
```

## JSONSchema

Validates output against a JSON Schema definition.

```python
from mankinds_eval.methods.heuristic import JSONSchema

method = JSONSchema(
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    },
    extract_from_markdown=True,
    strict=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `dict` | Required | JSON Schema definition |
| `extract_from_markdown` | `bool` | `True` | Extract JSON from markdown code blocks |
| `strict` | `bool` | `True` | Fail on additional properties not in schema |

### Result

- `passed`: `True` if output conforms to schema
- `score`: `1.0` if valid, `0.0` otherwise
- `errors`: List of validation error messages
- `error_paths`: JSON paths where errors occurred

### Example: Function Call Validation

```python
method = JSONSchema(
    schema={
        "type": "object",
        "properties": {
            "function": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["function", "arguments"],
    }
)
```

## NoRefusal

Detects if the LLM refused to answer the request.

```python
from mankinds_eval.methods.heuristic import NoRefusal

method = NoRefusal(
    patterns=None,          # Use default patterns
    extend_default=True,    # Extend defaults with custom patterns
    case_sensitive=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patterns` | `list[str]` | `None` | Custom regex patterns for refusal detection |
| `extend_default` | `bool` | `True` | Extend default patterns with custom ones |
| `case_sensitive` | `bool` | `False` | Case-sensitive pattern matching |

### Default Patterns

Detects common refusal phrases in multiple languages:
- English: "I'm sorry", "I cannot", "As an AI", "I'm not able to"
- French: "Je suis désolé", "Je ne peux pas"
- German: "Es tut mir leid", "Ich kann nicht"
- Spanish: "Lo siento", "No puedo"

### Result

- `passed`: `True` if no refusal detected
- `score`: `1.0` if no refusal, `0.0` if refusal detected
- `is_refusal`: Boolean indicating if refusal was detected
- `matched_patterns`: List of patterns that matched
- `matched_texts`: Actual text that matched

### Example: Custom Patterns

```python
method = NoRefusal(
    patterns=[r"(?i)contact support", r"(?i)out of scope"],
    extend_default=True,
)
```

## Combining Methods

Use multiple heuristic methods together:

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import (
    ExactMatch, FuzzyMatch, TextLength, ContainsAny
)

scorer = Scorer(
    name="comprehensive_check",
    methods=[
        ExactMatch(),
        FuzzyMatch(threshold=0.7),
        TextLength(min_length=10, max_length=1000),
        ContainsAny(substrings=["answer", "result", "solution"]),
    ]
)
```
