# Composite Methods

Composite methods allow you to combine multiple evaluation methods into complex evaluation pipelines. They enable sophisticated evaluation logic by aggregating results from child methods or conditionally executing different methods based on sample characteristics.

## CompositeMethod

Combines multiple evaluation methods into a single composite evaluation with configurable aggregation strategies.

```python
from mankinds_eval.methods.composite import CompositeMethod
from mankinds_eval.methods.heuristic import FuzzyMatch, ContainsAll

method1 = FuzzyMatch(threshold=0.8)
method2 = ContainsAll(substrings=["important", "keyword"])

composite = CompositeMethod(
    methods=[method1, method2],
    mode="all",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `methods` | `list[Method]` | Required | List of child methods to execute |
| `mode` | `str` | `"all"` | Aggregation mode: `"all"`, `"any"`, `"weighted"`, `"sequential"` |
| `weights` | `dict[str, float]` | `None` | Weights per method name for weighted/sequential modes |
| `threshold` | `float` | `None` | Score threshold for pass/fail in weighted/sequential modes |
| `stop_on_fail` | `bool` | `False` | Stop execution on first failure (sequential mode only) |
| `name` | `str` | `None` | Optional custom name for the composite method |

### Aggregation Modes

#### Mode: `all`

All methods must pass for the composite to pass. The score is the average of all method scores.

```python
from mankinds_eval.methods.composite import CompositeMethod
from mankinds_eval.methods.heuristic import ExactMatch, TextLength

composite = CompositeMethod(
    methods=[
        ExactMatch(),
        TextLength(min_length=10, max_length=500),
    ],
    mode="all",
)
# passed=True only if both methods pass
# score = average of both scores
```

!!! tip "Use case"
    Use `all` mode when you need multiple quality gates that must all be satisfied, such as format validation combined with content matching.

#### Mode: `any`

At least one method must pass for the composite to pass. The score is the maximum of all method scores.

```python
from mankinds_eval.methods.composite import CompositeMethod
from mankinds_eval.methods.heuristic import ContainsAny, FuzzyMatch

composite = CompositeMethod(
    methods=[
        ContainsAny(substrings=["yes", "correct", "right"]),
        FuzzyMatch(threshold=0.9, target="expected"),
    ],
    mode="any",
)
# passed=True if either method passes
# score = max of both scores
```

!!! tip "Use case"
    Use `any` mode when you have alternative success criteria, such as accepting either an exact match or a semantically similar response.

#### Mode: `weighted`

Calculates a weighted average score. Pass/fail is determined by comparing the weighted score against a threshold.

```python
from mankinds_eval.methods.composite import CompositeMethod
from mankinds_eval.methods.heuristic import FuzzyMatch, BLEU, ROUGE

composite = CompositeMethod(
    methods=[
        FuzzyMatch(threshold=0.7),
        BLEU(threshold=0.5),
        ROUGE(threshold=0.5),
    ],
    mode="weighted",
    weights={
        "FuzzyMatch": 2.0,  # Higher importance
        "BLEU": 1.0,
        "ROUGE": 1.0,
    },
    threshold=0.6,
)
# score = weighted average (FuzzyMatch counts 2x)
# passed=True if weighted score >= 0.6
```

!!! note "Default weights"
    Methods not specified in the `weights` dictionary default to a weight of `1.0`.

#### Mode: `sequential`

Like `weighted`, but enriches the sample with method results after each execution. This allows later methods to access results from earlier methods.

```python
from mankinds_eval.methods.composite import CompositeMethod
from mankinds_eval.methods.heuristic import JSONValid, JSONSchema

composite = CompositeMethod(
    methods=[
        JSONValid(),
        JSONSchema(schema={"type": "object", "required": ["name"]}),
    ],
    mode="sequential",
    threshold=1.0,
    stop_on_fail=True,  # Stop if JSON is invalid
)
# Executes methods in order
# Each method can access previous results via sample.method_results
```

!!! tip "Use case"
    Use `sequential` mode when methods depend on each other, such as validating JSON syntax before checking schema compliance.

### YAML Configuration

```yaml
scorers:
  - name: comprehensive_check
    methods:
      - type: CompositeMethod
        mode: all
        methods:
          - type: ExactMatch
            case_sensitive: false
          - type: TextLength
            min_length: 10
            max_length: 1000
```

Weighted configuration example:

```yaml
scorers:
  - name: weighted_evaluation
    methods:
      - type: CompositeMethod
        mode: weighted
        threshold: 0.7
        weights:
          FuzzyMatch: 2.0
          BLEU: 1.0
        methods:
          - type: FuzzyMatch
            threshold: 0.8
          - type: BLEU
            threshold: 0.5
```

### Result

- `passed`: Determined by aggregation mode and threshold
- `score`: Aggregated score (average, max, or weighted average)
- `reason`: Description of the aggregation result
- `metadata`:
    - `mode`: The aggregation mode used
    - `num_methods`: Total number of child methods
    - `num_executed`: Number of methods actually executed
    - `sub_results`: List of individual method results
    - `weights`: Weights used (if applicable)
    - `threshold`: Threshold used (if applicable)
    - `stopped_early`: Whether sequential execution stopped early (if applicable)

---

## ConditionalMethod

Conditionally executes one of two methods based on a condition function evaluated on the sample.

```python
from mankinds_eval.methods.composite import ConditionalMethod
from mankinds_eval.methods.heuristic import FuzzyMatch, ExactMatch
from mankinds_eval.core import Sample

def has_long_output(sample: Sample) -> bool:
    return len(sample.output) > 1000

conditional = ConditionalMethod(
    condition=has_long_output,
    if_true=FuzzyMatch(threshold=0.7),   # Use fuzzy for long outputs
    if_false=ExactMatch(),               # Use exact for short outputs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | `Callable[[Sample], bool]` | Required | Function that evaluates whether to use if_true or if_false |
| `if_true` | `Method` | Required | Method to execute when condition returns True |
| `if_false` | `Method` | `None` | Method to execute when condition returns False |
| `name` | `str` | `None` | Optional custom name for the conditional method |

### Examples

#### Basic conditional evaluation

```python
from mankinds_eval.methods.composite import ConditionalMethod
from mankinds_eval.methods.heuristic import JSONSchema, ContainsAll
from mankinds_eval.core import Sample

def expects_json(sample: Sample) -> bool:
    return "json" in sample.input.lower()

conditional = ConditionalMethod(
    condition=expects_json,
    if_true=JSONSchema(schema={"type": "object"}),
    if_false=ContainsAll(substrings=["answer"]),
)
```

#### Conditional based on sample metadata

```python
from mankinds_eval.methods.composite import ConditionalMethod
from mankinds_eval.methods.heuristic import FuzzyMatch, ROUGE
from mankinds_eval.core import Sample

def is_summarization_task(sample: Sample) -> bool:
    metadata = sample.metadata or {}
    return metadata.get("task_type") == "summarization"

conditional = ConditionalMethod(
    condition=is_summarization_task,
    if_true=ROUGE(threshold=0.5, rouge_type="rouge-l"),
    if_false=FuzzyMatch(threshold=0.8),
)
```

#### Without if_false (neutral result)

```python
from mankinds_eval.methods.composite import ConditionalMethod
from mankinds_eval.methods.heuristic import JSONValid
from mankinds_eval.core import Sample

def has_expected_json(sample: Sample) -> bool:
    return sample.expected is not None and sample.expected.startswith("{")

conditional = ConditionalMethod(
    condition=has_expected_json,
    if_true=JSONValid(),
    # No if_false - returns neutral result (score=None, passed=None)
)
```

### Result

- `passed`: Result from the executed method (or `None` if no method executed)
- `score`: Score from the executed method (or `None` if no method executed)
- `reason`: Reason from the executed method
- `metadata`:
    - `condition_result`: Boolean result of the condition evaluation
    - `executed_method`: Name of the method that was executed (or `None`)
    - `original_metadata`: Original metadata from the executed method (if any)

---

## Nesting Composite Methods

Composite methods can contain other composite methods, enabling complex evaluation hierarchies.

```python
from mankinds_eval.methods.composite import CompositeMethod, ConditionalMethod
from mankinds_eval.methods.heuristic import (
    ExactMatch, FuzzyMatch, JSONValid, JSONSchema, TextLength
)
from mankinds_eval.core import Sample

# Inner composite: format validation
format_check = CompositeMethod(
    methods=[
        JSONValid(),
        JSONSchema(schema={"type": "object", "required": ["result"]}),
    ],
    mode="all",
    name="FormatValidation",
)

# Inner composite: content quality
content_check = CompositeMethod(
    methods=[
        FuzzyMatch(threshold=0.7),
        TextLength(min_length=10),
    ],
    mode="weighted",
    weights={"FuzzyMatch": 2.0, "TextLength": 1.0},
    threshold=0.6,
    name="ContentQuality",
)

# Outer composite: combine both
comprehensive_eval = CompositeMethod(
    methods=[format_check, content_check],
    mode="all",
    name="ComprehensiveEvaluation",
)
```

!!! note "Required fields"
    Nested composites automatically compute their required fields as the union of all child methods' required fields.

---

## Best Practices

!!! tip "Choose the right aggregation mode"
    - **`all`**: Strict validation where every check must pass
    - **`any`**: Flexible validation with alternative success paths
    - **`weighted`**: Balanced scoring when some criteria matter more than others
    - **`sequential`**: Dependent validations where order matters

!!! tip "Use meaningful names"
    Provide custom `name` parameters to make results easier to understand:
    ```python
    composite = CompositeMethod(
        methods=[...],
        mode="all",
        name="SafetyAndQualityCheck",
    )
    ```

!!! tip "Set appropriate thresholds"
    For `weighted` and `sequential` modes, choose thresholds that reflect your quality requirements:
    ```python
    # Strict threshold for production
    CompositeMethod(mode="weighted", threshold=0.8, ...)
    
    # Lenient threshold for development
    CompositeMethod(mode="weighted", threshold=0.5, ...)
    ```

!!! tip "Use stop_on_fail for efficiency"
    In `sequential` mode, enable `stop_on_fail` to skip expensive evaluations when an early check fails:
    ```python
    CompositeMethod(
        methods=[cheap_syntax_check, expensive_llm_judge],
        mode="sequential",
        threshold=0.8,
        stop_on_fail=True,
    )
    ```

!!! tip "Inspect sub_results for debugging"
    Access individual method results through metadata:
    ```python
    result = await composite.evaluate(sample)
    for sub in result.metadata["sub_results"]:
        print(f"{sub['method_name']}: {sub['score']}")
    ```
