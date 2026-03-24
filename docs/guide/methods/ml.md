# ML Methods

Machine learning methods use pre-trained models for semantic evaluation. These methods require the `[ml]` extra.

```bash
pip install mankinds-eval[ml]
```

## EmbeddingsSimilarity

Compares output to expected/input using semantic embeddings.

```python
from mankinds_eval.methods.ml import EmbeddingsSimilarity

method = EmbeddingsSimilarity(
    threshold=0.7,
    compare="output_vs_expected",
    model="all-MiniLM-L6-v2",
    device="cpu",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum similarity score to pass |
| `compare` | `str` | `"output_vs_expected"` | Comparison mode (see below) |
| `model` | `str` | `"all-MiniLM-L6-v2"` | Sentence transformer model |
| `device` | `str` | `"cpu"` | Device: `"cpu"`, `"cuda"`, `"mps"` |

### Comparison Modes

- `output_vs_expected`: Compare output to expected answer
- `output_vs_input`: Compare output to input (for relevance)

### Result

- `passed`: `True` if similarity >= threshold
- `score`: Cosine similarity (0.0 to 1.0)

### Model Options

Popular sentence transformer models:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Medium | Better |
| `all-MiniLM-L12-v2` | 120MB | Fast | Better |

## SentimentAnalysis

Analyzes sentiment of the output.

```python
from mankinds_eval.methods.ml import SentimentAnalysis

method = SentimentAnalysis(
    expected_sentiment="positive",
    threshold=0.7,
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_sentiment` | `str` | Required | Expected sentiment: `"positive"`, `"negative"`, `"neutral"` |
| `threshold` | `float` | `0.5` | Minimum confidence for the expected sentiment |
| `model` | `str` | `"distilbert-base-uncased-finetuned-sst-2-english"` | Sentiment model |
| `device` | `str` | `"cpu"` | Device for inference |

### Result

- `passed`: `True` if detected sentiment matches expected with sufficient confidence
- `score`: Confidence score for the expected sentiment
- `detected_sentiment`: The sentiment with highest confidence
- `confidence`: Confidence scores for all sentiments

## PIIDetection

Detects personally identifiable information (PII) in output.

```python
from mankinds_eval.methods.ml import PIIDetection

method = PIIDetection(
    fail_on_detection=True,
    entity_types=["PERSON", "EMAIL", "PHONE_NUMBER", "CREDIT_CARD"],
    threshold=0.8,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fail_on_detection` | `bool` | `True` | Fail if PII is detected |
| `entity_types` | `list[str]` | All types | Types of PII to detect |
| `threshold` | `float` | `0.5` | Minimum confidence for detection |
| `model` | `str` | `"dslim/bert-base-NER"` | NER model for detection |

### Entity Types

- `PERSON`: Names of people
- `EMAIL`: Email addresses
- `PHONE_NUMBER`: Phone numbers
- `CREDIT_CARD`: Credit card numbers
- `SSN`: Social security numbers
- `ADDRESS`: Physical addresses
- `DATE_OF_BIRTH`: Birth dates

### Result

- `passed`: `True` if no PII detected (when `fail_on_detection=True`)
- `score`: `1.0` if passed, `0.0` otherwise
- `detected_entities`: List of detected PII with types and positions

## ZeroShotClassification

Classifies output into predefined categories without training.

```python
from mankinds_eval.methods.ml import ZeroShotClassification

method = ZeroShotClassification(
    labels=["technical", "casual", "formal"],
    expected_label="technical",
    threshold=0.5,
    model="facebook/bart-large-mnli",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `list[str]` | Required | Candidate labels for classification |
| `expected_label` | `str` | `None` | Expected label (pass if matches) |
| `threshold` | `float` | `0.5` | Minimum confidence for expected label |
| `model` | `str` | `"facebook/bart-large-mnli"` | Zero-shot classification model |
| `multi_label` | `bool` | `False` | Allow multiple labels |
| `device` | `str` | `"cpu"` | Device for inference |

### Result

- `passed`: `True` if top label matches expected (or threshold met)
- `score`: Confidence for expected label or top label
- `predicted_label`: Label with highest confidence
- `all_scores`: Confidence scores for all labels

## Device Selection

For GPU acceleration:

```python
# CUDA (NVIDIA GPUs)
method = EmbeddingsSimilarity(device="cuda")

# Apple Silicon
method = EmbeddingsSimilarity(device="mps")

# CPU (default)
method = EmbeddingsSimilarity(device="cpu")

# Auto-detect
method = EmbeddingsSimilarity(device="auto")
```

## Model Caching

Models are cached locally after first download. Set cache directory:

```python
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/path/to/cache"
```

## Toxicity

Detects toxic, harmful, or offensive content in the output.

```python
from mankinds_eval.methods.ml import Toxicity

method = Toxicity(
    threshold=0.5,
    model="unitary/toxic-bert",
    target="output",
    device="cpu",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | Toxicity score above which content is considered toxic |
| `model` | `str` | `"unitary/toxic-bert"` | Toxicity detection model |
| `target` | `str` | `"output"` | Field to analyze: `"output"` or `"input"` |
| `device` | `str` | `None` | Device: `"cpu"`, `"cuda"`, `"mps"`, or `None` for auto |

### Result

- `passed`: `True` if content is not toxic (toxicity below threshold)
- `score`: Inverted score (1.0 = non-toxic, 0.0 = highly toxic)
- `toxicity_score`: Raw toxicity score (0.0 to 1.0)
- `is_toxic`: Boolean indicating if toxic

## LanguageDetection

Detects and verifies the language of the output. Uses `langdetect` (included in base dependencies).

```python
from mankinds_eval.methods.ml import LanguageDetection

method = LanguageDetection(
    expected_language="en",
    confidence_threshold=0.8,
    target="output",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_language` | `str` | `None` | Expected ISO 639-1 language code (e.g., `"en"`, `"fr"`) |
| `match_input_language` | `bool` | `False` | Verify output matches input language |
| `confidence_threshold` | `float` | `0.8` | Minimum confidence for detection |
| `target` | `str` | `"output"` | Field to analyze: `"output"` or `"input"` |

### Language Codes

Common codes: `en` (English), `fr` (French), `de` (German), `es` (Spanish), `it` (Italian), `pt` (Portuguese), `zh-cn` (Chinese), `ja` (Japanese), `ko` (Korean), `ar` (Arabic), `ru` (Russian)

### Result

- `passed`: `True` if detected language matches expected (or `None` if no expected)
- `score`: Detection confidence (0.0 to 1.0)
- `detected_language`: ISO 639-1 code
- `detected_language_name`: Human-readable name
- `top_languages`: Top 3 detected languages with confidence

### Use Cases

```python
# Verify response is in English
LanguageDetection(expected_language="en")

# Verify response matches input language
LanguageDetection(match_input_language=True)

# Just detect language (no pass/fail)
LanguageDetection()
```

## Example: Content Safety Scorer

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.ml import PIIDetection, Toxicity, LanguageDetection

scorer = Scorer(
    name="content_safety",
    methods=[
        PIIDetection(fail_on_detection=True),
        Toxicity(threshold=0.5),
        LanguageDetection(expected_language="en"),
    ]
)
```
