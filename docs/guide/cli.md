# CLI Reference

mankinds-eval provides a command-line interface for running evaluations without writing code.

## Installation

The CLI is installed automatically with mankinds-eval:

```bash
pip install mankinds-eval
mankinds-eval --help
```

## Commands

### version

Display the installed version:

```bash
mankinds-eval version
```

### init

Initialize a new scorer configuration file:

```bash
mankinds-eval init
```

Options:

| Option | Description |
|--------|-------------|
| `--name NAME` | Scorer name (default: my_scorer) |
| `--output FILE` | Output file path (default: scorer_config.yaml) |
| `--format FORMAT` | Format: yaml or json (default: yaml) |
| `--template TEMPLATE` | Template: basic, ml, llm, comprehensive |

Examples:

```bash
# Create basic config
mankinds-eval init --name qa_scorer --output qa_config.yaml

# Create LLM-focused config
mankinds-eval init --template llm --name llm_scorer

# Create comprehensive config with all method types
mankinds-eval init --template comprehensive --output full_config.yaml
```

### run

Run an evaluation:

```bash
mankinds-eval run --config CONFIG --data DATA [OPTIONS]
```

Required arguments:

| Argument | Description |
|----------|-------------|
| `--config FILE` | Path to scorer configuration file |
| `--data FILE` | Path to data file (CSV, JSONL, JSON) |

Optional arguments:

| Option | Description |
|--------|-------------|
| `--output FILE` | Output file path (default: results.json) |
| `--format FORMAT` | Output format: json or html (default: json) |
| `--max-concurrent N` | Override max concurrent evaluations |
| `--timeout SECONDS` | Override timeout per sample |
| `--quiet` | Suppress progress output |
| `--verbose` | Show detailed output |

Examples:

```bash
# Basic run
mankinds-eval run --config scorer.yaml --data evaluations.jsonl

# Specify output
mankinds-eval run --config scorer.yaml --data data.csv --output results.json

# Generate HTML scorecard
mankinds-eval run --config scorer.yaml --data data.jsonl --format html --output scorecard.html

# Override concurrency
mankinds-eval run --config scorer.yaml --data data.jsonl --max-concurrent 5

# Quiet mode (no progress)
mankinds-eval run --config scorer.yaml --data data.jsonl --quiet
```

### config

Manage global configuration:

```bash
mankinds-eval config SUBCOMMAND
```

Subcommands:

#### config init

Create global configuration file:

```bash
mankinds-eval config init
```

Creates `~/.mankinds_eval/config.yaml` with default settings.

#### config show

Display current configuration:

```bash
mankinds-eval config show
```

#### config set

Set a configuration value:

```bash
mankinds-eval config set KEY VALUE
```

Examples:

```bash
mankinds-eval config set llm.default_model gpt-4o-mini
mankinds-eval config set ml.device cuda
mankinds-eval config set scoring.max_concurrent 20
```

#### config get

Get a configuration value:

```bash
mankinds-eval config get KEY
```

Example:

```bash
mankinds-eval config get llm.default_model
```

#### config validate

Validate a scorer configuration file:

```bash
mankinds-eval config validate FILE
```

Example:

```bash
mankinds-eval config validate scorer_config.yaml
```

## Data File Formats

The CLI automatically detects data format based on file extension:

### CSV

```csv
input,output,expected
"What is 2+2?","4","4"
"Capital of France?","Paris","Paris"
```

Run:

```bash
mankinds-eval run --config scorer.yaml --data evaluations.csv
```

### JSONL

```json
{"input": "What is 2+2?", "output": "4", "expected": "4"}
{"input": "Capital of France?", "output": "Paris", "expected": "Paris"}
```

Run:

```bash
mankinds-eval run --config scorer.yaml --data evaluations.jsonl
```

### JSON

```json
[
  {"input": "What is 2+2?", "output": "4", "expected": "4"},
  {"input": "Capital of France?", "output": "Paris", "expected": "Paris"}
]
```

Run:

```bash
mankinds-eval run --config scorer.yaml --data evaluations.json
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Data loading error |
| 4 | Evaluation error |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MANKINDS_EVAL_CONFIG` | Path to global config file |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |

## Examples

### Complete Workflow

```bash
# 1. Initialize config
mankinds-eval init --name qa_eval --template basic

# 2. Edit scorer_config.yaml as needed
# 3. Prepare your data in evaluations.jsonl

# 4. Validate config
mankinds-eval config validate scorer_config.yaml

# 5. Run evaluation
mankinds-eval run --config scorer_config.yaml --data evaluations.jsonl --output results.json

# 6. Generate HTML report
mankinds-eval run --config scorer_config.yaml --data evaluations.jsonl --format html --output report.html
```

### Batch Processing

```bash
# Process multiple data files
for file in data/*.jsonl; do
    output="results/$(basename "$file" .jsonl)_results.json"
    mankinds-eval run --config scorer.yaml --data "$file" --output "$output"
done
```

### CI/CD Integration

```yaml
# .github/workflows/evaluate.yml
name: Evaluate
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install mankinds-eval[all]
      - run: mankinds-eval run --config scorer.yaml --data test_data.jsonl --output results.json
      - uses: actions/upload-artifact@v3
        with:
          name: results
          path: results.json
```
