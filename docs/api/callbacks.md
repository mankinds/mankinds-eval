# Callbacks

The callback system allows you to hook into the evaluation process for progress
tracking, logging, or custom integrations.

## EvaluationEvent

::: mankinds_eval.callbacks.EvaluationEvent

## Callback

Base class for implementing custom callbacks.

::: mankinds_eval.callbacks.Callback
    options:
      members:
        - on_evaluation_start
        - on_evaluation_end
        - on_sample_start
        - on_sample_complete
        - on_method_complete
        - on_error

## CallbackManager

::: mankinds_eval.callbacks.CallbackManager
    options:
      members:
        - add
        - remove
        - clear
        - dispatch

## Built-in Callbacks

### ProgressCallback

Displays a progress bar using rich.

::: mankinds_eval.callbacks.ProgressCallback

### LoggingCallback

Logs evaluation progress to stdout.

::: mankinds_eval.callbacks.LoggingCallback
