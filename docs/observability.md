# Observability & Debugging

When building complex agent topologies, understanding why a model failed validation or how it corrected itself is critical. `dspyer` provides first-class observable tracing capabilities to make these runtime self-correction loops transparent.

---

## 1. Ambient Console Logging

You can enable print tracing globally to diagnose model behaviors without changing your code. Simply set the `DSPYER_TRACE` environment variable:

*   **`DSPYER_TRACE=1` or `DSPYER_TRACE=true`**: Prints execution details to `stderr` **only when** a self-correction retry or validation failure occurred. Happy paths that succeed on the first attempt remain silent.
*   **`DSPYER_TRACE=all`**: Prints details for every single node invocation (including happy paths).
*   **Unset or `0`/`false`**: Completely silent.

```bash
export DSPYER_TRACE=1
# Now run your application. Failed attempts and subsequent corrections will output directly to stderr.
```

### Example Print Format

```text
dspyer · answer · 2 attempts · 0.91s · ✓ corrected [failed fields: citations]
────────────────────────────────────────────────────────────────────────
attempt 1 [Answer]  ✗ validation failed (0.42s)
   citations       Answer must cite at least one source   got: []
   feedback sent → "Field 'citations': Answer must cite at least one source (Value got: [])"
attempt 2 [Answer]  ✓ passed (0.49s)
   text = 'Apache-2.0 [doc_1].'   citations = ['doc_1']
────────────────────────────────────────────────────────────────────────
```

---

## 2. Programmatic Trace Extraction

Pass `trace=True` to the `@self_correcting` decorator, or `_trace=True` to the transpiled graph compiler program. This attaches a `SelfCorrectionTrace` instance to returning models, prediction results, or raised exceptions.

Use `dspyer.get_trace()` to extract the trace uniformly:

```python
from dspyer import get_trace, self_correcting

@self_correcting(schema=Answer, max_retries=3, trace=True)
def answer_question(query: str) -> Answer:
    pass

try:
    result = answer_question(query="Tell me about Python.")
    trace = get_trace(result)
    if trace:
        print(f"Passed! Retries: {trace.retries}, Duration: {trace.duration_s}s")
except Exception as err:
    trace = get_trace(err)
    if trace:
        print(f"Failed! Retries taken: {trace.retries}, Errors: {trace.failed_fields}")
```

> [!NOTE]
> `get_trace()` is designed to be highly resilient. If a target object is deepcopied or serialized (which can strip custom attributes), `get_trace()` falls back to a global thread-safe registry to guarantee successful trace resolution.

---

## 3. Custom Observability Sinks (`on_trace` Callback)

To stream validation metrics into logging backends (such as Datadog, OpenTelemetry, or Langfuse), register an `on_trace` callback:

```python
def send_to_telemetry(trace):
    logging.info(
        "Self-Correction Occurred",
        extra={
            "corrected": trace.corrected,
            "retries": trace.retries,
            "attempts": trace.as_dict(),
        }
    )

@self_correcting(max_retries=3, trace=True, on_trace=send_to_telemetry)
def process_data(payload: str) -> DataSchema:
    pass
```

> [!IMPORTANT]
> The `on_trace` callback runs in isolation. Any exceptions raised inside your callback are swallowed and logged as warnings, ensuring telemetry issues never impact the core agent runtime.

---

## 4. API Reference

### `SelfCorrectionTrace`

| Member | Type | Description |
|---|---|---|
| `name` | `str` | Name of the decorated function or compiled node. |
| `attempts` | `list[Attempt]` | One entry per execution attempt, in order. |
| `duration_s` | `float` | Total wall-clock time across all attempts. |
| `corrected` | `bool` (property) | `True` if an early attempt failed but a later one succeeded. |
| `failed` | `bool` (property) | `True` if every attempt failed (retries exhausted). |
| `retries` | `int` (property) | Number of retry attempts beyond the first call. |
| `failed_fields` | `list[str]` (property) | Unique dot-notated field paths that failed validation at any point. |
| `as_dict()` | `dict` | JSON-serializable form, for logging or shipping to your own telemetry stack. |
| `pretty_string()` | `str` | The rendered, human-readable trace (same format the console printer uses). |
| `print()` | `None` | Prints `pretty_string()` to `sys.stderr`. |

### `Attempt`

Each item in `trace.attempts` captures one iteration of the self-correction loop.

| Field | Type | Description |
|---|---|---|
| `number` | `int` | 1-indexed attempt number. |
| `node_name` | `str \| None` | The node/schema this attempt ran for (used in the `[node_name]` prefix). |
| `success` | `bool` | Whether this attempt passed validation. |
| `duration_s` | `float` | Time spent on this attempt's model call. |
| `error_feedback` | `str \| None` | The natural-language feedback sent back to the model after a failure (`None` on the final successful attempt). |
| `validation_errors` | `list[ValidationErrorDetail]` | The specific fields that failed on this attempt. |
| `outputs` | `dict` | The raw outputs produced on this attempt. |

`ValidationErrorDetail` carries `loc` (field path), `msg` (the validator's message), `type` (Pydantic error type), and `input` (the offending value).

### Example

```python
from dspyer import self_correcting, get_trace

@self_correcting(schema=Answer, max_retries=3, trace=True)
def answer(question: str) -> Answer:
    """Answer the question and cite your sources."""

try:
    result = answer(question="What license is dspyer under?")
    trace = get_trace(result)
except Exception as err:
    trace = get_trace(err)   # the trace is attached to the exception too

print(f"corrected={trace.corrected} failed={trace.failed} retries={trace.retries}")
print(f"fields that failed at any point: {trace.failed_fields}")

for a in trace.attempts:
    status = "ok" if a.success else "failed"
    print(f"  attempt {a.number} [{a.node_name}] {status} ({a.duration_s:.2f}s)")
    for e in a.validation_errors:
        print(f"     {'.'.join(map(str, e.loc))}: {e.msg}  (got: {e.input!r})")

# Ship the structured form to your own stack:
my_logger.info("llm_self_correction", extra=trace.as_dict())
```

---

## 5. Validation Reporting

For batch analysis, log per-run validation outcomes, then generate a summary report of where your nodes fail:

```python
program = AgentTranspiler.compile(graph, validation_log_path="logs/validation.jsonl")
```

At any point, compile these raw JSON lines into a human-readable performance report:

```python
from dspyer.utils import generate_validation_report

print(generate_validation_report("logs/validation.jsonl"))
```

Example report:
```text
==================================================
           dspyer Batch Validation Report
==================================================

Node: Synthesizer
--------------------------------------------------
  Total Runs: 10
  Successful Runs: 8 (80.0%)
  Failed Runs: 2 (20.0%)
  Retry Rate: 40.0% (4/10 runs required retries)
  Average Retries: 0.80 per run
  Top Failing Pydantic Fields:
    - citations: 4 errors (66.7% of total errors)
    - answer: 2 errors (33.3% of total errors)

==================================================
```
