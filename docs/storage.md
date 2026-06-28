# Storage & Observability

`dspyer` features thread-safe pluggable logging storage adapters to record runtime metrics, compile validation reports, and capture datasets for offline training.

---

## 1. Pluggable Storage Adapters

To prevent performance blockages under concurrent LLM calls, logging writes are delegated to a storage adapter interface. By default, `dspyer` uses [FileStorageAdapter](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py) which appends lines to local files asynchronously in thread pools using `asyncio.to_thread`.

### Creating a Custom Storage Adapter

You can redirect logs to external databases (e.g. SQLite, PostgreSQL, or vector databases) by subclassing the [BaseStorageAdapter](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py) and registering it:

```python
from dspyer.utils import BaseStorageAdapter, set_storage_adapter

class MongoDBStorageAdapter(BaseStorageAdapter):
    def append_line(self, target: str, line: str) -> None:
        # target represents the log filepath/identifier
        db[target].insert_one({"payload": line})

    async def append_line_async(self, target: str, line: str) -> None:
        await db_client.async_insert(target, line)

# Register the adapter globally
set_storage_adapter(MongoDBStorageAdapter())
```

---

## 2. Validation Reports

To discover which nodes in your agent workflow are failing schemas and costing the most latency, enable validation logging:

```python
program = AgentTranspiler.compile(graph, validation_log_path="logs/validation.jsonl")
```

At any point, compile these raw JSON lines into a human-readable performance report:

```python
from dspyer.utils import generate_validation_report

report = generate_validation_report("logs/validation.jsonl")
print(report)
```

This aggregates total runs, successes, failures, retry rates, and ranks the most frequent failing Pydantic validation field keys to pinpoint prompts that need optimization.

---

## 3. Self-Correction Dataset Flywheel

Self-correction is a powerful runtime feature, but it adds latency. We can convert self-corrected runs into permanent training examples to fine-tune the model so that it succeeds on the *first* attempt.

# Recording Successful Retries

Pass `dataset_log_path` to your graph compiler or decorator. If a node fails validation initially but eventually succeeds after retries, `dspyer` logs the initial input and the final corrected output:

```python
program = AgentTranspiler.compile(graph, dataset_log_path="logs/flywheel.jsonl")
```

### Loading and Replaying for Optimizers

Load the logged examples into a standard `dspy.Example` list, ready for any DSPy teleprompter:

```python
from dspyer.utils import load_logged_dataset

trainset = load_logged_dataset(
    dataset_log_path="logs/flywheel.jsonl",
    input_keys=["query"]  # Declare which fields serve as inputs to the model
)

# Compile using any optimizer
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=my_metric)
optimized_module = optimizer.compile(program, trainset=trainset)
```

---

## 4. Observable Self-Correction Tracing

Production agent runs can be highly dynamic, making validation loops hard to debug. `dspyer` provides first-class observable tracing capabilities to make these self-correction loops transparent.

### Ambient Console Logging

You can enable ambient printing of execution traces globally without modifying a single line of code. Simply set the `DSPYER_TRACE` environment variable:

*   **`DSPYER_TRACE=1` or `DSPYER_TRACE=true`**: Prints execution details to `stderr` **only when** a self-correction retry or validation failure occurred. Happy paths that succeed on the first attempt remain silent.
*   **`DSPYER_TRACE=all`**: Prints details for every single node invocation (including happy paths).
*   **Unset or `0`/`false`**: Completely silent.

```bash
export DSPYER_TRACE=1
# Now run your application. Failed attempts and subsequent corrections will output directly to stderr.
```

### Programmatic Trace Extraction

Pass `trace=True` to the `@self_correcting` decorator, or `_trace=True` to the transpiled graph compiler program. This attaches a `SelfCorrectionTrace` instance to returning models, prediction results, or raised exceptions.

Use `dspyer.get_trace()` to extract the trace uniformly:

```python
from dspyer import get_trace, self_correcting

@self_correcting(max_retries=3, trace=True)
def query_agent(question: str) -> AnswerSchema:
    pass

try:
    result = query_agent(question="Calculate the total revenue.")
    trace = get_trace(result)
    if trace:
        print(f"Correction Retries: {trace.retries}")
        print(f"Total Duration: {trace.duration_s}s")
        # Dump full attempts payload
        print(trace.as_dict())
except Exception as err:
    trace = get_trace(err)
    if trace:
        print(f"Validation failed after {trace.retries} retries.")
        print(f"Errors found: {trace.failed_fields}")
```

> [!NOTE]
> `get_trace()` is designed to be highly resilient. If a target object is deepcopied or serialized (which can strip custom attributes), `get_trace()` falls back to a global thread-safe registry to guarantee successful trace resolution.

### Custom Observability Sinks (`on_trace` Callback)

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

### Nested Traces & Thread Safety

For nested module structures (e.g. executing several `@self_correcting` functions inside an outer compiled program), the framework dynamically links attempts to a parent trace using `ContextVar` propagation. Sub-step attempts are appended to the root trace in chronological order, formatted with their respective `[node_name]` tags.

### Pretty Print Console Structure

The pretty trace renderer displays skimmable headers showing the final status (`✓ corrected`, `✗ FAILED`, or `✓ passed`), failed fields, call duration, and auto-truncated field values (capping strings and lists/dicts at 80 characters to keep stdout clean):

```text
==================================================
           dspyer Self-Correction Trace
==================================================
Target: Synthesizer
Final Status: ✓ corrected
Failed Fields: citations
Duration: 0.45s
Attempts: 2
--------------------------------------------------

Attempt 1 [Synthesizer] (duration: 0.12s) - Status: ✗ FAILED
  Failed Fields: citations
  Pydantic Validation Error:
    Field 'citations': Answer must cite at least one source. (Value got: [])
  Outputs:
    answer = 'Apache-2.0.'
    citations = []

Attempt 2 [Synthesizer] (duration: 0.33s) - Status: ✓ SUCCESS
  Outputs:
    answer = 'Apache-2.0 [doc_1].'
    citations = ['doc_1']
==================================================
```

