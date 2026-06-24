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

### Recording Successful Retries

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
