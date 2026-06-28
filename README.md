<div align="center" markdown="1">

# ⚡ dspyer

**Reliable, optimizable LLM steps with zero boilerplate.** Typed outputs, automatic self-correction, and one-call prompt tuning.

[![CI Build](https://github.com/theramkm/dspyer/actions/workflows/ci.yml/badge.svg)](https://github.com/theramkm/dspyer/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blueviolet?style=flat-square&logo=github)](https://theramkm.github.io/dspyer/)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theramkm/dspyer/blob/main/notebooks/dspyer_playground.ipynb)

</div>

---

![dspyer Architecture Flow](https://raw.githubusercontent.com/theramkm/dspyer/main/assets/hero_demo.svg)

---

## The problem

Every LLM call in a real agent ends up wrapped in the same defensive code: parse the JSON, catch the missing field, re-prompt, and re-tune the prompt by hand every time you swap models. Two things make that painful:

- **Validation is manual.** You write `try/except` and retry loops to catch malformed JSON and missing fields, on every call.
- **Prompts decay.** Move from one model to another and your hand-tuned prompt quietly stops working. There's no systematic way to re-tune it from data.

[Stanford DSPy](https://github.com/stanfordnlp/dspy) fixes the second problem properly: it treats prompts as parameters you optimize against a metric, instead of strings you edit by hand. But using DSPy directly means learning Signatures, Predictors, and Modules, and restructuring your code around them.

**dspyer is a thin layer that gets you both without the rewrite.** You write a normal, type-hinted Python function. dspyer enforces the output schema, self-correcting when the model gets it wrong, and compiles the step to a standard `dspy.Module` so you can hand it to any DSPy optimizer and drop it straight back into your existing LangGraph or agent loop.

> **Honest scope:** dspyer does not reinvent optimization. The optimizer is DSPy's. dspyer's job is the runtime around it, schema-validated self-correction, observable retries, and keeping a real agent graph optimizable with far less boilerplate. If you have a single simple call and won't optimize it, you may not need dspyer at all, and that's fine.

## What you get

- **Typed, validated outputs** - wrap any LLM step in a Pydantic schema; malformed output never reaches your code.
- **Automatic self-correction** - on a validation failure, dspyer feeds the error back to the model and retries until it conforms. ([jump](#quickstart-self-correction-in-30-seconds-no-api-key))
- **Observable retries** - watch every failure, the feedback sent, and the repair, with one env var or programmatically via `get_trace()`. The part DSPy doesn't give you. ([jump](#observable-self-correction))
- **One-call prompt optimization** - each step compiles to a standard `dspy.Module`, so any DSPy optimizer can tune it; then save and load. ([jump](#prompt-optimization-tune-save-load))
- **LangGraph drop-in** - no rewrite; only your reasoning nodes get wrapped. ([jump](#drop-into-your-existing-langgraph-no-rewrite))
- **Works anywhere** - sync, async, and class forms; OpenAI / Anthropic / Gemini / local Ollama; pooled-connection `DirectLM` runtime for lower latency.

---

## Install

```bash
pip install dspyer
# or:  uv add dspyer
```

Latest pre-release from source:

```bash
pip install git+https://github.com/theramkm/dspyer.git
```

---

## Quickstart: self-correction in 30 seconds (no API key)

The smallest possible win. Decorate a normal typed function: the parameters become inputs, the docstring becomes the instructions, and the return type is the schema dspyer enforces. The mock model below "forgets" to cite a source on its first try, fails validation, and repairs itself, all offline.

```python
import dspy
from pydantic import BaseModel, Field, field_validator
from dspyer import self_correcting, MockCompletionResult

# 1. Your output contract. The validator is the reliability guarantee.
class Answer(BaseModel):
    text: str
    citations: list[str] = Field(description="Sources, e.g. ['doc_1']")

    @field_validator("citations")
    @classmethod
    def must_cite(cls, v):
        if not v:
            raise ValueError("Answer must cite at least one source.")
        return v

# 2. Decorate a plain function. No DSPy syntax, no try/except.
@self_correcting(schema=Answer, max_retries=3)
def answer(question: str) -> Answer:
    """Answer the question and cite your sources."""

# 3. Offline mock backend: returns an uncited answer first, then a cited one.
class MockLM(dspy.LM):
    def __init__(self): super().__init__(model="mock")
    def forward(self, prompt=None, messages=None, **kw):
        saw_feedback = "feedback" in str(prompt or messages)
        good = '{"text": "Apache-2.0 [doc_1].", "citations": ["doc_1"]}'
        bad  = '{"text": "Apache-2.0.",          "citations": []}'
        return MockCompletionResult(good if saw_feedback else bad, "mock")

dspy.configure(lm=MockLM())

result = answer(question="What license is dspyer under?")
print(result.text)       # Apache-2.0 [doc_1].
print(result.citations)  # ['doc_1']
```

That's the whole feature in one decorator: a typed result you can trust, with the retry-and-repair loop handled for you. To watch it happen, set `DSPYER_TRACE=1` (see [Observable self-correction](#observable-self-correction)).

To run the same thing against a real provider (OpenAI, Gemini, Anthropic, or a local Ollama model), see [`examples/quickstart.py`](https://github.com/theramkm/dspyer/blob/main/examples/quickstart.py).

---

## Core capabilities

### The decorator: sync, async, and class forms

The decorator works on async functions and on `dspy.Module` classes too.

```python
class SolverOutput(BaseModel):
    answer: str
    steps: list[str]

# async is fully supported and non-blocking
@self_correcting(max_retries=3)
async def solve(question: str) -> SolverOutput:
    """Answer the question and outline the logic steps."""

result = await solve(question="What is the capital of France?")
```

```python
# decorate a dspy.Module class to wrap its nested predictors
@self_correcting(schema=SolverOutput, max_retries=3)
class Solver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict("question -> answer, steps")

    def forward(self, question):
        return self.solve(question=question)
```

### Observable self-correction

This is the part DSPy and most agent stacks don't give you: a self-correction loop you can actually watch. When a step fails validation and repairs itself, dspyer can show you exactly what happened, what failed, what feedback it sent the model, and what came back.

**Ambient console logging (zero code changes).** Set one environment variable:

```bash
export DSPYER_TRACE=1     # print a trace ONLY when a run struggled (corrected or failed)
export DSPYER_TRACE=all   # print a trace for every run, including clean passes
# unset / 0 / false        # silent
```

Running the quickstart above with `DSPYER_TRACE=1` prints the corrected run to stderr:

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

Practical diagnostics first: happy paths are filtered out by default so your log stream stays signal, not noise.

**Programmatic access.** Pass `trace=True` to attach a `SelfCorrectionTrace` to the result (or to a raised exception), and read it back with `get_trace()`:

```python
from dspyer import get_trace

@self_correcting(schema=Answer, max_retries=3, trace=True)
def answer(question: str) -> Answer:
    """Answer the question and cite your sources."""

try:
    result = answer(question="Tell me about Python.")
    trace = get_trace(result)
    print(f"passed · retries={trace.retries} · {trace.duration_s:.2f}s")
except Exception as err:
    trace = get_trace(err)
    print(f"failed · retries={trace.retries} · fields={trace.failed_fields}")
```

`trace=True` only attaches the object; it never prints. Printing is controlled solely by `DSPYER_TRACE`.

**Route traces to your own stack.** Pass an `on_trace` callback to ship the structured trace anywhere (Datadog, Langfuse, your logger). Callbacks are isolated; an exception in your callback is swallowed and logged, never crashing the call it observed.

```python
def to_sink(trace):
    datadog.send_event(title="LLM self-correction", text=trace.pretty_string())

@self_correcting(schema=Answer, max_retries=3, trace=True, on_trace=to_sink)
def answer(question: str) -> Answer:
    """Answer the question and cite your sources."""
```

> Tracing covers the self-correction loop on sync and async calls. It is not currently attached to `astream` event streams.

### Prompt optimization (tune, save, load)

Because each step compiles to a standard `dspy.Module`, you optimize it with any DSPy teleprompter against your own metric, then save and load the result. The optimizer is DSPy's; dspyer just makes the step reachable from your graph.

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, pred, trace=None) -> bool:
    return example.sentiment.lower() == pred.sentiment.lower()

optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
optimized = optimizer.compile(program, trainset=trainset)

optimized.save_prompts("agent_config.json")     # save tuned instructions
production_program.load_prompts("agent_config.json")  # load in prod
```

> The bundled [`examples/benchmark.py`](https://github.com/theramkm/dspyer/blob/main/examples/benchmark.py) uses a **simulated backend** to illustrate the optimize-measure loop end to end (it shows a 60% -> 90% lift on a toy sentiment task). The number is illustrative, not a benchmark of real-model accuracy. Swap in a real model and your own held-out metric to measure the actual lift on your task.

### Drop into your existing LangGraph (no rewrite)

You don't replace your orchestrator. Compile individual dspyer nodes and call them inside the LangGraph nodes you already have. Your deterministic and tool nodes stay plain Python; only the reasoning nodes get wrapped.

```python
compiled_agent = AgentTranspiler.compile(graph)

def run_agent_node(state):
    pred = compiled_agent(query=state["user_query"])
    return {"agent_response": pred.answer, "citations": pred.citations}
```

You can also scaffold an existing LangGraph `StateGraph` into a `dspyer.Graph`, preserving non-LLM nodes as native passthroughs:

```python
from dspyer import from_langgraph

node_mappings = {
    "Clean": StatefulNode("Clean", CleanInput, CleanOutput, instructions="Normalize the query"),
    "Solve": StatefulNode("Solve", SolveInput, SolveOutput, instructions="Answer the query"),
}
graph = from_langgraph(builder, node_mappings=node_mappings)
program = AgentTranspiler.compile(graph)
```

### Validation reporting

Log per-run validation outcomes, then generate a summary of where your nodes fail.

```python
program = AgentTranspiler.compile(graph, validation_log_path="logs/validation.jsonl")

from dspyer import generate_validation_report
print(generate_validation_report("logs/validation.jsonl"))
```

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

### Self-correction dataset flywheel

Capture successful self-corrections as input/output pairs, then replay them as a training set, so the model's own repairs become tomorrow's few-shot examples.

```python
program = AgentTranspiler.compile(graph, dataset_log_path="logs/flywheel.jsonl")

from dspyer import load_logged_dataset
trainset = load_logged_dataset(dataset_log_path="logs/flywheel.jsonl", input_keys=["query"])
```

### Async & streaming

```python
result = await program.aforward(query="Alice and Bob went to Paris")

async for event in program.astream(query="Alice and Bob went to Paris"):
    print(f"{event['event']} · {event.get('node')}")
```

### Pluggable storage adapters

Swap the default thread-pooled file logger for your own backend by implementing `BaseStorageAdapter`.

```python
from dspyer import BaseStorageAdapter, set_storage_adapter

class CustomDatabaseAdapter(BaseStorageAdapter):
    def append_line(self, target: str, line: str) -> None:
        db.insert(target, line)
    async def append_line_async(self, target: str, line: str) -> None:
        await db.async_insert(target, line)

set_storage_adapter(CustomDatabaseAdapter())
```

### Lower-latency LM runtime (`DirectLM`)

For latency-sensitive workloads, `DirectLM` is a pooled-connection runtime that talks to providers directly and bypasses LiteLLM. It works with Ollama, OpenAI, Anthropic, and Gemini, and supports both sync and async calls.

```python
from dspyer import DirectLM
import dspy

dspy.configure(lm=DirectLM(model="ollama/llama3", api_base="http://localhost:11434"))
```

---

## More

| Feature | Summary |
|---|---|
| `use_cot=True` | Inject chain-of-thought rationales without polluting the output schema. |
| `@dspyer_node` | Declare a node's input/output schema explicitly, bypassing graph AST parsing. |
| `ImmutableState.merge()` | Reconcile parallel branches with `last_write_wins`, `combine_lists`, or `raise`. |
| `StatefulNode` params | Per-node `max_retries` and custom `refine_instructions`. |
| `DirectLM` | Pooled-connection LM runtime that bypasses LiteLLM for Ollama / OpenAI / Anthropic / Gemini. |

Full docs: [theramkm.github.io/dspyer](https://theramkm.github.io/dspyer/)

---

## Project status

Stable release: [![PyPI Version](https://img.shields.io/pypi/v/dspyer.svg?style=flat-square&color=blue)](https://pypi.org/project/dspyer/) [![Latest Release](https://img.shields.io/github/v/release/theramkm/dspyer?style=flat-square&color=blueviolet)](https://github.com/theramkm/dspyer/releases). Actively developed. Green CI across Python 3.10-3.14, fully type-checked (`mypy`) and linted (`ruff`), with a comprehensive test suite. Issues and PRs welcome.

## License

[Apache License 2.0](https://github.com/theramkm/dspyer/blob/main/LICENSE).
