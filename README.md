<div align="center" markdown="1">

# ⚡ dspyer

**Reliable, optimizable LLM steps with zero DSPy boilerplate: typed outputs, automatic self-correction, and one-call prompt tuning.**

[![CI Build](https://github.com/theramkm/dspyer/actions/workflows/ci.yml/badge.svg)](https://github.com/theramkm/dspyer/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blueviolet?style=flat-square&logo=github)](https://theramkm.github.io/dspyer/)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theramkm/dspyer/blob/main/notebooks/dspyer_playground.ipynb)

</div>

---

![dspyer Architecture Flow](https://raw.githubusercontent.com/theramkm/dspyer/main/assets/hero_demo.svg)

---

## Why dspyer?

If you are building production agents with LangChain, LangGraph, or custom LLM API loops, you face three primary challenges:
1. **Prompt Decay**: When you upgrade models (e.g., from GPT-4o to Claude 3.5 Sonnet), your carefully engineered prompt strings fail. They need manual, tedious re-tuning.
2. **Brittle Validations**: You write verbose `try/except` loops and custom logic to catch malformed JSON and missing fields from the LLM.
3. **No Systematic Tuning**: There is no simple way to optimize prompts programmatically or automatically select the best few-shot exemplars for your specific tasks.

**Stanford DSPy** solves this by treating prompts as *parameters* that can be compiled and optimized against a dataset. However, adopting DSPy directly requires learning a complex new syntax (Signatures, Predictors, Modules) and rewriting your entire codebase.

**dspyer** acts as an ergonomic bridge: it transpiles standard Python functions, Pydantic schemas, and agent graphs into optimized `dspy.Module` instances under the hood, allowing you to drop them straight back into your existing orchestrator. You write standard, PEP 484 type-hinted Python functions; `dspyer` compiles them into optimizable `dspy.Module` objects you can hand to any DSPy teleprompter.

---

## Key Benefits

* **No vendor lock-in**: Compiles to a standard `dspy.Module`; use any DSPy optimizer and `dspy.save`/`load`.
* **Self-correction loops**: Failed Pydantic validation auto-generates feedback and re-queries the model until it conforms.
* **Telemetry and validation reports**: OpenTelemetry spans plus per-node failure summaries.
* **Dataset flywheel**: Successful self-corrections are logged as input/output pairs you can replay as a trainset.
* **`DirectLM` runtime**: Bypasses LiteLLM with persistent pooled HTTP connections.

Each is shown with runnable code under [Core Capabilities](#core-capabilities).

---

## Install

Install standard releases directly from PyPI:

```bash
pip install dspyer
# or using uv:
uv add dspyer
```

Alternatively, install the latest pre-release directly from GitHub:

```bash
pip install git+https://github.com/theramkm/dspyer.git
# or using uv:
uv add git+https://github.com/theramkm/dspyer.git
```

---

## Quickstart: Self-Correction in 30 Seconds (No API Key)

This runs completely offline using a mock model backend. The node contract requires an answer with at least one citation. The mock "forgets" the citation on the first try, fails validation, receives the correction feedback, and successfully repairs itself.

```python
import dspy
from pydantic import BaseModel, Field, field_validator
from dspyer.graph import Graph, StatefulNode
from dspyer.compiler import AgentTranspiler, MockCompletionResult

# 1. Describe the schema contract you want the LLM to honor
class Query(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str = Field(description="Answer referencing the sources")
    citations: list[str] = Field(description="Sources cited, e.g. ['doc_1']")

    @field_validator("citations")
    @classmethod
    def must_cite(cls, v):
        if not v:  # Ensure we cite at least one source
            raise ValueError("Answer must cite at least one source.")
        return v

# 2. Define an optimizable, self-correcting node
node = StatefulNode(
    "Synthesizer", Query, RAGResponse,
    instructions="Answer the query and cite sources.",
    max_retries=3,
)
graph = Graph()
graph.add_node(node)
graph.set_entry_point("Synthesizer")
program = AgentTranspiler.compile(graph)

# 3. Offline mock: configuration and run
# (Hiding MockLM details for readability; click below to expand)
```

<details>
<summary>Click to view MockLM configuration (for offline testing)</summary>

```python
class MockLM(dspy.LM):
    def __init__(self): super().__init__(model="mock")
    def forward(self, prompt=None, messages=None, **kw):
        saw_feedback = "feedback" in str(prompt or messages)
        good = '{"answer": "Apache-2.0 [doc_1].", "citations": ["doc_1"]}'
        bad  = '{"answer": "Apache-2.0.", "citations": []}'
        return MockCompletionResult(good if saw_feedback else bad, "mock")

dspy.configure(lm=MockLM())
```
</details>

```python
r = program(query="What license is dspyer under?")

print("Answer:   ", r.answer)                                   # Apache-2.0 [doc_1].
print("Citations:", r.citations)                                # ['doc_1']
print("Self-correction loops:", r["_metadata"]["refinement_steps_taken"])  # 1
```

*   **Live Run**: Run `python examples/quickstart.py` to run this against a live provider (OpenAI, Gemini, Ollama, Anthropic).
*   **Offline Example**: Try `python examples/run_rag_verifier.py` to test detailed verification logic.

---

## Core Capabilities

### 1. Zero-Boilerplate Decorator
Wrap any plain typed Python function. The parameters map to inputs, the docstring acts as instructions, and the return annotation defines the schema:

```python
from dspyer import self_correcting
from pydantic import BaseModel

class SolverOutput(BaseModel):
    answer: str
    steps: list[str]

@self_correcting(max_retries=3)
def solve(question: str) -> SolverOutput:
    """Answer the question and outline the logic steps."""
    # Body is intentionally empty; dspyer generates the call from the signature
    pass

# Returns a SolverOutput instance
result = solve(question="What is the capital of France?")
```

You can also decorate standard `dspy.Module` classes to automatically wrap nested predictors:

```python
@self_correcting(schema=SolverOutput, max_retries=3)
class Solver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict("question -> answer, steps")

    def forward(self, question):
        return self.solve(question=question)
```

### 2. Prompt Optimization (Tune, Save, Load)
Compile your transpiled program, optimize against a dataset using any DSPy teleprompter, and save the serialized config to JSON:

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, pred, trace=None) -> bool:
    return example.sentiment.lower() == pred.sentiment.lower()

optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
optimized = optimizer.compile(program, trainset=trainset)

# Save prompts
optimized.save_prompts("agent_config.json")

# Load in production
production_program.load_prompts("agent_config.json")
```

On a bundled sentiment benchmark ([`examples/benchmark.py`](https://github.com/theramkm/dspyer/blob/main/examples/benchmark.py), run with a simulated backend), optimization lifts accuracy **60% → 90%**, tuning only the reasoning node.

### 3. Orchestrator Integration (LangGraph)
You do not need to replace your orchestrator. You can compile individual `dspyer` nodes and invoke them inside existing LangGraph nodes:

```python
compiled_agent = AgentTranspiler.compile(graph)

def run_agent_node(state):
    pred = compiled_agent(query=state["user_query"])
    return {"agent_response": pred.answer, "citations": pred.citations}
```

Alternatively, scaffold an entire LangGraph `StateGraph` topology into a `dspyer.Graph` automatically. Non-LLM nodes are preserved as native Python passthroughs:

```python
from dspyer import from_langgraph

node_mappings = {
    "Clean": StatefulNode("Clean", CleanInput, CleanOutput, instructions="Normalize the query"),
    "Solve": StatefulNode("Solve", SolveInput, SolveOutput, instructions="Answer the query"),
}
graph = from_langgraph(builder, node_mappings=node_mappings)
program = AgentTranspiler.compile(graph)
```

### 4. Telemetry & Validation Reporting
Enable validation logging to capture production failure metadata:

```python
program = AgentTranspiler.compile(graph, validation_log_path="logs/validation.jsonl")
```

Generate a summary report detailing per-node error rates and failing Pydantic fields:

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

### 5. Self-Correction Dataset Flywheel
Configure `dataset_log_path` on either the `@self_correcting` decorator or during transpilation compilation to capture successful self-correction runs (saving the initial input and the final corrected output):

```python
program = AgentTranspiler.compile(graph, dataset_log_path="logs/flywheel.jsonl")
```

Then, load the logged executions using `load_logged_dataset` to dynamically generate a clean training dataset of `dspy.Example` objects:

```python
from dspyer.utils import load_logged_dataset

# We must specify which keys act as model inputs
trainset = load_logged_dataset(
    dataset_log_path="logs/flywheel.jsonl",
    input_keys=["query"]
)
```

### 6. Escape Hatch Node Decorator (`@dspyer_node`)

Avoid brittle AST static analysis on complex node callables by using the [@dspyer_node](https://github.com/theramkm/dspyer/blob/main/dspyer/decorator.py) decorator. It explicitly defines a node contract, instructions, and schemas directly on functions:

```python
from dspyer import dspyer_node

class ExtractorInput(BaseModel):
    query: str

class ExtractorOutput(BaseModel):
    entities: list[str]

@dspyer_node(
    input_model=ExtractorInput,
    output_model=ExtractorOutput,
    instructions="Extract named entities from the user query."
)
def extract_entities_node(state):
    # This node is explicitly registered with its typing contract
    # Bypasses AST static analysis during LangGraph conversion
    pass
```

### 7. Async & Streaming Pipelines

For concurrent web environments (like FastAPI), compile programs to execute asynchronously via [aforward](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) or stream intermediate events via [astream](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py):

```python
program = AgentTranspiler.compile(graph, output_model=ExtractorOutput)

# 1. Async forward call
result = await program.aforward(query="Alice and Bob went to Paris")
print(result.entities)

# 2. Async event streaming
async for event in program.astream(query="Alice and Bob went to Paris"):
    print(f"Event: {event['event']} | Node: {event.get('node')}")
```

### 8. Pluggable Storage Adapters

Register custom thread-safe storage engines for production dataset logging and validation reporting using the [BaseStorageAdapter](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py) interface. By default, it falls back to a thread-pooled, non-blocking [FileStorageAdapter](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py):

```python
from dspyer.utils import BaseStorageAdapter, set_storage_adapter

class CustomDatabaseAdapter(BaseStorageAdapter):
    def append_line(self, target: str, line: str) -> None:
        # Custom synchronous DB write
        db.insert(target, line)

    async def append_line_async(self, target: str, line: str) -> None:
        # Custom non-blocking async DB write
        await db.async_insert(target, line)

# Register custom adapter globally
set_storage_adapter(CustomDatabaseAdapter())
```

---

## Additional References

| Feature | Summary |
|---|---|
| `use_cot=True` | Injects chain-of-thought rationales dynamically without polluting output schemas. |
| `ImmutableState.merge()` | Standard merge policies (`last_write_wins`, `combine_lists`, `raise`) to reconcile parallel branches. |
| `StatefulNode` parameters | Per-node `max_retries` and custom `refine_instructions` configurations. |
| `@dspyer_node` | Bypasses graph AST parsing with explicit input/output schema metadata declarations. |
| `aforward` / `astream` | Non-blocking async execution and fine-grained graph step streaming. |
| Copy-on-Write (COW) | High-speed dictionary state patching that preserves untouched branches. |
| Pluggable Storage | Thread-safe database and custom file adapters for production telemetry log sinks. |

---

## Project Status

Stable release (`0.3.3`), actively developed. Green CI across Python 3.10 to 3.14, fully type-checked (`mypy`) and linted (`ruff`), with a 69-case test suite. Issues and PRs are welcome.

## License

[Apache License 2.0](https://github.com/theramkm/dspyer/blob/main/LICENSE).
