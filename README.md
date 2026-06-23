<div align="center">

# ⚡ dspyer

**Make any LLM step in your agent reliable and self-improving: typed outputs, automatic schema-validated retries, and one-call prompt optimization. Incrementally, without rewriting your stack.**

[![CI Build](https://github.com/theramkm/dspyer/actions/workflows/ci.yml/badge.svg)](https://github.com/theramkm/dspyer/actions/workflows/ci.yml)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theramkm/dspyer/blob/main/notebooks/dspyer_playground.ipynb)

</div>

---

![dspyer Hero Demo](assets/hero_demo.svg)

---

LLM nodes fail in messy ways: malformed JSON, missing fields, a citation that isn't there. The usual fix is hand-written `try/except`, re-prompting glue, and prompts you re-tune by hand every time you switch models.

**dspyer** turns an LLM step into a typed unit that **repairs its own output against a Pydantic schema** and can be **prompt-optimized with one call** by [DSPy](https://github.com/stanfordnlp/dspy). Add it to a single function with a decorator, or compile a whole multi-step agent, and drop the result straight into your existing [LangGraph](https://github.com/langchain-ai/langgraph).

### What you get

- **Self-correction loops**: when output fails Pydantic validation, dspyer auto-generates natural-language feedback and re-queries the model until it conforms (or hits your retry budget).
- **One-call prompt optimization**: compile to a standard `dspy.Module` and tune instructions + few-shot exemplars with any DSPy teleprompter (`BootstrapFewShot`, `MIPROv2`). Save the result as JSON, load it in production.
- **Hybrid by design**: deterministic/tool nodes stay native Python (fast, exact); only your LLM-reasoning nodes get compiled and optimized. No waste, no corruption.
- **Fits your stack**: Pydantic schemas, DSPy optimizers, LangGraph orchestration, OpenTelemetry tracing. Typed, tested, and incremental.

---

## Install

> **Pre-release (`0.2.0`)**: not yet on PyPI. Install from GitHub:

```bash
pip install git+https://github.com/theramkm/dspyer.git
# or:  uv add git+https://github.com/theramkm/dspyer.git
```

---

## Quickstart: self-correcting output in 30 seconds (no API key)

This runs offline. A node must return an answer **with a citation**; the model "forgets" the citation on its first try, fails validation, and dspyer repairs it automatically.

```python
import dspy
from pydantic import BaseModel, Field, field_validator
from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.compiler import AgentTranspiler, MockCompletionResult

# 1. Describe the contract you want the LLM to honor
class Query(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str = Field(description="Answer referencing the sources")
    citations: list[str] = Field(description="Sources cited, e.g. ['doc_1']")

    @field_validator("citations")
    @classmethod
    def must_cite(cls, v):
        if not v:
            raise ValueError("Answer must cite at least one source.")
        return v

# 2. Turn it into an optimizable, self-correcting node
node = StatefulNode(
    "Synthesizer", Query, RAGResponse,
    instructions="Answer the query and cite sources.",
    max_retries=3,
)
graph = Graph(); graph.add_node(node); graph.set_entry_point("Synthesizer")
program = AgentTranspiler.compile(graph)

# 3. Offline mock: omits the citation until it sees the correction feedback
class MockLM(dspy.LM):
    def __init__(self): super().__init__(model="mock")
    def forward(self, prompt=None, messages=None, **kw):
        saw_feedback = "feedback" in str(prompt or messages)
        good = '{"answer": "Apache-2.0 [doc_1].", "citations": ["doc_1"]}'
        bad  = '{"answer": "Apache-2.0.", "citations": []}'
        return MockCompletionResult(good if saw_feedback else bad, "mock")

dspy.configure(lm=MockLM())
r = program(query="What license is dspyer under?")

print("Answer:   ", r.answer)                                   # Apache-2.0 [doc_1].
print("Citations:", r.citations)                                # ['doc_1']
print("Self-correction loops:", r["_metadata"]["refinement_steps_taken"])  # 1
```

To run it for real against a live language model, run: `python examples/quickstart.py` (which automatically checks your API key and runs the self-correcting RAG flow). For the offline version, try: `python examples/run_rag_verifier.py`.

---

## How it works

dspyer compiles an agent graph, but it does **not** turn everything into an LLM call. It classifies each node:

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Your agent graph                                            │
   │                                                              │
   │   clean_text ──▶  reason  ──▶  format_output                 │
   │   (deterministic) (LLM)        (deterministic)               │
   │        │            │                 │                      │
   │     native       compiled to        native                  │
   │     Python       dspy.Module        Python                  │
   │                  + typed I/O                                 │
   │                  + self-correction                          │
   │                  + optimizable                              │
   └─────────────────────────────────────────────────────────────┘
```

Deterministic nodes run as plain Python (fast, exact, never hallucinated). Only LLM-reasoning nodes are compiled into typed `dspy.Module`s that self-correct and can be optimized. That's the whole idea: **add reliability and optimization to the parts that need it, and leave the rest alone.**

---

## Core capabilities

### 1. Self-correction

**On a single module: one decorator.** The fastest way in. Wrap any `dspy.Module`/`dspy.Predict` and bad outputs repair themselves against your schema:

```python
from dspy_transpiler import self_correcting
from pydantic import BaseModel

class SolverOutput(BaseModel):
    answer: str
    steps: list[str]

@self_correcting(schema=SolverOutput, max_retries=3)
class Solver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict("question -> answer, steps")

    def forward(self, question):
        return self.solve(question=question)
```

**On a graph: per node.** Set `max_retries` on a `StatefulNode` (see the Quickstart). Validation failures, malformed JSON, and missing fields all trigger the same repair loop.

### 2. Prompt optimization: tune, save, load

Stop hand-tuning prompts. Compile, optimize against a small labeled set with any DSPy teleprompter, then serialize the tuned instructions + exemplars for production. ([`examples/optimize_compiled_graph.py`](examples/optimize_compiled_graph.py))

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, pred, trace=None) -> bool:
    return example.sentiment.lower() == pred.sentiment.lower()

optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
optimized = optimizer.compile(program, trainset=trainset)

optimized.save_prompts("agent_config.json")   # ship this artifact
# ... in production ...
production_program.load_prompts("agent_config.json")
```

On the bundled 20-example sentiment benchmark ([`examples/benchmark.py`](examples/benchmark.py)), optimization moves accuracy **60% → 90%** with only the reasoning node tuned. *(Run under a deterministic mock LM to demonstrate the mechanism; plug in a real model for real numbers.)*

### 3. LangGraph integration

**Drop-in (recommended).** Compile a dspyer node and call it inside any LangGraph node function with no rewrite:

```python
compiled_agent = AgentTranspiler.compile(graph)   # a normal dspy.Module

def run_agent_node(state):
    pred = compiled_agent(query=state["user_query"])
    return {"agent_response": pred.answer, "citations": pred.citations}
```

**Scaffold from an existing graph (`from_langgraph`).** Translate a `StateGraph` topology into dspyer. Deterministic nodes are preserved as native passthroughs; LLM nodes you want optimized are mapped explicitly with narrow schemas:

```python
from dspy_transpiler import from_langgraph

node_mappings = {
    "Clean": StatefulNode("Clean", CleanInput, CleanOutput, instructions="Normalize the query"),
    "Solve": StatefulNode("Solve", SolveInput, SolveOutput, instructions="Answer the query"),
}
graph = from_langgraph(builder, node_mappings=node_mappings)
program = AgentTranspiler.compile(graph)
```

> Calling `from_langgraph(builder)` with **no** mappings scaffolds only the topology. Auto-generated LLM nodes are docstring-driven stubs that do **not** preserve your original function logic. Pass `node_mappings` for any node you want to run for real. (Lambdas / `partial` / C-callables can't be statically analyzed and safely fall back to native passthrough.)

### 4. Observability

Native OpenTelemetry tracing. Point it at Arize Phoenix, Langfuse, or Jaeger and every retry cycle, failed payload, and Pydantic error shows up as span attributes (`validation.failed`, `retry.{n}.failed_output`, `retry.{n}.error`). Install the extra: `pip install "dspyer[otel] @ git+https://github.com/theramkm/dspyer.git"`. A runnable setup is in [`notebooks/dspyer_playground.ipynb`](notebooks/dspyer_playground.ipynb).

---

## When to use dspyer (and when not)

**Use it if** you have a multi-step agent with LLM-reasoning nodes, you want those prompts optimized programmatically, and you want schema-validated retries without writing your own catch/re-query logic.

**Reach for something else if** your nodes are purely deterministic/tool/routing (nothing to optimize), or you just want a one-time "port my whole agent to DSPy" rewrite: an LLM-assisted rewrite is simpler for that. dspyer's value is the reusable runtime, not the translation.

---

## Feature reference

| Feature | Summary |
|---|---|
| `@self_correcting` | One-line schema-validated retry loop for any `dspy.Module` / `Predict`. |
| `StatefulNode(..., max_retries=N)` | Per-node retry budget + custom `refine_instructions`. |
| `use_cot=True` | Injects a `rationale` field for chain-of-thought without polluting your output schema; reasoning surfaces in `result["_metadata"]["rationales"]`. |
| `ImmutableState.merge(policy=...)` | Reconcile parallel branches: `last_write_wins`, `combine_lists`, or `raise`. |
| `from_langgraph(...)` | Scaffold a LangGraph `StateGraph` into a dspyer graph (hybrid native/LLM). |
| `save_prompts` / `load_prompts` | Serialize tuned instructions + exemplars to JSON and rehydrate in production. |
| `DirectLM` | Optional `dspy.BaseLM` adapter that bypasses LiteLLM at runtime with pooled `httpx` connections (header-based auth). For latency-critical paths; standard `dspy.LM` is the default. |

---

## Project status

Pre-release (`0.2.0`), actively developed. Green CI across Python 3.10–3.14, fully type-checked (`mypy`) and linted (`ruff`), with a 55-case test suite. APIs may shift before `1.0`. Issues and PRs welcome.

## License

[Apache License 2.0](LICENSE).
