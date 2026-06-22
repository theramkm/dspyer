# ⚡ dspyer

> **Transpile stateful, imperative graph topologies into declarative, auto-optimizable DSPy modules.**


[![CI Build](https://github.com/theramkm/dspyer/actions/workflows/ci.yml/badge.svg)](https://github.com/theramkm/dspyer/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)

---

![dspyer Transpiler Flow](assets/dspyer_flow.png)

## 🎯 What is dspyer?

In 2026, manual prompt engineering is dead. We use DSPy to statistically optimize prompt weights and instructions. But mapping complex, imperative state machines (with loops, branches, and retries) to DSPy's declarative format has been notoriously difficult.

**`dspyer` solves this.** It parses stateful graphs, handles immutable state transitions, executes validation/self-correction loops, and automatically compiles them into standard `dspy.Module` classes. Your agent workflows are now ready for **zero-shot learning optimization** via DSPy teleprompters.

---

## 🚀 Quick Start in 60 Seconds

### 1. Install

For modern **uv** workflows (Recommended):
```bash
# Add as a dependency to your current project:
uv add git+https://github.com/theramkm/dspyer.git

# Or install locally in editable mode:
uv pip install -e .
```

For legacy **pip** workflows:
```bash
pip install -e .
```

### 2. Define your steps and graph

```python
import dspy
from pydantic import BaseModel, Field
from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.compiler import AgentTranspiler

# Define I/O schemas for your steps
class InputSchema(BaseModel):
    raw_text: str = Field(description="Raw query from the customer")

class OutputSchema(BaseModel):
    intent: str = Field(description="Support, Sales, or General Info")

# Declare your agent node
node = StatefulNode(
    name="Classifier",
    input_model=InputSchema,
    output_model=OutputSchema,
    instructions="Identify the customer's primary intent."
)

# Build the execution graph
graph = Graph()
graph.add_node(node)
graph.set_entry_point("Classifier")

# ⚡ Transpile to a declarative DSPy module
program = AgentTranspiler.compile(graph)

# Configure your backend model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Run the program!
result = program(raw_text="I want to upgrade my subscription plan.")
print(result)
```

---

## 💎 Elite 2026 Features

### 🔄 1. Dynamic Validation & Self-Correction Loops
If your model output fails Pydantic schema validation, `dspyer` automatically initiates a correction retry loop, generating natural-language feedback describing the validation failure, and prompting the model to repair its response.

### 🔀 2. State Conflict Resolution Merging
Execute parallel paths concurrently, then reconcile diverging dictionaries cleanly using RFC 7396 JSON Merge Patch policies:
```python
# Reconcile diverging state branches (concatenates list elements, resolves other keys)
merged = state_a.merge(state_b, policy="combine_lists")
```

### 🏎️ 3. Zero-Dependency `DirectLM` Adapter
Ditch heavy API packaging (like LiteLLM). Connect directly to Ollama, OpenAI, Anthropic, and Google Gemini with built-in async connection pooling and jittered backoff:
```python
from dspy_transpiler.compiler import DirectLM

lm = DirectLM(model="google/gemini-2.5-flash")
dspy.configure(lm=lm)
```

### 📈 4. Refinement Loss Metric Logging
Each compiled module records the number of self-correction steps and path steps taken, returning this metadata under `_metadata`:
```python
metadata = result["_metadata"]
print(f"Correction retries: {metadata['refinement_steps_taken']}")
print(f"Total steps run: {metadata['step_count']}")
```
*Use this metrics payload as a penalty term in your optimizer loss function to optimize for low latency and high accuracy.*

---

## 🗺️ Mapping Framework Topologies (LangGraph / PydanticAI)

`dspyer` compiles a custom stateful `Graph` architecture. You can easily map your existing workflows from LangGraph or PydanticAI onto `dspyer` using the following patterns:

### 1. LangGraph Nodes to StatefulNodes
In LangGraph, nodes are functions that receive state and return state patches. In `dspyer`, nodes are declared with explicit Pydantic `input_model` and `output_model` boundaries to enable DSPy-level optimizations:

```python
# 1. Declare Pydantic boundary models
class SearchInput(BaseModel):
    query: str

class SearchOutput(BaseModel):
    results: list[str]

# 2. Declare the StatefulNode
search_node = StatefulNode(
    name="WebSearch",
    input_model=SearchInput,
    output_model=SearchOutput,
    instructions="Query search engines to resolve user query details."
)

# 3. Add to Graph
graph.add_node(search_node)
```

### 2. LangGraph Conditional Edges to Router Nodes
In LangGraph, conditional routing functions route based on the state. You can translate this directly into `dspyer` using python callables as routers:

```python
def check_relevance_router(state: dict) -> str:
    if len(state.get("results", [])) > 0:
        return "proceed"
    return "retry"

graph.add_conditional_edges(
    "WebSearch",
    check_relevance_router,
    {"proceed": "SummarizerNode", "retry": "WebSearch"}
)
```

---

## 🎨 Advanced Example: Loops & Parallel Branches

Want to see complex cycles and parallel execution in action? Run our pre-packaged script:

```bash
uv run examples/run_parallel_loop.py
```

This runs a parser-router feedback loop, splits execution into concurrent sentiment-analysis and tag-extraction threads, and merges the outputs.

---

## 🛡️ License

This project is licensed under the [Apache License 2.0](LICENSE).

