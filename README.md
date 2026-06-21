# dspyer

An open-source Python library that transpiles stateful, imperative agent workflows (like LangGraph, CrewAI, or PydanticAI state machines) into declarative, optimizable DSPy programs (Modules, Signatures, and Pipelines).

## Features

- **Immutable State Engine**: Keeps state transitions clean and traceable using RFC 7396 JSON Merge Patch updates.
- **State Conflict Resolution**: Supports merging divergent execution paths using configurable conflict policies (`last_write_wins`, `combine_lists`, `raise`).
- **DirectLM Adapter**: Zero-dependency `dspy.LM` subclass wrapping Ollama, OpenAI, Anthropic, and Google Gemini with connection pooling and backoff.
- **Self-Correction & Refinement**: Programmatically handles model schema validation errors using automatic, natural-language feedback refinement loops.
- **Refinement tracking**: Counts correction loops during forward execution passes to allow joint-loss optimization configurations.
- **Optimizer-Ready**: Registers dynamic predictors as discoverable module properties to integrate directly with DSPy optimizers like `BootstrapFewShot` and `MIPROv2`.

## Installation

```bash
pip install .
```

## Quick Start

```python
from dspy_transpiler.graph import StatefulNode, Graph
from dspy_transpiler.compiler import AgentTranspiler
from pydantic import BaseModel, Field

# Define Step Schemas
class InputSchema(BaseModel):
    raw_text: str = Field(description="Raw source text to parse")

class OutputSchema(BaseModel):
    extracted_name: str = Field(description="Name extracted from raw text")

# Define Workflow Step
node = StatefulNode(
    name="Extractor",
    input_model=InputSchema,
    output_model=OutputSchema,
)

# Build Graph
graph = Graph()
graph.add_node(node)
graph.set_entry_point("Extractor")

# Transpile to DSPy Module
program = AgentTranspiler.compile(graph)

# Execute with state inputs
result = program(raw_text="Hello, my name is John Doe.")
print(result)
```

## Advanced Features

### 1. State Conflict Resolution Merging
You can merge two states (e.g. from parallel execution branches) using:
```python
merged_state = state_a.merge(state_b, policy="combine_lists")
```
Supported policies:
- `"last_write_wins"` (default): Replaces conflicting values with values from the merged state.
- `"combine_lists"`: Concatenates lists under the same key. Mismatches of other types fall back to `last_write_wins`.
- `"raise"`: Raises a `ValueError` if a key exists in both states with mismatching values.

### 2. Zero-Dependency DirectLM Adapter
Use the `DirectLM` class to interface with multiple LLM providers natively through DSPy without heavy packages:
```python
import dspy
from dspy_transpiler.compiler import DirectLM

# Configure DirectLM as global backend
lm = DirectLM(model="google/gemini-2.5-flash")
dspy.configure(lm=lm)
```

### 3. Refinement Loss Optimization
Each transpiled program tracks retries and step counts. The returned output dictionary contains a `_metadata` payload:
```python
result = program(raw_text="Test prompt")
metadata = result["_metadata"]

# Optimize for high-accuracy and low-latency
refinement_retries = metadata["refinement_steps_taken"]
steps_taken = metadata["step_count"]
```
