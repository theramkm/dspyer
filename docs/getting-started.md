# Getting Started

This guide walks you through installing `dspyer` and building your first self-correcting agent module.

---

## Installation

Install the stable release directly from PyPI:

```bash
pip install dspyer
# or using uv:
uv add dspyer
```

If you need the optional OpenTelemetry tracing integration or LangGraph converter dependencies, install them using:

```bash
pip install dspyer[otel,langgraph]
# or using uv:
uv add dspyer --optional otel --optional langgraph
```

---

## Core Concept: Stateful Graph Transpilation

Unlike traditional DSPy programs that require you to explicitly declare signatures and predictors upfront, `dspyer` lets you define your workflow as a state machine where each step (or node) is associated with an input schema, an output schema, and instructions.

`dspyer` compiles this topology under the hood into a single declarative [TranspiledAgentProgram](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) subclass. Every dynamic predictor is registered as a model parameter, allowing DSPy teleprompters (optimizers) to tune your prompts automatically.

---

## 5-Minute Tutorial: Building a Citation Validator

This tutorial builds an offline citation verification graph. The synthesizer node requires the generated answer to contain at least one source citation. If the model fails to return a citation on the first attempt, `dspyer` catches the validation failure, prompts the model with formatting feedback, and repeats the execution up to three times.

### 1. Define the Typing Contracts

We use standard Pydantic models to define what inputs are expected and what outputs must be returned:

```python
from pydantic import BaseModel, Field, field_validator

class Query(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str = Field(description="The synthesized answer text")
    citations: list[str] = Field(description="List of cited source document keys")

    @field_validator("citations")
    @classmethod
    def must_contain_citation(cls, v):
        if not v:
            raise ValueError("The answer must cite at least one source document (e.g. ['doc_1']).")
        return v
```

### 2. Configure the Graph and Compile

Declare a [StatefulNode](https://github.com/theramkm/dspyer/blob/main/dspyer/graph.py) representing the synthesis step, set up the [Graph](https://github.com/theramkm/dspyer/blob/main/dspyer/graph.py), and compile it:

```python
from dspyer import Graph, StatefulNode, AgentTranspiler

# Initialize a node with instructions and a validation contract
node = StatefulNode(
    name="Synthesizer",
    input_model=Query,
    output_model=RAGResponse,
    instructions="Answer the user query thoroughly. You must cite relevant sources in the citations field.",
    max_retries=3
)

# Build the topology
graph = Graph()
graph.add_node(node)
graph.set_entry_point("Synthesizer")

# Compile into an optimizable DSPy Module
program = AgentTranspiler.compile(graph)
```

### 3. Run the Program

To test this program offline, configure DSPy with a MockLM that simulates a validation error on the first call and returns a valid result on the retry:

```python
import dspy
from dspyer.compiler import MockCompletionResult

class CitationMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-citation")

    def forward(self, prompt=None, messages=None, **kwargs):
        prompt_str = str(prompt or messages)
        # Check if the prompt contains validation feedback from the previous failure
        if "feedback" in prompt_str.lower():
            # Correct answer returned during retry loop
            valid_json = '{"answer": "According to doc_1, dspyer is Apache-2.0.", "citations": ["doc_1"]}'
            return MockCompletionResult(valid_json, "mock")
        else:
            # Malformed answer (missing citations) returned on first attempt
            invalid_json = '{"answer": "dspyer is Apache-2.0.", "citations": []}'
            return MockCompletionResult(invalid_json, "mock")

# Set the global model config
dspy.configure(lm=CitationMockLM())

# Run the program
result = program(query="What license is dspyer released under?")

print("Synthesized Answer:", result.answer)
print("Extracted Citations:", result.citations)
print("Self-Correction Steps:", result["_metadata"]["refinement_steps_taken"])
```

When you run this script, it will print:
```text
Synthesized Answer: According to doc_1, dspyer is Apache-2.0.
Extracted Citations: ['doc_1']
Self-Correction Steps: 1
```
The node successfully recovered from the schema validation failure and returned compliant data!

To inspect what happened behind the scenes during this self-correction run, you can enable console tracing by setting `DSPYER_TRACE=1` in your environment (for details, see the [Observability & Debugging Guide](observability.md#1-ambient-console-logging)).
