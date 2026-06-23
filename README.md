# ⚡ dspyer

> **dspyer is a runtime for building self-correcting, DSPy-optimizable agent nodes — typed inputs/outputs, automatic Pydantic-validated retry loops, and one-call prompt optimization — that drop straight into your existing LangGraph.**

[![CI Build](https://github.com/theramkm/dspyer/actions/workflows/ci.yml/badge.svg)](https://github.com/theramkm/dspyer/actions/workflows/ci.yml)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Colab Playground](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theramkm/dspyer/blob/main/notebooks/dspyer_playground.ipynb)

---

![dspyer Transpiler Flow](assets/dspyer_flow.png)

> [!NOTE]
> ### 💡 When to Use dspyer (and when not to)
> *   **Use it if**: You have a multi-step agent with LLM-reasoning nodes, you want those prompts optimized programmatically by DSPy, and you want schema-validated auto-retry loops without writing custom catch/re-query logic.
> *   **Don't use it for**: Purely deterministic/tool/routing nodes (since they don't have prompt weights to optimize), or a simple one-time "rewrite my whole agent in DSPy" port task (an LLM-assisted rewrite is simpler for that).

---

## 🔌 Drop-In LangGraph Upgrades

You do not need to rewrite your entire system to get the benefit of prompt optimization. The most powerful way to use `dspyer` is as a **drop-in node optimizer** inside your existing LangGraph workflows. 

You can build, transpile, and optimize a complex, multi-step sub-graph using `dspyer`, and run the compiled `dspy.Module` directly inside a single LangGraph node function:

```python
from pydantic import BaseModel, Field
from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.compiler import AgentTranspiler

# --- 1. Define and compile the optimizable agent node with dspyer ---
class AgentInput(BaseModel):
    query: str

class AgentOutput(BaseModel):
    answer: str
    citations: list[str]

# Define the node boundaries (input schema, output schema, initial instruction)
agent_node = StatefulNode(
    name="Agent",
    input_model=AgentInput,
    output_model=AgentOutput,
    instructions="Answer queries using context citations."
)

graph = Graph()
graph.add_node(agent_node)
graph.set_entry_point("Agent")

# Compile into a standard, optimizable dspy.Module
compiled_agent = AgentTranspiler.compile(graph)


# --- 2. Drop it directly into your existing LangGraph node function ---
from langgraph.graph import StateGraph

class LangGraphState(dict):
    user_query: str
    agent_response: str
    citations: list[str]

# The existing LangGraph node function
def run_agent_node(state: LangGraphState):
    # Call the compiled dspyer module like a standard python function
    prediction = compiled_agent(query=state["user_query"])
    
    # Return updates back to the LangGraph state
    return {
        "agent_response": prediction.answer,
        "citations": prediction.citations
    }

# Build and orchestrate your LangGraph workflow normally
workflow = StateGraph(LangGraphState)
workflow.add_node("agent_node", run_agent_node)
# ... add_edge, set_entry_point, compile ...
```

---

## 🚀 Try It In 10 Seconds (No API Key Required)

Copy-paste the snippets below to see self-correcting validation loops and prompt optimization configs in action locally.

### 1. Install

> [!IMPORTANT]
> **Pre-Release Status**: `dspyer` is currently in pre-release (`0.1.0`) and is not yet published on PyPI. Always install it directly from the GitHub repository:

```bash
# Add as a dependency using uv:
uv add git+https://github.com/theramkm/dspyer.git

# Or install via pip:
pip install git+https://github.com/theramkm/dspyer.git
```

### 2. Self-Correction & Validation Loops (The Quickstart)

Run this zero-config snippet to see `dspyer` enforce schema constraints, catch a missing citation error, auto-generate natural-language corrective feedback, and re-query the model to repair itself:

```python
import dspy
from pydantic import BaseModel, Field, field_validator
from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.compiler import AgentTranspiler

# 1. Define input & output schemas with strict validation rules
class UserQuery(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str = Field(description="The synthesized answer referencing the context")
    citations: list[str] = Field(description="Citations cited in the answer (e.g. ['doc_1'])")

    @field_validator("citations")
    @classmethod
    def must_have_citations(cls, val):
        if not val or len(val) == 0:
            raise ValueError("Synthesized answer must contain at least one valid source citation.")
        return val

# 2. Build the node boundaries
agent_node = StatefulNode(
    name="Synthesizer",
    input_model=UserQuery,
    output_model=RAGResponse,
    instructions="Answer queries and cite document sources.",
    max_retries=3  # Allow up to 3 correction retry loops
)

graph = Graph()
graph.add_node(agent_node)
graph.set_entry_point("Synthesizer")
program = AgentTranspiler.compile(graph)

# 3. Create a Mock LM that fails validation on the 1st call and succeeds on the 2nd (retry)
class SelfCorrectionMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-correction-model")
        self.call_count = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.call_count += 1
        prompt_str = str(prompt or messages)
        
        if "failed_output" in prompt_str or "feedback" in prompt_str:
            # Succesful correction pass (provides citation)
            content = '{"answer": "dspyer uses Apache-2.0 [doc_1].", "citations": ["doc_1"]}'
        else:
            # Wipes out source citation to trigger validation failure
            content = '{"answer": "dspyer uses Apache-2.0.", "citations": []}'

        class MockChoiceMessage:
            def __init__(self, content_str):
                self.content = content_str
                self.role = "assistant"
                self.reasoning_content = None

        class MockChoice:
            def __init__(self, content_str):
                self.message = MockChoiceMessage(content_str)
                self.finish_reason = "stop"
                self.index = 0

        class MockResult:
            def __init__(self, content_str):
                self.choices = [MockChoice(content_str)]
                self.model = "mock-correction-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)

# 4. Configure and run
dspy.configure(lm=SelfCorrectionMockLM())
result = program(query="What is the license of dspyer?")

print(f"Answer: {result.answer}")
print(f"Citations: {result.citations}")
print(f"Total correction loops run: {result['_metadata']['refinement_steps_taken']}")
```

---

## 🎬 Hero Demo & Benchmarks

To see the core value proposition of `dspyer` in action:
1. **Hybrid Optimization**: The dynamic transpiler auto-detects LLM reasoning steps vs. deterministic nodes. Only the reasoning node is compiled into a `dspy.Module` for optimization — the other nodes remain native, high-performance Python functions.
2. **Auto-Validated Self-Correction**: When the reasoning node fails schema validation, natural-language correction feedback is automatically generated to retry the model.
3. **One-Call Prompt Tuning**: Programmatic optimization tunes prompt instructions on the reasoning node using a standard teleprompter, without altering native Python code.

### 🎥 Live Terminal Run (60-Second Demo)

Here is a recording of running [examples/hero_demo.py](examples/hero_demo.py) showcasing scaffolding, compilation, auto-retry loops, and prompt tuning:

![dspyer Hero Demo](assets/hero_demo.svg)

### 📈 Metrics & Benchmarks

We run a sentiment classification benchmark on a dataset of 20 evaluation examples (see [examples/benchmark.py](examples/benchmark.py)). We measure performance before vs. after prompt optimization (using standard `BootstrapFewShot` teleprompter tuning). *(Note: This benchmark is run locally under a deterministic mock LLM simulator to demonstrate the execution mechanism and optimization behavior reliably).*

| Phase | Metric (Accuracy) | Latency / Nodes Optimized |
|---|---|---|
| **Before Optimization** | **60.0%** | Baseline |
| **After Optimization** | **90.0%** | **+30.0% Accuracy** (Only reasoning node optimized) |

---

## 🎯 Prompt Optimization Lifecycle: Tune, Save & Load

Prompt engineering is fragile. If you upgrade your model backend (e.g. from Claude 3.5 Sonnet to GPT-4o-mini), your handcoded prompts will fail. 

`dspyer` allows you to compile your topology, run standard DSPy optimizers to tune node instructions and few-shot exemplars on a validation dataset, serialize the optimized configuration, and rehydrate it in production.

Below is the complete lifecycle workflow (see [examples/optimize_compiled_graph.py](examples/optimize_compiled_graph.py)):

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy_transpiler import AgentTranspiler, from_langgraph

# 1. Scaffold your StateGraph topology to dspyer and compile
dspyer_graph = from_langgraph(langgraph_builder, node_mappings=mappings)
program = AgentTranspiler.compile(dspyer_graph)

# 2. Define your training dataset (inputs and expected outputs)
trainset = [
    dspy.Example(input_text="I am very happy.", sentiment="positive").with_inputs("input_text"),
    dspy.Example(input_text="This product is bad.", sentiment="negative").with_inputs("input_text")
]

# 3. Setup evaluation metric
def metric(example, pred, trace=None) -> bool:
    return example.sentiment.lower() == pred.sentiment.lower()

# 4. Run the DSPy prompt optimizer
optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
optimized_program = optimizer.compile(program, trainset=trainset)

# 5. Save the optimized prompt instructions & exemplars to a JSON config
optimized_program.save_prompts("optimized_agent_config.json")

# 6. Rehydrate the configurations in a fresh production instance
fresh_graph = from_langgraph(langgraph_builder, node_mappings=mappings)
production_program = AgentTranspiler.compile(fresh_graph)

production_program.load_prompts("optimized_agent_config.json")
# production_program now executes using the optimized prompt weights!
```

---

## 📊 Production-Grade Observability (Arize Phoenix)

To demystify self-correction validator loops, `dspyer` integrates natively with **OpenTelemetry**. Traces are structured to capture every retry cycle, failed JSON payload, and Pydantic exception details directly in your visualization UI (**Arize Phoenix**, **Langfuse**, or **Jaeger**).

### 1. Configure OpenTelemetry Setup
```python
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Configure Tracer pointing to collector (e.g. Phoenix at port 6006)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)

# Instrument DSPy model layers & dspyer graph routing
DSPyInstrumentor().instrument()
```

### 2. Launch Local Phoenix server
```bash
pip install arize-phoenix openinference-instrumentation-dspy opentelemetry-exporter-otlp
phoenix start
```

### 🔍 Attributes Exposed on Span Validation Failure
When a validation failure occurs, inspect the active span attributes in Phoenix:
* **`validation.failed`**: `True` indicating a validation exception occurred.
* **`validation.error.count`**: The number of fields that failed validation during this retry pass.
* **`validation.error.{index}.field`**: The exact field path that failed validation (e.g., `citations`).
* **`validation.error.{index}.message`**: Natural language reason (e.g., `List must have at least 1 item`).
* **`validation.error.{index}.input`**: The original invalid value generated by the model.
* **`retry.{attempt}.error`**: The feedback text injected back into the signature to prompt self-correction.
* **`retry.{attempt}.failed_output`**: The raw JSON payload that failed parsing/validation.

---

## 🔄 Scaffolding from an existing LangGraph (`from_langgraph`)

Instead of rewriting your state machine by hand, `dspyer` provides a native topology scaffolder to bridge a LangGraph StateGraph topology into a `dspyer` graph structure.

For a runnable, behavior-preserving result, you should map your LLM-reasoning nodes explicitly using `node_mappings` with narrow input/output Pydantic schemas. This ensures your nodes receive only the context they need and do not wipe out other state fields:

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from dspy_transpiler import from_langgraph, AgentTranspiler
from dspy_transpiler.graph import StatefulNode

# 1. Define your standard LangGraph StateGraph
builder = StateGraph(AgentState)
builder.add_node("Clean", clean_node)
builder.add_node("Solve", solve_node)
builder.add_edge(START, "Clean")
builder.add_edge("Clean", "Solve")
builder.add_edge("Solve", END)

# 2. Map nodes with explicit, narrow schemas
class CleanInput(BaseModel):
    question: str = Field(description="Raw user query")

class CleanOutput(BaseModel):
    cleaned: str = Field(description="Normalized query text")

class SolveInput(BaseModel):
    cleaned: str = Field(description="Normalized query text")

class SolveOutput(BaseModel):
    answer: str = Field(description="Final answer text")

node_mappings = {
    "Clean": StatefulNode("Clean", CleanInput, CleanOutput, instructions="Clean query"),
    "Solve": StatefulNode("Solve", SolveInput, SolveOutput, instructions="Answer query"),
}

# 3. Scaffold the graph topology, attach schemas, and compile!
graph = from_langgraph(builder, node_mappings=node_mappings)
program = AgentTranspiler.compile(graph)
```

> [!NOTE]
> **Scaffold Mode**: `from_langgraph(builder)` with no `node_mappings` scaffolds only the **graph topology** (entry point, edges, branches). Auto-generated nodes use the node's docstring as the LLM instruction and do **not** preserve your original function logic — they ignore node inputs and regenerate state from the instruction alone. For a runnable, behavior-preserving result, pass `node_mappings` with explicit per-node schemas (below).

> [!IMPORTANT]
> **Static Code Analysis Limitations**: The automatic LLM-detection relies on static AST parsing of the function source code (`inspect.getsource`). Nodes defined as lambdas, dynamic callables, `functools.partial`, or C-extensions cannot be analyzed statically and will safely default to deterministic passthrough nodes (i.e. they will execute native Python code and will not be compiled or optimized by DSPy). If you want such dynamic callables optimized, map them explicitly via `node_mappings`.

---

## 💎 Advanced Features Reference

### 🔄 1. Per-Node Configurations
Fine-tune parameters at the individual node level rather than relying on global execution constraints:
```python
node = StatefulNode(
    name="Synthesizer",
    input_model=InputModel,
    output_model=OutputModel,
    max_retries=5,                              # Local retry budget
    refine_instructions="Rewrite the text precisely focusing on formatting.", # Refinement prompt
    use_cot=True                                # Enable CoT reasoning
)
```

### 🧠 2. Opt-In Chain-of-Thought Autoinjection
Enable logical reasoning paths programmatically without polluting your core structured Pydantic models. Passing `use_cot=True` to a node:
1. Dynamically injects a `rationale: str` field in the signature.
2. Directs the LLM to output its reasoning.
3. Automatically bypasses standard Pydantic schema validation for the `rationale` field.
4. Exposes all execution node reasoning traces in the output prediction metadata: `result["_metadata"]["rationales"]`.

### 🔀 3. State Conflict Resolution Merging
When running parallel execution branches, state dictionaries can diverge. Reconcile states cleanly using `ImmutableState.merge()`:
```python
# Reconcile diverging state branches
# combine_lists: Concatenates list values; resolves other keys with last-write-wins
merged_state = state_a.merge(state_b, policy="combine_lists")
```
*Supported policies: `last_write_wins`, `combine_lists`, and `raise` (for strict validation).*

### 🏎️ 4. Direct `DirectLM` Client (Bypassing LiteLLM)
*Standard DSPy `dspy.LM` adapters are recommended by default. However, for latency-critical production environments, `DirectLM` (inheriting from `dspy.BaseLM`) provides an optimized wrapper bypassing LiteLLM entirely at execution time.*
* Reuses persistent, pooled sync/async HTTP connections (using `httpx.Limits(max_keepalive_connections=10)`).
* Protects API tokens by passing keys securely inside authentication headers rather than URL query variables.
* Integrates provider token usage statistics directly into downstream DSPy cost trackers.
```python
from dspy_transpiler.compiler import DirectLM

lm = DirectLM(model="anthropic/claude-3-5-sonnet")
dspy.configure(lm=lm)
```

### 🪄 5. Standalone `@self_correcting` Decorator
If you want standard DSPy self-correction capabilities without the Graph compilation ceremony, apply `@self_correcting` directly to `dspy.Module` classes or individual `dspy.Predict` / `dspy.ChainOfThought` instances:

```python
from dspy_transpiler import self_correcting
from pydantic import BaseModel, Field

class SolverOutput(BaseModel):
    answer: str
    confidence: float = Field(gt=0.8)

# 1. Class-level decoration (auto-wraps all Predict/ChainOfThought attributes inside)
@self_correcting(schema=SolverOutput, max_retries=3)
class SolverModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(TestSignature)

    def forward(self, question):
        return self.generate(question=question)

# 2. Instance-level decoration
predictor = dspy.Predict(TestSignature)
wrapped_predictor = self_correcting(schema=SolverOutput, max_retries=2)(predictor)
```

### 💾 6. Prompt Configuration Serialization (`save_prompts` & `load_prompts`)
Serialize node instructions and refinement instructions to a clean JSON file and load them in production:

```python
# Save optimized configuration
program.save_prompts("optimized_prompts.json")

# Rehydrate optimized configuration
program.load_prompts("optimized_prompts.json")
```

The exported JSON maps node names to their respective optimized prompts:
```json
{
    "Synthesizer": {
        "instructions": "Answer queries and cite document sources precisely.",
        "refine_instructions": "Review original inputs and correct failed_output focusing on missing citations."
    }
}
```

---

## 🛡️ License

This project is licensed under the [Apache License 2.0](LICENSE).
