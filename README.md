# ⚡ dspyer

> **Transpile stateful, cyclic agent graphs into declarative, auto-optimizable DSPy modules.**

[![CI Build](https://github.com/theramkm/dspyer/actions/workflows/ci.yml/badge.svg)](https://github.com/theramkm/dspyer/actions/workflows/ci.yml)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg?style=flat-square&logo=python)](https://github.com/theramkm/dspyer/actions)
[![Colab Playground](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theramkm/dspyer/blob/main/notebooks/dspyer_playground.ipynb)

---

![dspyer Transpiler Flow](assets/dspyer_flow.png)

## 🎯 The Paradigm Clash (Why dspyer?)

Building production-grade AI agents presents a fundamental conflict of design patterns:

1. **Agents are Stateful, Imperative, & Cyclic**: Orchestrators like **LangGraph** rely on mutable state dictionaries, complex conditional routing loop transitions (e.g., repeating a tool call on failure), and dynamic execution paths.
2. **DSPy is Declarative, Functional, & Linear**: DSPy treats LLM prompts as parameterized layers (`dspy.Predict(Signature)`). To run its powerful optimizers (**MIPROv2**, **BootstrapFewShot**) and automatically tune prompt text and few-shot exemplars, DSPy requires a linear, declarative execution trace (`dspy.Module.forward()`).

If you try to map a complex, looping agent directly to DSPy, optimization metrics break because the execution graph is dynamic and stateful.

**`dspyer` bridges this gap.** It allows you to define stateful, cyclic agent topologies using Pydantic schema boundaries, and transpiles them into standard `dspy.Module` programs. You get robust stateful execution, cyclic routing, and validator-driven self-correction, all fully compatible with DSPy's algorithmic optimization engines.

---

## 🔌 The USP: Drop-In LangGraph Upgrades

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

Copy-paste this snippet to build a 2-node intent extractor and classifier, compile it, and run it locally using a built-in Mock LM—no credentials or network calls required.

### 1. Install

> [!IMPORTANT]
> **Pre-Release Status**: `dspyer` is currently in pre-release (`0.1.0`) and is not yet published on PyPI. Always install it directly from the GitHub repository:

```bash
# Add as a dependency using uv:
uv add git+https://github.com/theramkm/dspyer.git

# Or install via pip:
pip install git+https://github.com/theramkm/dspyer.git
```

### 2. Run the Zero-Config Quickstart

```python
import dspy
from pydantic import BaseModel, Field
from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.compiler import AgentTranspiler

# 1. Define input & output schemas
class ExtractionInput(BaseModel):
    raw_text: str = Field(description="The source text to analyze")

class ExtractionOutput(BaseModel):
    user_name: str = Field(description="Name of the user extracted from text")
    rough_query: str = Field(description="User query context")

class ClassificationInput(BaseModel):
    rough_query: str = Field(description="Extracted rough query")

class ClassificationOutput(BaseModel):
    classified_intent: str = Field(description="Category intent: Support, Sales, Info")

# 2. Build graph nodes
node_extraction = StatefulNode(
    name="EntityExtractor",
    input_model=ExtractionInput,
    output_model=ExtractionOutput,
    instructions="Extract the user's name and their query context from raw text."
)

node_classification = StatefulNode(
    name="IntentClassifier",
    input_model=ClassificationInput,
    output_model=ClassificationOutput,
    instructions="Classify the intent of the rough query into: Support, Sales, or Info."
)

# 3. Build graph topology
graph = Graph()
graph.add_node(node_extraction)
graph.add_node(node_classification)
graph.set_entry_point("EntityExtractor")
graph.add_edge("EntityExtractor", "IntentClassifier")

# 4. Transpile graph into a declarative DSPy module
program = AgentTranspiler.compile(graph)

# 5. Create a Mock LM to intercept LLM calls
class QuickstartMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-quickstart-model")

    def forward(self, prompt=None, messages=None, **kwargs):
        prompt_str = str(prompt or messages)
        if "user_name" in prompt_str:
            content = '{"user_name": "Alice", "rough_query": "buy a subscription"}'
        elif "classified_intent" in prompt_str:
            content = '{"classified_intent": "Sales"}'
        else:
            content = '{}'

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
                self.model = "mock-quickstart-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)

# 6. Configure DSPy and execute
dspy.configure(lm=QuickstartMockLM())
result = program(raw_text="Hello, my name is Alice. I would like to buy a subscription.")
print(result)
```

---

## 🎯 Core Production Workflow: Optimize, Save & Load

Prompt engineering is fragile. If you upgrade your model backend (e.g. from Claude 3.5 Sonnet to GPT-4o-mini), your hardcoded prompts will fail. 

`dspyer` allows you to compile your topology, run standard DSPy optimizers to tune node instructions and few-shot exemplars on a validation dataset, serialize the optimized configuration, and rehydrate it in production.

Below is the complete lifecycle workflow (see [examples/optimize_compiled_graph.py](examples/optimize_compiled_graph.py)):

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy_transpiler import AgentTranspiler, from_langgraph

# 1. Convert your StateGraph topology to dspyer and compile
dspyer_graph = from_langgraph(langgraph_builder, node_configs=configs)
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
optimized_program.save_config("optimized_agent_config.json")

# 6. Rehydrate the configurations in a fresh production instance
fresh_graph = from_langgraph(langgraph_builder, node_configs=configs)
production_program = AgentTranspiler.compile(fresh_graph)

production_program.load_config("optimized_agent_config.json")
# production_program now executes using the optimized prompt weights!
```

---

## 🛡️ Dynamic Validation & Self-Correction Loops

If your model output fails Pydantic schema validation at runtime (e.g., missing a field, failing validation rules, or emitting invalid JSON), `dspyer` automatically initiates a self-correction retry loop:

1. **Translates the Validation Error**: Formats the Pydantic error trace into clear, human-readable natural language feedback.
2. **Invokes Node Refiner**: Sends the original inputs, the failed JSON output, and the validation error message back to the model.
3. **Self-Correction**: The model corrects the output dynamically. The cycle repeats until validation passes or `max_retries` is reached.

```python
class SearchOutput(BaseModel):
    answer: str
    citations: list[str] = Field(min_items=1, description="At least one source citation required.")

# If the LLM generates an answer with empty citations, the node will intercept
# the ValidationError, format the feedback, and invoke self-correction automatically.
search_node = StatefulNode(
    name="Search",
    input_model=SearchInput,
    output_model=SearchOutput,
    max_retries=3  # Customize retry limit per-node
)
```

To run a complete self-correcting RAG verification loop:
```bash
uv run examples/run_rag_verifier.py
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

---

## 🔄 Automated LangGraph Converter (`from_langgraph`)

`dspyer` provides a native converter to parse an existing LangGraph `StateGraph` or `CompiledStateGraph` structure directly into a `dspyer.Graph`:

```python
from langgraph.graph import StateGraph, START, END
from dspy_transpiler import from_langgraph, AgentTranspiler

# 1. Define your standard LangGraph StateGraph
builder = StateGraph(MyStateSchema)
builder.add_node("AgentNode", agent_function)  # Node docstring serves as instructions
builder.add_node("ToolNode", tool_function)
builder.add_edge(START, "AgentNode")
builder.add_conditional_edges("AgentNode", router, {"call_tool": "ToolNode", "end": END})

# 2. Convert it dynamically
# Auto-generated nodes inherit schema boundaries from MyStateSchema
dspyer_graph = from_langgraph(builder)

# 3. Compile it to an optimizable DSPy Module!
program = AgentTranspiler.compile(dspyer_graph)
```

---

## 🛡️ License

This project is licensed under the [Apache License 2.0](LICENSE).
