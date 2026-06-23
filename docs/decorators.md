# Decorators & Custom Nodes

`dspyer` provides decorators to enforce validation contracts, manage self-correction retry loops, and register metadata to override compiler settings.

---

## 1. `@self_correcting` Decorator

The `@self_correcting` decorator wraps standard `dspy.Module` classes, predictors, or functions to enforce structured output validation and automatic model re-queries.

### Decorating plain typed Python functions

You can decorate a simple Python function to compile a dynamic signature. The function signature defines inputs, return annotations define the output schema, and the docstring serves as instructions:

```python
from dspy_transpiler import self_correcting
from pydantic import BaseModel

class CodeOutput(BaseModel):
    code: str
    explanation: str

@self_correcting(max_retries=3)
def generate_python_code(task: str) -> CodeOutput:
    """Generate high-quality, documented Python code matching the user task."""
    pass

# When called, dspyer compiles a predictor and returns a validated CodeOutput instance
result = generate_python_code(task="Write a binary search algorithm.")
print(result.code)
```

### Decorating custom `dspy.Module` classes

When applied to a class, `@self_correcting` automatically walks all child attributes during `__init__` and wraps any `dspy.Predict` or `dspy.COTS` instances:

```python
@self_correcting(schema=CodeOutput, max_retries=3)
class PythonCoder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict("task -> code, explanation")

    def forward(self, task):
        return self.generator(task=task)
```

---

## 2. `@dspyer_node` Decorator (Escape Hatch)

When compiling an entire graph structure (e.g. from an existing LangGraph workflow), `dspyer` statically analyzes function source codes to extract input accesses and return variables. 

For complex functions with dynamic logic or nested mappings, this AST parser can become brittle. The [@dspyer_node](file:///Users/ram/play/dspyer/dspy_transpiler/decorator.py) decorator acts as a developer escape hatch to explicitly define node interfaces and bypass AST analysis completely:

```python
from dspy_transpiler import dspyer_node

class SolverInput(BaseModel):
    problem: str

class SolverOutput(BaseModel):
    solution: str
    confidence: float

@dspyer_node(
    input_model=SolverInput,
    output_model=SolverOutput,
    instructions="Analyze the problem and provide a detailed solution with confidence score."
)
def solver_node_callable(state):
    # This node is compiled directly using the declared models
    # AST analysis is completely skipped during Graph compilation
    problem = state.get("problem")
    # Custom python logic ...
    return {"solution": "Parsed result", "confidence": 0.9}
```
