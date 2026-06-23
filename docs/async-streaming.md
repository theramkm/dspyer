# Async & Streaming Pipelines

For concurrent environments and interactive user interfaces (like chat applications or web backends), `dspyer` provides async execution and granular step event streaming.

---

## 1. Async Program Execution (`aforward`)

To prevent blocking the async event loop during LLM invocation, compile your graph and execute it asynchronously via [aforward](../dspyer/compiler.py). 

`aforward` runs model predictions inside thread pools using standard async scheduling wrappers. It is fully compatible with concurrent web frameworks like FastAPI:

```python
from dspyer import AgentTranspiler

# Compile the graph
program = AgentTranspiler.compile(graph, output_model=TargetSchema)

# Execute asynchronously inside FastAPIs or other async runtimes
async def handle_request(query_str: str):
    result = await program.aforward(query=query_str)
    return {"status": "success", "data": result.model_dump()}
```

### Generic Type Parameters for Autocomplete

By passing the `output_model` parameter to `compile`, `aforward` returns a fully typed instance of your output Pydantic model. This enables modern code editors (like VS Code or Cursor) to provide static autocomplete and type hinting out-of-the-box:

```python
# The IDE automatically knows that 'result' has 'solution' and 'confidence' fields
result = await program.aforward(problem="...")
print(result.solution)      # Verified Autocomplete
print(result.confidence)    # Verified Autocomplete
```

---

## 2. Event Streaming (`astream`)

If you want to stream intermediate executions to the frontend in real-time, call the [astream](../dspyer/compiler.py) generator. 

`astream` is an async generator that yields structured dictionaries tracking the step-by-step progress of your state machine:

```python
async for event in program.astream(query="Find details on topic X"):
    event_type = event["event"]
    node_name = event.get("node")
    
    if event_type == "node_start":
        print(f"[*] Starting execution of node: {node_name}")
    
    elif event_type == "node_end":
        print(f"[+] Completed node {node_name} with state patch: {event['patch']}")
    
    elif event_type == "validation_error":
        print(f"[!] Validation failed at {node_name}: {event['error']}")
```

### Streaming Event Types

*   `node_start`: Dispatched immediately before a node executes.
*   `node_end`: Dispatched after a node successfully completes, including its validated state patch.
*   `validation_error`: Dispatched when a node output violates its Pydantic contract, triggering a self-correction retry.
