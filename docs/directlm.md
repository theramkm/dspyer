# DirectLM (Performance Runtime)

When running agent workflows in production, execution speed and connection pooling are critical. LiteLLM adds middleware layers and imports that can introduce execution overhead. `dspyer` bundles `DirectLM`, a high-performance model client runtime that bypasses LiteLLM entirely at execution time.

---

## 1. How DirectLM Works

`DirectLM` is a custom `dspy.BaseLM` subclass that wraps a connection-pooled HTTP client (`DirectClient`). It integrates directly with DSPy's global runtime, history tracking, and teleprompt optimization, but sends requests directly to provider APIs.

It features:
* **Optional HTTP Connection Pooling**: Utilizes `httpx.AsyncClient` and `httpx.Client` to keep TCP connections alive.
* **Jittered Exponential Backoff**: Automatically retries rate limits (HTTP 429) and network errors.
* **No LiteLLM Overhead**: Bypasses dynamic translations at runtime, cutting latency down to raw provider speeds.

---

## 2. Usage

To configure the global DSPy runtime to use `DirectLM`:

```python
import dspy
from dspyer import DirectLM

# Configure connection pooling to a local Ollama model
lm = DirectLM(
    model="ollama/gemma4:e2b", 
    api_base="http://localhost:11434",
    max_network_retries=3,
)
dspy.configure(lm=lm)
```

`DirectLM` parses provider prefixes automatically from the `model` string (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`, `google/gemini-1.5-pro`, or `ollama/llama3`).

---

## 3. Supported Providers

`DirectLM` natively supports direct connection protocols for:

| Provider Prefix | Environment Key | Default Endpoint |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| `google` or `gemini` | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com` |
| `ollama` | None | `http://localhost:11434` |
