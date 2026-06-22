import json
from typing import Any, Dict

import pytest
from pydantic import BaseModel, Field

from dspy_transpiler.compiler import AgentTranspiler, GraphExecutionError
from dspy_transpiler.graph import Graph, StatefulNode


# Schemas for testing
class CompilerTestInput(BaseModel):
    input_text: str = Field(description="Source text")


class CompilerTestOutput(BaseModel):
    parsed_value: int = Field(description="An integer parsed value")


class RouterOutput(BaseModel):
    next_step: str = Field(description="The key of the next node to route to")


def test_compiler_parameter_discovery():
    node_a = StatefulNode(
        name="NodeA",
        input_model=CompilerTestInput,
        output_model=CompilerTestOutput,
        instructions="Convert input_text to parsed_value",
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")

    program = AgentTranspiler.compile(graph)

    predictors = dict(program.named_predictors())
    assert f"predictor_{node_a.name}" in predictors
    assert f"refiner_{node_a.name}" in predictors


def test_compiler_forward_success_no_retry(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    class MockOutput:
        parsed_value = 100

    def mock_predict(**kwargs):
        assert kwargs["input_text"] == "hello"
        return MockOutput()

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict)

    result = program(input_text="hello")
    assert result["input_text"] == "hello"
    assert result["parsed_value"] == 100
    assert result["_metadata"] == {"refinement_steps_taken": 0, "step_count": 1}


def test_compiler_forward_with_refinement(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    class MockInvalidOutput:
        parsed_value = "not-an-int"

    class MockValidOutput:
        parsed_value = 42

    predict_calls = 0
    refine_calls = 0

    def mock_predict(**kwargs):
        nonlocal predict_calls
        predict_calls += 1
        return MockInvalidOutput()

    def mock_refine(**kwargs):
        nonlocal refine_calls
        refine_calls += 1
        return MockValidOutput()

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict)
    monkeypatch.setattr(program, "refiner_NodeA", mock_refine)

    result = program(input_text="hello", max_retries=2)

    assert predict_calls == 1
    assert refine_calls == 1
    assert result["input_text"] == "hello"
    assert result["parsed_value"] == 42
    assert result["_metadata"] == {"refinement_steps_taken": 1, "step_count": 1}


def test_compiler_forward_exhausted_retries(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    class MockInvalidOutput:
        parsed_value = "invalid"

    def mock_predict_or_refine(**kwargs):
        return MockInvalidOutput()

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict_or_refine)
    monkeypatch.setattr(program, "refiner_NodeA", mock_predict_or_refine)

    with pytest.raises(GraphExecutionError) as excinfo:
        program(input_text="hello", max_retries=2)

    assert "Execution failed at graph node 'NodeA' after 2 retries" in str(excinfo.value)


def test_compiler_forward_with_python_router(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    node_b = StatefulNode(
        name="NodeB", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )

    def mock_router(state: Dict[str, Any]) -> str:
        if state.get("parsed_value") == 10:
            return "route_b"
        return "end"

    graph = Graph()
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.set_entry_point("NodeA")

    # Register conditional edge with python callable router
    graph.add_conditional_edges("NodeA", mock_router, {"route_b": "NodeB"})

    program = AgentTranspiler.compile(graph)

    class MockOutputA:
        parsed_value = 10

    class MockOutputB:
        parsed_value = 99

    monkeypatch.setattr(program, "predictor_NodeA", lambda **kw: MockOutputA())
    monkeypatch.setattr(program, "predictor_NodeB", lambda **kw: MockOutputB())

    result = program(input_text="hello")
    # Verified execution went NodeA -> Router -> NodeB
    assert result["parsed_value"] == 99


def test_compiler_forward_with_node_router(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    # LLM Router Node
    router_node = StatefulNode(
        name="LLMRouter", input_model=CompilerTestInput, output_model=RouterOutput
    )
    node_b = StatefulNode(
        name="NodeB", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )

    graph = Graph()
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.set_entry_point("NodeA")

    # Semantic node routing
    graph.add_conditional_edges("NodeA", router_node, {"route_b": "NodeB"})

    program = AgentTranspiler.compile(graph)

    class MockOutputA:
        parsed_value = 10

    class MockOutputRouter:
        next_step = "route_b"

    class MockOutputB:
        parsed_value = 500

    monkeypatch.setattr(program, "predictor_NodeA", lambda **kw: MockOutputA())
    monkeypatch.setattr(program, "predictor_LLMRouter", lambda **kw: MockOutputRouter())
    monkeypatch.setattr(program, "predictor_NodeB", lambda **kw: MockOutputB())

    result = program(input_text="hello")
    # Verify execution ran NodeA -> Router Node execution -> NodeB
    assert result["parsed_value"] == 500
    assert result["next_step"] == "route_b"


def test_compiler_forward_loop_limit(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )

    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    # Create self-loop: NodeA -> NodeA
    graph.add_edge("NodeA", "NodeA")

    program = AgentTranspiler.compile(graph)

    class MockOutput:
        parsed_value = 1

    monkeypatch.setattr(program, "predictor_NodeA", lambda **kw: MockOutput())

    # 1. Test raise limit policy
    with pytest.raises(RuntimeError) as excinfo:
        program(input_text="hello", max_steps=5, on_loop_limit="raise")
    assert "Graph execution exceeded max_steps limit of 5" in str(excinfo.value)

    # 2. Test return limit policy
    result = program(input_text="hello", max_steps=5, on_loop_limit="return")
    assert result["input_text"] == "hello"
    assert result["parsed_value"] == 1
    assert result["_metadata"] == {"refinement_steps_taken": 0, "step_count": 6}


def test_json_repair_clean_markdown():
    from dspy_transpiler.compiler import repair_and_parse_json

    raw = 'Here is the result:\n```json\n{"user_name": "Alice", "age": 30}\n```\nHope it helps!'
    parsed = repair_and_parse_json(raw)
    assert parsed == {"user_name": "Alice", "age": 30}


def test_json_repair_truncated_bracket():
    from dspy_transpiler.compiler import repair_and_parse_json

    # Truncated dictionary
    raw = '{"name": "Alice", "hobbies": ["reading", "coding"'
    parsed = repair_and_parse_json(raw)
    assert parsed == {"name": "Alice", "hobbies": ["reading", "coding"]}


def test_json_repair_truncated_quote():
    from dspy_transpiler.compiler import repair_and_parse_json

    # Cut off in the middle of a string value
    raw = '{"name": "Alice", "status": "workin'
    parsed = repair_and_parse_json(raw)
    assert parsed == {"name": "Alice", "status": "workin"}


def test_json_repair_trailing_comma():
    from dspy_transpiler.compiler import repair_and_parse_json

    # Cut off after a value and a comma
    raw = '{"id": 101, "tags": ["a", "b"], '
    parsed = repair_and_parse_json(raw)
    assert parsed == {"id": 101, "tags": ["a", "b"]}


def test_direct_client_payload_formatting():
    from dspy_transpiler.compiler import DirectClient

    # 1. Ollama formatting
    client_ollama = DirectClient(provider="ollama", model="llama3")
    url, headers, payload = client_ollama._get_request_details(
        prompt="hi", system_prompt="be polite"
    )
    assert url == "http://localhost:11434/api/chat"
    assert payload["messages"][0] == {"role": "system", "content": "be polite"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}

    # 2. OpenAI formatting
    client_openai = DirectClient(provider="openai", model="gpt-5.5", api_key="sk-test")
    url, headers, payload = client_openai._get_request_details(prompt="hi")
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"

    # 3. Anthropic formatting
    client_claude = DirectClient(provider="anthropic", model="claude-4.8", api_key="sk-ant")
    url, headers, payload = client_claude._get_request_details(
        prompt="hi", system_prompt="be concise"
    )
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-ant"
    assert payload["system"] == "be concise"

    # 4. Google formatting
    client_gemini = DirectClient(provider="google", model="gemini-3.5-flash", api_key="ai-gemini")
    url, headers, payload = client_gemini._get_request_details(
        prompt="hi", system_prompt="think first"
    )
    assert "gemini-3.5-flash" in url
    assert "key=" not in url
    assert headers["x-goog-api-key"] == "ai-gemini"
    assert payload["systemInstruction"]["parts"][0]["text"] == "think first"


def test_direct_client_mock_execution(monkeypatch):
    import urllib.request

    from dspy_transpiler.compiler import DirectClient

    client = DirectClient(provider="openai", model="gpt-5.5", api_key="sk-mock")

    # Mock httpx response if httpx is used
    class MockHttpxResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Mocked response content"}}]}

    # Mock urllib response if urllib is fallback
    class MockUrllibResponse:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "Mocked response content"}}]}
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import dspy_transpiler.compiler as compiler

    if compiler.HAS_HTTPX:
        import httpx

        monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: MockHttpxResponse())

        # Async mock
        async def mock_async_post(*a, **k):
            return MockHttpxResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_async_post)
    else:
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: MockUrllibResponse())

    # Verify sync call
    res_sync = client.generate_sync("hello")
    assert res_sync == "Mocked response content"

    # Verify async call
    import asyncio

    res_async = asyncio.run(client.generate_async("hello"))
    assert res_async == "Mocked response content"


def test_direct_lm_adapter_execution(monkeypatch):
    from dspy_transpiler.compiler import DirectLM, MockCompletionResult

    # Instantiate DirectLM
    lm = DirectLM(model="openai/gpt-4o-mini", api_key="sk-test")
    assert lm.model == "openai/gpt-4o-mini"
    assert lm.client.provider == "openai"
    assert lm.client.model == "gpt-4o-mini"

    # Mock DirectClient sync & async generate methods
    def mock_generate_sync(prompt, system_prompt=None):
        assert prompt == "What is the capital of France?"
        assert system_prompt == "Be helpful"
        return '{"capital": "Paris"}'

    async def mock_generate_async(prompt, system_prompt=None):
        assert prompt == "What is the capital of France?"
        assert system_prompt == "Be helpful"
        return '{"capital": "Paris"}'

    monkeypatch.setattr(lm.client, "generate_sync", mock_generate_sync)
    monkeypatch.setattr(lm.client, "generate_async", mock_generate_async)

    # 1. Verify sync forward call
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    res_sync = lm.forward(messages=messages)
    assert isinstance(res_sync, MockCompletionResult)
    assert res_sync.choices[0].message.content == '{"capital": "Paris"}'
    assert res_sync.model == "openai/gpt-4o-mini"

    # 2. Verify async forward call
    import asyncio

    res_async = asyncio.run(lm.aforward(messages=messages))
    assert isinstance(res_async, MockCompletionResult)
    assert res_async.choices[0].message.content == '{"capital": "Paris"}'


def test_compiler_refinement_steps_taken(monkeypatch):
    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    class MockInvalidOutput:
        parsed_value = "not-an-int"

    class MockValidOutput:
        parsed_value = 42

    predict_calls = 0
    refine_calls = 0

    def mock_predict(**kwargs):
        nonlocal predict_calls
        predict_calls += 1
        return MockInvalidOutput()

    def mock_refine(**kwargs):
        nonlocal refine_calls
        refine_calls += 1
        if refine_calls == 1:
            return MockInvalidOutput()  # second failure, so another retry
        return MockValidOutput()  # final success

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict)
    monkeypatch.setattr(program, "refiner_NodeA", mock_refine)

    result = program(input_text="hello", max_retries=3)

    assert predict_calls == 1
    assert refine_calls == 2
    assert program.refinement_steps_taken == 2
    assert result["_metadata"]["refinement_steps_taken"] == 2
    assert result["_metadata"]["step_count"] == 1


def test_compiler_prompt_optimization_compatibility(monkeypatch):
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    from dspy_transpiler.compiler import AgentTranspiler, DirectLM

    node_a = StatefulNode(
        name="NodeA",
        input_model=CompilerTestInput,
        output_model=CompilerTestOutput,
        instructions="Format text to int value",
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    lm = DirectLM(model="openai/gpt-4o-mini", api_key="sk-test")

    mock_responses = ['{"parsed_value": 123}', '{"parsed_value": 456}']
    resp_idx = 0

    def mock_generate_sync(prompt, system_prompt=None):
        nonlocal resp_idx
        res = mock_responses[resp_idx % len(mock_responses)]
        resp_idx += 1
        return res

    monkeypatch.setattr(lm.client, "generate_sync", mock_generate_sync)
    dspy.configure(lm=lm)

    trainset = [
        dspy.Example(input_text="abc", parsed_value=123).with_inputs("input_text"),
        dspy.Example(input_text="def", parsed_value=456).with_inputs("input_text"),
    ]

    def simple_metric(example, pred, trace=None) -> bool:
        try:
            return int(example.parsed_value) == int(pred.parsed_value)
        except Exception:
            return False

    optimizer = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1)
    optimized_program = optimizer.compile(program, trainset=trainset)

    assert optimized_program is not None
    assert len(list(optimized_program.named_predictors())) > 0


def test_compiler_concurrency_isolation(monkeypatch):
    import concurrent.futures

    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    class MockInvalidOutput:
        parsed_value = "not-an-int"

    class MockValidOutput:
        parsed_value = 42

    def mock_predict(**kwargs):
        return MockInvalidOutput()

    def mock_refine(**kwargs):
        return MockValidOutput()

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict)
    monkeypatch.setattr(program, "refiner_NodeA", mock_refine)

    def run_thread(input_val):
        res = program(input_text=input_val, max_retries=1)
        return res["_metadata"]["refinement_steps_taken"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_thread, f"input_{i}") for i in range(5)]
        results = [f.result() for f in futures]

    # Each execution thread should register exactly 1 refinement step without crosstalk
    for r in results:
        assert r == 1


def test_direct_client_pooling_lifecycle():
    import asyncio

    import httpx

    from dspy_transpiler.compiler import DirectClient

    client = DirectClient(provider="openai", model="gpt-4", api_key="sk-test")

    # Initially, pool clients are None
    assert client._sync_client is None
    assert client._async_client is None

    # Retrieve sync client
    sync_client = client._get_sync_client()
    assert isinstance(sync_client, httpx.Client)
    assert client._sync_client is sync_client

    # Repeated calls should return the same instance (pooling)
    assert client._get_sync_client() is sync_client

    # Retrieve async client
    async_client = client._get_async_client()
    assert isinstance(async_client, httpx.AsyncClient)
    assert client._async_client is async_client
    assert client._get_async_client() is async_client

    # Test closing
    client.close()
    assert client._sync_client is None
    # async client is still there since close() only closes sync client
    assert client._async_client is async_client

    # Test async close
    asyncio.run(client.aclose())
    assert client._async_client is None


def test_json_repair_fallback_and_direct():
    from dspy_transpiler.compiler import repair_and_parse_json

    # Test direct parsing of valid json
    assert repair_and_parse_json('{"foo": "bar"}') == {"foo": "bar"}

    # Test json-repair on broken json (e.g., missing closing brace)
    assert repair_and_parse_json('{"foo": "bar"') == {"foo": "bar"}

    # Test json-repair on broken list
    assert repair_and_parse_json("[1, 2, 3") == [1, 2, 3]

    # Test json-repair with markdown fences and text around it
    assert repair_and_parse_json('some text ```json\n{"foo": "bar"}\n``` other text') == {
        "foo": "bar"
    }


def test_transpiled_program_prediction_return(monkeypatch):
    import dspy
    from pydantic import BaseModel

    from dspy_transpiler.compiler import AgentTranspiler
    from dspy_transpiler.graph import Graph, StatefulNode

    class CompilerTestInput(BaseModel):
        input_text: str

    class CompilerTestOutput(BaseModel):
        output_text: str

    node_a = StatefulNode(
        name="NodeA", input_model=CompilerTestInput, output_model=CompilerTestOutput
    )
    graph = Graph()
    graph.add_node(node_a)
    graph.set_entry_point("NodeA")
    program = AgentTranspiler.compile(graph)

    class MockOutput:
        output_text = "test_ok"

    monkeypatch.setattr(program, "predictor_NodeA", lambda **k: MockOutput())

    # Verify that program returns a dspy.Prediction and handles legacy keywords
    res = program(input_text="hello", max_retries=3, max_steps=5)
    assert isinstance(res, dspy.Prediction)
    assert res.output_text == "test_ok"
    assert res["_metadata"]["step_count"] == 1


def test_direct_client_transient_retry_success(monkeypatch):
    import httpx

    from dspy_transpiler.compiler import DirectClient

    client = DirectClient(
        provider="openai",
        model="gpt-4",
        api_key="sk-test",
        max_network_retries=2,
        base_delay=0.01,
    )

    call_count = 0

    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code != 200:
                req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                res = httpx.Response(self.status_code, request=req)
                raise httpx.HTTPStatusError("Transient Error", request=req, response=res)

        def json(self):
            return {"choices": [{"message": {"content": "Retry Success"}}]}

    def mock_post(url, headers, content):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse(429)
        return MockResponse(200)

    # Setup the sync client pool first
    sync_client = client._get_sync_client()
    monkeypatch.setattr(sync_client, "post", mock_post)

    res = client.generate_sync("hello")
    assert res == "Retry Success"
    assert call_count == 2


def test_direct_client_retry_exhaustion(monkeypatch):
    import httpx
    import pytest

    from dspy_transpiler.compiler import DirectClient

    client = DirectClient(
        provider="openai",
        model="gpt-4",
        api_key="sk-test",
        max_network_retries=1,
        base_delay=0.01,
    )

    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            res = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("Transient Failure", request=req, response=res)

    sync_client = client._get_sync_client()
    monkeypatch.setattr(sync_client, "post", lambda *a, **k: MockResponse(500))

    with pytest.raises(httpx.HTTPStatusError):
        client.generate_sync("hello")


def test_telemetry_span_otel_integration(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    # Mock opentelemetry modules in sys.modules
    mock_trace = MagicMock()
    sys.modules["opentelemetry"] = MagicMock()
    sys.modules["opentelemetry.trace"] = mock_trace

    import dspy_transpiler.telemetry as tel
    from dspy_transpiler import telemetry

    monkeypatch.setattr(tel, "HAS_OTEL", True)

    class MockSpan:
        def __init__(self):
            self.attributes = {}
            self.status_code = None
            self.description = None
            self.exceptions = []

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status_code = status

        def record_exception(self, e):
            self.exceptions.append(e)

        def end(self):
            pass

    class MockTracer:
        def start_span(self, name, context=None):
            return MockSpan()

    monkeypatch.setattr(tel, "otel_tracer", MockTracer())

    with telemetry.trace_span("test_node", {"input_key": "input_val"}) as span:
        assert span.otel_span is not None
        span.set_attribute("custom_attr", "custom_val")
        span.set_status("ERROR", "test failure")

    assert span.otel_span.attributes["input.input_key"] == "input_val"
    assert span.otel_span.attributes["custom_attr"] == "custom_val"
    assert span.otel_span.status_code is not None

    # Clean up sys.modules after test
    sys.modules.pop("opentelemetry", None)
    sys.modules.pop("opentelemetry.trace", None)
